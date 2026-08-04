"""
MLP next-char LM (NPLM-style, no attention) — leak-free CCA benchmark.

The model sees a window of BLOCK chars and predicts ONLY the next char, so
mean-pooling the window is strictly causal (all inputs precede the target).
Without attention, context-conditioned activation is the only mechanism for
conditional computation -> the honest test of CCA.

Env: MIX=1 NCIPH=k ITERS=n GROUPS=g
"""

import math
import os
import time

import torch
import torch.nn as nn

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB = int(os.environ.get("EMB", 32))
BLOCK = 64
HID = int(os.environ.get("HID", 512))
BATCH = 256
ITERS = int(os.environ.get("ITERS", 2000))
EVAL_ITERS = 100
LR = float(os.environ.get("LR", 1e-3))
GROUPS = int(os.environ.get("CCA_GROUPS", 64))   # note: name "GROUPS" is stripped by the sandbox
ALPHA = float(os.environ.get("ALPHA", 1.0))
GAMMA = float(os.environ.get("GAMMA", 2.0))


# ---------------- data (regime mixture) ---------------- #
def rot13(t):
    f = lambda ch: (chr((ord(ch) - 97 + 13) % 26 + 97) if "a" <= ch <= "z"
                    else chr((ord(ch) - 65 + 13) % 26 + 65) if "A" <= ch <= "Z"
                    else ch)
    return "".join(f(c) for c in t)


def atbash(t):
    f = lambda ch: (chr(122 - (ord(ch) - 97)) if "a" <= ch <= "z"
                    else chr(90 - (ord(ch) - 65)) if "A" <= ch <= "Z" else ch)
    return "".join(f(c) for c in t)


def make_cipher(seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(26, generator=g).tolist()
    lo = {chr(97 + i): chr(97 + perm[i]) for i in range(26)}
    up = {chr(65 + i): chr(65 + perm[i]) for i in range(26)}
    table = str.maketrans({**lo, **up})
    return lambda t: t.translate(table)


def build_regimes():
    regs = [lambda t: t, rot13, atbash, lambda t: t.swapcase(),
            lambda t: t[::-1]]
    regs += [make_cipher(1000 + k) for k in range(int(os.environ.get("NCIPH", 0)))]
    return regs


class Data:
    def __init__(self, mix):
        with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
        texts = [f(text) for f in build_regimes()] if mix else [text]
        chars = sorted(set("".join(texts)))
        self.vocab = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        self.train, self.val = [], []
        for t in texts:
            d = torch.tensor([stoi[c] for c in t], dtype=torch.long)
            n = int(0.9 * len(d))
            self.train.append(d[:n])
            self.val.append(d[n:])

    def get_batch(self, split, gen):
        pools = self.train if split == "train" else self.val
        xs, ys = [], []
        regs = torch.randint(len(pools), (BATCH,), generator=gen)
        for r in regs.tolist():
            d = pools[r]
            i = int(torch.randint(len(d) - BLOCK - 1, (1,), generator=gen))
            xs.append(d[i:i + BLOCK])
            ys.append(d[i + BLOCK])
        return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


# ---------------- CCA (2D variant) ---------------- #
class CCA(nn.Module):
    def __init__(self, ctx_dim, feat_dim, groups=GROUPS):
        super().__init__()
        assert feat_dim % groups == 0
        self.groups, self.feat_dim = groups, feat_dim
        self.G = nn.Sequential(nn.Linear(ctx_dim, 64), nn.SiLU(),
                               nn.Linear(64, 5 * groups))
        nn.init.zeros_(self.G[-1].weight)
        nn.init.zeros_(self.G[-1].bias)
        self.theta_0 = nn.Parameter(torch.zeros(5 * groups),
                                    requires_grad=os.environ.get("FREEZE_T0", "0") != "1")
        self.alpha = nn.Parameter(torch.full((groups,), ALPHA),
                                  requires_grad=os.environ.get("FREEZE_A", "0") != "1")
        self.gamma = GAMMA
        self.shuffle_ctx = False
        self.collect = False
        self._buf = []

    def reset_stats(self):
        self._buf = []

    def coeff_variance(self):
        if not self._buf:
            return None
        c = torch.cat(self._buf, 0)
        return c.var(dim=0, unbiased=False).mean(dim=1).cpu()

    def forward(self, x, c):
        # x [B, F], c [B, C]
        B, Fdim = x.shape
        if self.shuffle_ctx:
            c = c[torch.randperm(B, device=c.device)]
        theta = self.theta_0 + self.gamma * torch.tanh(self.G(c))
        theta = theta.view(B, 5, self.groups)
        if self.collect:
            self._buf.append(theta.detach())
        rep = Fdim // self.groups
        a0, a1, a2, b0, b1 = (theta[:, i, :].repeat_interleave(rep, dim=-1)
                              for i in range(5))
        alpha = self.alpha.repeat_interleave(rep).view(1, Fdim)
        silu = x * torch.sigmoid(x)
        return silu + alpha * (a0 + a1 * x + a2 * x * x) / (1 + (b0 + b1 * x) ** 2)


# ---------------- models ---------------- #
class MLPLM(nn.Module):
    """emb -> flatten -> fc1 -> act -> fc2 -> act -> head"""

    def __init__(self, vocab, mode):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(vocab, EMB)
        self.fc1 = nn.Linear(BLOCK * EMB, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.head = nn.Linear(HID, vocab)
        if mode == "cca":
            self.act1 = CCA(EMB, HID)
            self.act2 = CCA(EMB, HID)
        else:
            self.act1 = self.act2 = nn.SiLU()

    def cca_layers(self):
        return [m for m in (self.act1, self.act2) if isinstance(m, CCA)]

    def set_cca(self, **kw):
        for a in self.cca_layers():
            for k, v in kw.items():
                setattr(a, k, v)
            if kw.get("collect"):
                a.reset_stats()

    def forward(self, idx):
        e = self.emb(idx)                       # [B, BLOCK, EMB]
        c = e.mean(dim=1)                       # causal: window precedes target
        h = self.fc1(e.flatten(1))
        h = self.act1(h, c) if self.mode == "cca" else self.act1(h)
        h = self.fc2(h)
        h = self.act2(h, c) if self.mode == "cca" else self.act2(h)
        return self.head(h)


# ---------------- train / eval ---------------- #
@torch.no_grad()
def evaluate(model, data, *, shuffle_ctx=False, collect=False):
    model.eval()
    model.set_cca(collect=collect, shuffle_ctx=shuffle_ctx)
    gen = torch.Generator().manual_seed(SEED)
    crit = nn.CrossEntropyLoss()
    tot, n_tok = 0.0, 0
    t0 = time.time()
    for _ in range(EVAL_ITERS):
        x, y = data.get_batch("val", gen)
        tot += crit(model(x), y).item()
        n_tok += x.numel()
    dt = time.time() - t0
    model.set_cca(collect=False, shuffle_ctx=False)
    vl = tot / EVAL_ITERS
    var = None
    if collect:
        vs = [a.coeff_variance() for a in model.cca_layers()]
        vs = [v for v in vs if v is not None]
        if vs:
            var = torch.stack(vs).mean(0)
    return {"val_loss": vl, "ppl": math.exp(vl),
            "throughput": n_tok / dt, "coeff_var": var}


def train(mode, data):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    model = MLPLM(data.vocab, mode).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    glr = float(os.environ.get("GLR", 1.0))     # LR multiplier for generator G
    g_params, other = [], []
    for n, p in model.named_parameters():
        (g_params if ".G." in n else other).append(p)
    opt = torch.optim.AdamW(
        [{"params": other, "lr": LR},
         {"params": g_params, "lr": LR * glr}])
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    model.train()
    for it in range(ITERS):
        x, y = data.get_batch("train", gen)
        loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    tt = time.time() - t0
    return model, tt, n_params


def main():
    mix = os.environ.get("MIX", "1") == "1"
    data = Data(mix)
    print(f"mixture={mix} regimes={len(data.train)} vocab={data.vocab} "
          f"iters={ITERS}")

    results = {}
    for mode in ("silu", "cca"):
        model, tt, np_ = train(mode, data)
        ev = evaluate(model, data, collect=(mode == "cca"))
        results[mode] = (model, ev, tt, np_)
        print(f"{mode:5s} | loss {ev['val_loss']:.4f} | ppl {ev['ppl']:7.3f} "
              f"| params {np_/1e6:.2f}M | {tt:.1f}s")

    sh = evaluate(results["cca"][0], data, shuffle_ctx=True)
    print(f"shuf  | loss {sh['val_loss']:.4f} | ppl {sh['ppl']:7.3f}")

    b, c = results["silu"][1], results["cca"][1]
    gl = 100 * (b["val_loss"] - c["val_loss"]) / b["val_loss"]
    gp = 100 * (b["ppl"] - c["ppl"]) / b["ppl"]
    print(f"\ngap: loss -{gl:.2f}%  ppl -{gp:.2f}%")
    if c["coeff_var"] is not None:
        print(f"coeff var (mean): {float(c['coeff_var'].mean()):.4e}")


if __name__ == "__main__":
    main()

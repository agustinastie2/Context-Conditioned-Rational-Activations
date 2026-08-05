"""
Variant sweep to maximize the CCA-vs-baseline gap on Tiny Shakespeare char-LM.
Each run ~12-20s. Same seed, same iters for every variant.
"""

import math
import os
import sys
import time
import hashlib
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = int(os.environ.get("DIM", 128))
N_LAYERS = int(os.environ.get("LAYERS", 3))
N_HEADS = int(os.environ.get("HEADS", 4))
BLOCK_SIZE = int(os.environ.get("BLOCK", 64))
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_ITERS = int(os.environ.get("ITERS", 500))
EVAL_ITERS = 50
LR = 3e-3
HIST_LEN = int(os.environ.get("HIST_LEN", 192))


def load_text():
    """Load the checksum-pinned Tiny Shakespeare corpus over verified HTTPS."""
    path = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"
    expected = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
               "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/"
               "data/tinyshakespeare/input.txt")
        urllib.request.urlretrieve(url, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Tiny Shakespeare checksum mismatch: {digest}")
        temporary.replace(path)
    return path.read_text(encoding="utf-8")


def rot13(t):
    def f(ch):
        if "a" <= ch <= "z":
            return chr((ord(ch) - 97 + 13) % 26 + 97)
        if "A" <= ch <= "Z":
            return chr((ord(ch) - 65 + 13) % 26 + 65)
        return ch
    return "".join(f(c) for c in t)


def atbash(t):
    def f(ch):
        if "a" <= ch <= "z":
            return chr(122 - (ord(ch) - 97))
        if "A" <= ch <= "Z":
            return chr(90 - (ord(ch) - 65))
        return ch
    return "".join(f(c) for c in t)


REGIMES = [
    lambda t: t,                  # identity
    rot13,                        # ROT13 cipher
    atbash,                       # Atbash cipher
    lambda t: t.swapcase(),       # case swap
    lambda t: t[::-1],            # reversed text
]


def make_cipher(seed):
    """Random lowercase+uppercase substitution cipher (vocab-preserving)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(26, generator=g).tolist()
    lo = {chr(97 + i): chr(97 + perm[i]) for i in range(26)}
    up = {chr(65 + i): chr(65 + perm[i]) for i in range(26)}
    table = str.maketrans({**lo, **up})
    return lambda t: t.translate(table)


N_CIPH = int(os.environ.get("NCIPH", 0))
REGIMES = REGIMES + [make_cipher(1000 + k) for k in range(N_CIPH)]


class CharDataset:
    """Single-domain (MIX=0) or multi-regime mixture (MIX=1) char dataset."""

    def __init__(self, text, mix=False):
        self.mix = mix
        texts = [f(text) for f in REGIMES] if mix else [text]
        chars = sorted(set("".join(t for t in texts)))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.train, self.val = [], []
        for t in texts:
            data = torch.tensor([self.stoi[c] for c in t], dtype=torch.long)
            n = int(0.9 * len(data))
            self.train.append(data[:n])
            self.val.append(data[n:])

    def get_batch(self, split, gen):
        pools = self.train if split == "train" else self.val
        xs, ys, hs = [], [], []
        regs = torch.randint(len(pools), (BATCH_SIZE,), generator=gen)
        for r in regs.tolist():
            data = pools[r]
            i = int(torch.randint(
                HIST_LEN, len(data) - BLOCK_SIZE, (1,), generator=gen))
            xs.append(data[i:i + BLOCK_SIZE])
            ys.append(data[i + 1:i + 1 + BLOCK_SIZE])
            hs.append(data[i - HIST_LEN:i])      # strictly-past history
        return (torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE),
                torch.stack(hs).to(DEVICE))


class CCA(nn.Module):
    """
    A(x,c) = silu(x) + alpha * (a0 + a1 x + a2 x^2) / (1 + (b0 + b1 x)^2)
    Variants:
      causal:   context = causal cumulative mean (per token) vs sequence mean
      groups:   coefficients per channel-group (G groups over feat dim)
      learn_a:  alpha learnable per group
    """

    def __init__(self, ctx_dim, feat_dim, *, causal=False, groups=1,
                 alpha=0.1, gamma=0.5, learn_alpha=False, deep_g=False,
                 ctx_mode=None):
        super().__init__()
        self.variant = os.environ.get("CCA_VARIANT", "full").lower()
        self.hinge = self.variant in ("hinge", "cca_hinge")
        self.external = self.variant in ("lite", "cca_lite", "rational",
                                         "dyrelu", "meta_acon")
        if self.hinge:
            groups = 16
        # ctx_mode: "mean" (full-seq, leaky), "cummean" (causal),
        #           "token" (c_t = H_t), "tokcum" (concat token + cummean)
        self.ctx_mode = ctx_mode or ("cummean" if causal else "mean")
        if self.ctx_mode == "tokcum":
            ctx_dim *= 2
        if self.external:
            from adaptive_activations import ActivationConfig, build_activation
            method = {"lite": "cca_lite"}.get(self.variant, self.variant)
            groups = int(os.environ.get("ACT_GROUPS", "16"))
            hidden = int(os.environ.get("ACT_HIDDEN", "16"))
            self.activation = build_activation(
                ActivationConfig(method, groups, hidden), ctx_dim, feat_dim)
            self.groups, self.feat_dim = groups, feat_dim
            self.shuffle_ctx = False
            self.collect = False
            self._buf = []
            return
        assert feat_dim % groups == 0
        self.groups, self.feat_dim = groups, feat_dim
        self.gamma = gamma
        n_out = 9 if self.hinge else 5
        if deep_g:
            # two-layer generator; last layer zero-init keeps exact-SiLU start
            generator_hidden = 16 if self.hinge else 64
            self.G = nn.Sequential(
                nn.Linear(ctx_dim, generator_hidden), nn.SiLU(),
                nn.Linear(generator_hidden, n_out * groups))
            nn.init.zeros_(self.G[-1].weight)
            nn.init.zeros_(self.G[-1].bias)
        else:
            self.G = nn.Linear(ctx_dim, n_out * groups)
            nn.init.zeros_(self.G.weight)
            nn.init.zeros_(self.G.bias)
        self.theta_0 = nn.Parameter(torch.zeros(5 * groups))
        if learn_alpha:
            self.alpha = nn.Parameter(torch.full((groups,), alpha))
        else:
            self.register_buffer("alpha", torch.full((groups,), alpha))
        self.shuffle_ctx = False
        self.collect = False
        self._buf = []

    def reset_stats(self):
        self._buf = []
        if self.external and hasattr(self.activation, "reset_stats"):
            self.activation.reset_stats()

    def coeff_variance(self):
        if self.external:
            return (self.activation.coeff_variance()
                    if hasattr(self.activation, "coeff_variance") else None)
        if not self._buf:
            return None
        c = torch.cat(self._buf, 0)             # [N, 5, G]
        return c.var(dim=0, unbiased=False).mean(dim=1).cpu()   # [5]

    def forward(self, x, H):
        # x [B,S,F]  H [B,S,C] (pre-MLP hidden states)
        B, S, Fdim = x.shape
        if self.ctx_mode == "hist":
            c = H               # H is already the encoded history, broadcast
        elif self.ctx_mode == "cummean":
            counts = torch.arange(1, S + 1, device=H.device).view(1, S, 1)
            c = H.cumsum(dim=1) / counts                       # [B,S,C]
        elif self.ctx_mode == "token":
            c = H
        elif self.ctx_mode == "tokcum":
            counts = torch.arange(1, S + 1, device=H.device).view(1, S, 1)
            c = torch.cat([H, H.cumsum(dim=1) / counts], dim=-1)
        else:                                                  # "mean" (leaky)
            c = H.mean(dim=1, keepdim=True).expand(B, S, H.size(-1))
        if self.shuffle_ctx:
            perm = torch.randperm(B, device=c.device)
            c = c[perm]
        if self.external:
            if hasattr(self.activation, "collect"):
                self.activation.collect = self.collect
            return self.activation(x, c)
        generated = torch.tanh(self.G(c))
        if self.hinge:
            generated = generated.view(B, S, 9, self.groups)
            theta = (self.theta_0.view(1, 1, 5, self.groups)
                     + self.gamma * generated[:, :, :5])
        else:
            theta = self.theta_0 + self.gamma * generated  # [B,S,5G]
            theta = theta.view(B, S, 5, self.groups)
        if self.collect:
            self._buf.append(theta.detach().reshape(-1, 5, self.groups))
        rep = Fdim // self.groups
        a0, a1, a2, b0, b1 = (
            theta[:, :, i, :].repeat_interleave(rep, dim=-1) for i in range(5)
        )
        alpha = self.alpha.repeat_interleave(rep).view(1, 1, Fdim)
        if self.hinge:
            w = 1.0 + generated[:, :, 5].repeat_interleave(rep, dim=-1)
            d = generated[:, :, 6].repeat_interleave(rep, dim=-1)
            amplitude = generated[:, :, 7].repeat_interleave(rep, dim=-1)
            threshold = generated[:, :, 8].repeat_interleave(rep, dim=-1)
            u = w * x + d
            hinge = amplitude * torch.sqrt((u - threshold).square() + 1e-4)
        else:
            u, hinge = x, 0.0
        silu = u * torch.sigmoid(u)
        return silu + alpha * (a0 + a1 * u + a2 * u * u) / (
            1 + (b0 + b1 * u) ** 2) + hinge


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, S, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // N_HEADS
        q, k, v = (t.view(B, S, N_HEADS, hd).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=DROPOUT if self.training else 0.0)
        return self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, S, C)))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM, 4 * EMBED_DIM)
        self.fc2 = nn.Linear(4 * EMBED_DIM, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.cfg = cfg
        if cfg["act"] == "cca":
            self.act = CCA(EMBED_DIM, 4 * EMBED_DIM,
                           causal=cfg.get("causal", False),
                           groups=cfg.get("groups", 1),
                           alpha=cfg.get("alpha", 0.1),
                           gamma=cfg.get("gamma", 0.5),
                           learn_alpha=cfg.get("learn_alpha", False),
                           deep_g=cfg.get("deep_g", False),
                           ctx_mode=cfg.get("ctx"))
        else:
            self.act = nn.SiLU()

    def forward(self, x, hctx=None):
        h = self.fc1(x)
        if self.cfg["act"] == "cca":
            H = hctx if (getattr(self.act, "ctx_mode", None) == "hist"
                         and hctx is not None) else x
            h = self.act(h, H)
        else:
            h = self.act(h)
        return self.drop(self.fc2(h))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(EMBED_DIM), nn.LayerNorm(EMBED_DIM)
        self.attn, self.mlp = Attn(), MLP(cfg)

    def forward(self, x, hctx=None):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x), hctx)


class Model(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        self.use_hist = cfg.get("ctx") == "hist" or cfg.get("hist_add", False)
        self.hist_add = cfg.get("hist_add", False)
        self.tok = nn.Embedding(vocab, EMBED_DIM)
        self.pos = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab, bias=False)
        self.head.weight = self.tok.weight
        if self.use_hist:
            self.hist_enc = nn.Sequential(
                nn.Linear(EMBED_DIM, EMBED_DIM), nn.SiLU())
        self.apply(self._init)
        # re-zero CCA generators after global init (exact SiLU at start)
        for a in self.cca_layers():
            if hasattr(a, "G"):
                last = a.G[-1] if isinstance(a.G, nn.Sequential) else a.G
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            elif getattr(a, "external", False):
                activation = a.activation
                if hasattr(activation, "G"):
                    nn.init.zeros_(activation.G[-1].weight)
                    nn.init.zeros_(activation.G[-1].bias)
                elif hasattr(activation, "generator"):
                    nn.init.zeros_(activation.generator.net[-1].weight)
                    nn.init.zeros_(activation.generator.net[-1].bias)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def cca_layers(self):
        return [b.mlp.act for b in self.blocks if isinstance(b.mlp.act, CCA)]

    def set_cca(self, **kw):
        for a in self.cca_layers():
            for k, v in kw.items():
                setattr(a, k, v)
            if kw.get("collect"):
                a.reset_stats()

    def forward(self, idx, targets=None, hist=None):
        B, S = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(S, device=idx.device)))
        hctx = None
        if self.use_hist and hist is not None:
            # mean-pooled embedding of strictly-past history, broadcast [B,S,C]
            hc = self.hist_enc(self.tok(hist).mean(dim=1))
            if self.hist_add:           # control: inject via input instead
                x = x + hc.unsqueeze(1)
            else:
                hctx = hc.unsqueeze(1).expand(B, S, -1)
        for b in self.blocks:
            x = b(x, hctx)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@torch.no_grad()
def evaluate(model, ds, *, shuffle_ctx=False, collect=False):
    model.eval()
    model.set_cca(collect=collect, shuffle_ctx=shuffle_ctx)
    gen = torch.Generator().manual_seed(SEED)
    losses, n_tok = [], 0
    t0 = time.time()
    for _ in range(EVAL_ITERS):
        x, y, h = ds.get_batch("val", gen)
        _, loss = model(x, y, h)
        losses.append(loss.item())
        n_tok += x.numel()
    dt = time.time() - t0
    model.set_cca(collect=False, shuffle_ctx=False)
    vl = sum(losses) / len(losses)
    var = None
    if collect:
        vs = [a.coeff_variance() for a in model.cca_layers()]
        vs = [v for v in vs if v is not None]
        if vs:
            var = torch.stack(vs).mean(0)
    return {"val_loss": vl, "ppl": math.exp(vl), "tok_s": n_tok / dt, "var": var}


def run(name, cfg, ds):
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    model = Model(ds.vocab_size, cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    if os.environ.get("PAPER_COMPILE", "0") == "1" and hasattr(torch, "compile"):
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model, dynamic=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    t0 = time.time()
    model.train()
    for it in range(MAX_ITERS):
        x, y, h = ds.get_batch("train", gen)
        _, loss = model(x, y, h)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    tt = time.time() - t0
    ev = evaluate(model, ds, collect=bool(model.cca_layers()))
    sh = evaluate(model, ds, shuffle_ctx=True) if model.cca_layers() else None
    print(f"{name:28s} | loss {ev['val_loss']:.4f} | ppl {ev['ppl']:6.3f} | "
          f"params {n_params/1e6:.2f}M | {tt:5.1f}s"
          + (f" | shuf {sh['val_loss']:.4f}" if sh else ""))
    return model, ev, sh, n_params, tt


def main():
    mix = os.environ.get("MIX", "0") == "1"
    ds = CharDataset(load_text(), mix=mix)
    print(f"mixture={mix}  vocab={ds.vocab_size}")
    variants = {
        "B0-silu": {"act": "silu"},
        "V1-orig": {"act": "cca"},
        "V2-causal": {"act": "cca", "causal": True},
        "V3-causal-g16": {"act": "cca", "causal": True, "groups": 16},
        "V4-causal-g16-lalpha": {"act": "cca", "causal": True, "groups": 16,
                                 "learn_alpha": True},
        "V5-causal-g64-lalpha": {"act": "cca", "causal": True, "groups": 64,
                                 "learn_alpha": True},
        "V6-causal-g16-a05": {"act": "cca", "causal": True, "groups": 16,
                              "learn_alpha": True, "alpha": 0.5, "gamma": 1.0},
        "V7-causal-g64-deep": {"act": "cca", "causal": True, "groups": 64,
                               "learn_alpha": True, "deep_g": True},
        "V8-causal-g64-deep-a05": {"act": "cca", "causal": True, "groups": 64,
                                   "learn_alpha": True, "deep_g": True,
                                   "alpha": 0.5, "gamma": 1.0},
        "V9-mean-g64-deep-a05": {"act": "cca", "causal": False, "groups": 64,
                                 "learn_alpha": True, "deep_g": True,
                                 "alpha": 0.5, "gamma": 1.0},
        "V10-causal-g512-deep-a05": {"act": "cca", "causal": True, "groups": 512,
                                     "learn_alpha": True, "deep_g": True,
                                     "alpha": 0.5, "gamma": 1.0},
        "V11-causal-g64-deep-a1g2": {"act": "cca", "causal": True, "groups": 64,
                                     "learn_alpha": True, "deep_g": True,
                                     "alpha": 1.0, "gamma": 2.0},
        "V12-causal-g128-deep-a1": {"act": "cca", "causal": True, "groups": 128,
                                    "learn_alpha": True, "deep_g": True,
                                    "alpha": 1.0, "gamma": 1.0},
        "V13-mean-g64-deep-a1g2": {"act": "cca", "causal": False, "groups": 64,
                                   "learn_alpha": True, "deep_g": True,
                                   "alpha": 1.0, "gamma": 2.0},
        "V14-mean-g128-deep-a05": {"act": "cca", "causal": False, "groups": 128,
                                   "learn_alpha": True, "deep_g": True,
                                   "alpha": 0.5, "gamma": 1.0},
        "V15-mean-g64-deep-a05-g2": {"act": "cca", "causal": False, "groups": 64,
                                     "learn_alpha": True, "deep_g": True,
                                     "alpha": 0.5, "gamma": 2.0},
        "V16-mean-g64-deep-a2g2": {"act": "cca", "causal": False, "groups": 64,
                                   "learn_alpha": True, "deep_g": True,
                                   "alpha": 2.0, "gamma": 2.0},
        "V17-mean-g128-deep-a1g2": {"act": "cca", "causal": False, "groups": 128,
                                    "learn_alpha": True, "deep_g": True,
                                    "alpha": 1.0, "gamma": 2.0},
        "V18-mean-g64-deep-a1g4": {"act": "cca", "causal": False, "groups": 64,
                                   "learn_alpha": True, "deep_g": True,
                                   "alpha": 1.0, "gamma": 4.0},
        # --- causal (no-leak) round: per-token context ---
        "T1-token-g64": {"act": "cca", "ctx": "token", "groups": 64,
                         "learn_alpha": True, "deep_g": True,
                         "alpha": 1.0, "gamma": 2.0},
        "T2-tokcum-g64": {"act": "cca", "ctx": "tokcum", "groups": 64,
                          "learn_alpha": True, "deep_g": True,
                          "alpha": 1.0, "gamma": 2.0},
        "T3-token-g512": {"act": "cca", "ctx": "token", "groups": 512,
                          "learn_alpha": True, "deep_g": True,
                          "alpha": 1.0, "gamma": 2.0},
        "T4-tokcum-g64-a05g1": {"act": "cca", "ctx": "tokcum", "groups": 64,
                                "learn_alpha": True, "deep_g": True,
                                "alpha": 0.5, "gamma": 1.0},
        # --- multi-regime mixture round (run with MIX=1) ---
        "M1-cummean-g64": {"act": "cca", "ctx": "cummean", "groups": 64,
                           "learn_alpha": True, "deep_g": True,
                           "alpha": 1.0, "gamma": 2.0},
        "M2-token-g64": {"act": "cca", "ctx": "token", "groups": 64,
                         "learn_alpha": True, "deep_g": True,
                         "alpha": 1.0, "gamma": 2.0},
        "M3-tokcum-g64": {"act": "cca", "ctx": "tokcum", "groups": 64,
                          "learn_alpha": True, "deep_g": True,
                          "alpha": 1.0, "gamma": 2.0},
        # --- history context (strictly past, beyond attention window) ---
        "H1-hist-g64": {"act": "cca", "ctx": "hist", "groups": 64,
                        "learn_alpha": True, "deep_g": True,
                        "alpha": 1.0, "gamma": 2.0},
        "B0H-silu-histadd": {"act": "silu", "hist_add": True},
    }
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = {}
    for name, cfg in variants.items():
        if only and name not in only:
            continue
        results[name] = run(name, cfg, ds)

    if "B0-silu" in results:
        b = results["B0-silu"][1]["val_loss"]
        print(f"\nbaseline loss {b:.4f} — gap vs baseline:")
        for n, (m, ev, sh, _, _) in results.items():
            if n == "B0-silu":
                continue
            g = 100 * (b - ev["val_loss"]) / b
            gp = 100 * (math.exp(b) - ev["ppl"]) / math.exp(b)
            print(f"  {n:28s} loss -{g:5.2f}%  ppl -{gp:5.2f}%")


if __name__ == "__main__":
    main()

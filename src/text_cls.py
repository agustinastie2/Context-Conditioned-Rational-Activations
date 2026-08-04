"""
Text classification — multi-regime ciphered topic classification.
=================================================================

We generate text from C "content classes" (each a distinct first-order Markov
source over a 27-symbol alphabet) and then encipher every sample with one of R
substitution ciphers (random alphabet permutations). The label is the content
class; the cipher is a nuisance regime. To classify content the model must
implicitly undo the cipher, which depends on identifying the regime — readable
from the unigram-frequency signature (the mean embedding). A single fixed
activation must serve all R*C combinations; CCA conditions on the pooled
context (which reveals the regime) and routes its nonlinearity accordingly.

Both models see the whole window, so the CCA gain is conditional-computation
capacity, mirroring the image-classification setup.

Controls: single-regime null (R=1) + context-shuffle ablation.
Writes results/text_cls.json (+_single).   Run: python text_cls.py
"""

import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import metrics
from cca_core import CCA1D, set_cca, mean_coeff_variance

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 27          # 26 letters + space
N_CLASSES = 4       # content classes
LEN = 48            # sample length
EMB = 32
HID = 256
BATCH = 256
STEPS = int(os.environ.get("STEPS", 1500))
LR = 2e-3
N_TRAIN = 40000
N_TEST = 8000
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1337,1338,1339").split(",")]
os.makedirs("results", exist_ok=True)


def mstd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


# --------------------------------------------------------------------------- #
#  Synthetic multi-regime data
# --------------------------------------------------------------------------- #
def make_sources(n, seed):
    """n first-order Markov transition matrices over VOCAB symbols."""
    g = torch.Generator().manual_seed(seed)
    mats = []
    for _ in range(n):
        # peaky transitions so classes are distinguishable
        logits = torch.randn(VOCAB, VOCAB, generator=g) * 2.5
        mats.append(torch.softmax(logits, dim=1))
    return mats


def make_ciphers(r, seed):
    g = torch.Generator().manual_seed(seed)
    perms = []
    for _ in range(r):
        p = torch.arange(VOCAB)
        p[:26] = torch.randperm(26, generator=g)   # keep space (idx 26) fixed
        perms.append(p)
    return perms


def gen_dataset(n, sources, ciphers, seed):
    g = torch.Generator().manual_seed(seed)
    C, R = len(sources), len(ciphers)
    labels = torch.randint(C, (n,), generator=g)
    regs = torch.randint(R, (n,), generator=g)
    X = torch.empty(n, LEN, dtype=torch.long)
    cur = torch.randint(VOCAB, (n,), generator=g)
    src = torch.stack(sources)            # [C, V, V]
    for t in range(LEN):
        probs = src[labels, cur]          # [n, V]
        cur = torch.multinomial(probs, 1, generator=g).squeeze(1)
        X[:, t] = cur
    # apply per-sample cipher
    ciph = torch.stack(ciphers)           # [R, V]
    X = ciph[regs].gather(1, X)           # remap symbols
    return X, labels


# --------------------------------------------------------------------------- #
#  Model: bag-of-embeddings MLP (mean embedding reveals the regime)
# --------------------------------------------------------------------------- #
class BoEClassifier(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(VOCAB, EMB)
        self.fc1 = nn.Linear(EMB, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.head = nn.Linear(HID, N_CLASSES)
        if mode == "cca":
            self.a1, self.a2 = CCA1D(EMB, HID), CCA1D(EMB, HID)
        else:
            self.a1 = self.a2 = nn.SiLU()

    def forward(self, idx):
        e = self.emb(idx)             # [B, L, EMB]
        ctx = e.mean(1)               # pooled context (unigram signature)
        h = self.fc1(ctx)
        h = self.a1(h, ctx) if self.mode == "cca" else self.a1(h)
        h = self.fc2(h)
        h = self.a2(h, ctx) if self.mode == "cca" else self.a2(h)
        return self.head(h)


def batches(X, y, gen, train):
    order = torch.randperm(len(X), generator=gen) if train else torch.arange(len(X))
    for i in range(0, len(X), BATCH):
        idx = order[i:i + BATCH]
        yield X[idx].to(DEVICE), y[idx].to(DEVICE)


@torch.no_grad()
def evaluate(model, Xte, yte, *, shuffle_ctx=False, collect=False):
    model.eval()
    set_cca(model, collect=collect, shuffle_ctx=shuffle_ctx)
    gen = torch.Generator().manual_seed(SEED)
    correct, total, t0 = 0, 0, time.time()
    for xb, yb in batches(Xte, yte, gen, False):
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    dt = time.time() - t0
    set_cca(model, collect=False, shuffle_ctx=False)
    return {"acc": correct / total, "throughput": total * LEN / dt,
            "coeff_var": (mean_coeff_variance(model) if collect else None)}


def train(mode, data):
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(SEED)
    model = BoEClassifier(mode).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    t0, step = time.time(), 0
    while step < STEPS:
        for xb, yb in batches(Xtr, ytr, gen, True):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= STEPS:
                break
    return model, time.time() - t0, n_par


def run(R, tag):
    global SEED
    raw = {"silu": [], "cca": [], "shuffle": []}
    coeff_vars, n_par_ref = [], {}
    for sd in SEEDS:
        SEED = sd
        sources = make_sources(N_CLASSES, sd)
        ciphers = make_ciphers(R, sd + 1)
        Xtr, ytr = gen_dataset(N_TRAIN, sources, ciphers, sd)
        Xte, yte = gen_dataset(N_TEST, sources, ciphers, sd + 99)
        data = (Xtr, ytr, Xte, yte)
        models = {}
        for mode in ("silu", "cca"):
            model, tt, n_par = train(mode, data)
            ev = evaluate(model, Xte, yte, collect=(mode == "cca"))
            models[mode] = model
            n_par_ref[mode] = n_par
            raw[mode].append(ev["acc"])
            if ev["coeff_var"] is not None:
                coeff_vars.append(ev["coeff_var"])
        sh = evaluate(models["cca"], Xte, yte, shuffle_ctx=True)
        raw["shuffle"].append(sh["acc"])
    print(f"\n[{tag}] classes={N_CLASSES} regimes={R} seeds={SEEDS}")
    out = {"task": "text_cls", "tag": tag, "n_regimes": R,
           "n_classes": N_CLASSES, "seeds": SEEDS, "models": {}}
    for mode in ("silu", "cca", "shuffle"):
        m, s = mstd(raw[mode])
        out["models"][mode] = {"acc": m, "acc_std": s,
                               "params": n_par_ref.get(mode)}
        print(f"  {mode:7s} acc {m*100:.2f}% +/- {s*100:.2f}")
    if coeff_vars:
        out["coeff_var"] = torch.stack(coeff_vars).mean(0).tolist()
    bm, _ = mstd(raw["silu"])
    cm, _ = mstd(raw["cca"])
    out["err_reduction_pct"] = 100 * ((1 - bm) - (1 - cm)) / (1 - bm) if bm < 1 else 0
    out["acc_gain_pp"] = 100 * (cm - bm)
    print(f"  -> error -{out['err_reduction_pct']:.1f}%  (+{out['acc_gain_pp']:.2f}pp)")
    measure_cost(out, models, (Xtr, ytr, Xte, yte))
    return out


def measure_cost(out, models, data):
    """Attach a cost block per model + the CCA viability ratio (last-seed models)."""
    Xte, yte = data[2], data[3]
    xb = Xte[:BATCH].to(DEVICE)
    yb = yte[:BATCH].to(DEVICE)
    base_par = out["models"]["silu"]["params"]
    for mode in ("silu", "cca"):
        model = models[mode]

        def fwd(model=model):
            return model(xb)

        def step(model=model):
            loss = F.cross_entropy(model(xb), yb)
            model.zero_grad(set_to_none=True)
            loss.backward()

        out["models"][mode]["cost"] = metrics.model_cost(
            model, DEVICE, forward_fn=fwd, train_step_fn=step,
            n_items=len(xb) * LEN, baseline_params=base_par)
    out["viability"] = metrics.viability(
        out["err_reduction_pct"], out["models"]["silu"]["cost"]["throughput"],
        out["models"]["cca"]["cost"]["throughput"])
    v = out["viability"]
    print(f"  cost: CCA {out['models']['cca']['cost']['params']/1e3:.0f}k par "
          f"({out['models']['cca']['cost']['params_overhead_pct']:+.0f}%), "
          f"throughput cost {v['throughput_cost_pct']:.0f}% -> "
          f"R={v['viability_ratio']:.2f} "
          f"({'rentable' if v['rentable'] else 'NOT rentable'})")


def main():
    multi = run(8, "multi-regime")
    with open("results/text_cls.json", "w") as f:
        json.dump(multi, f, indent=2)
    single = run(1, "single-regime")
    with open("results/text_cls_single.json", "w") as f:
        json.dump(single, f, indent=2)
    print("\nsaved results/text_cls.json (+_single)")


if __name__ == "__main__":
    main()

"""
Image classification — multi-regime MNIST with a small CNN.
===========================================================

We build K "regimes" by applying deterministic pixel transforms (identity,
invert, rotations, flips, transpose) to MNIST. The label (digit) is unchanged;
only the pixel statistics change. A single fixed pointwise activation must
serve every regime at once, whereas CCA can read a global summary of the
feature map and adapt its curve per image (e.g. effectively flip its "on"
direction for inverted inputs). Both models see the whole image, so any CCA
gain here is about *conditional computation / expressivity*, not extra info.

Controls:
  * single-regime null (identity only)   -> expect CCA ~ baseline
  * context-shuffle ablation             -> expect CCA collapses

Writes results/image_cls.json (+ _single).   Run: python image_cls.py
"""

import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import metrics
from cca_core import set_cca, mean_coeff_variance
from cca_factory import make_cca2d

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256
EPOCHS = int(os.environ.get("EPOCHS", 4))
LR = 2e-3
N_PER_CLASS_TRAIN = int(os.environ.get("NTRAIN", 4000))   # subset for speed
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1337,1338,1339").split(",")]
os.makedirs("results", exist_ok=True)


def mstd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


# --------------------------------------------------------------------------- #
#  Regimes
# --------------------------------------------------------------------------- #
def regimes(multi):
    if not multi:
        return [lambda x: x]
    return [
        lambda x: x,                       # identity
        lambda x: 1.0 - x,                 # invert (negative)
        lambda x: torch.rot90(x, 1, (-2, -1)),
        lambda x: torch.rot90(x, 2, (-2, -1)),
        lambda x: torch.rot90(x, 3, (-2, -1)),
        lambda x: x.transpose(-2, -1),     # transpose
        lambda x: torch.flip(x, (-1,)),    # horizontal flip
        lambda x: 1.0 - torch.flip(x, (-2,)),  # vertical flip + invert
    ]


def build_data(multi):
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=False, transform=tf)
    te = datasets.MNIST("./data", train=False, download=False, transform=tf)

    def subset(ds, n_per):
        X = ds.data.float().unsqueeze(1) / 255.0
        y = ds.targets
        if n_per is not None:
            idx = []
            g = torch.Generator().manual_seed(SEED)
            for c in range(10):
                ci = (y == c).nonzero(as_tuple=True)[0]
                ci = ci[torch.randperm(len(ci), generator=g)[:n_per]]
                idx.append(ci)
            idx = torch.cat(idx)
            X, y = X[idx], y[idx]
        return X, y

    Xtr, ytr = subset(tr, N_PER_CLASS_TRAIN)
    Xte, yte = subset(te, None)
    regs = regimes(multi)

    def expand(X, y, gen):
        # assign each sample a random regime
        r = torch.randint(len(regs), (len(X),), generator=gen)
        out = torch.empty_like(X)
        for ri, fn in enumerate(regs):
            m = r == ri
            if m.any():
                out[m] = fn(X[m])
        return out, y

    g = torch.Generator().manual_seed(SEED)
    Xtr, ytr = expand(Xtr, ytr, g)
    Xte, yte = expand(Xte, yte, g)
    return (Xtr, ytr), (Xte, yte), len(regs)


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        if mode == "cca":
            self.a1, self.a2 = make_cca2d(16), make_cca2d(32)
        else:
            self.a1 = self.a2 = nn.SiLU()

    def forward(self, x):
        # CCA2D derives its context from GAP(x) internally, so the call site
        # is identical to nn.SiLU's single-argument call.
        x = F.max_pool2d(self.a1(self.c1(x)), 2)
        x = F.max_pool2d(self.a2(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def iterate(X, y, gen, train):
    n = len(X)
    order = torch.randperm(n, generator=gen) if train else torch.arange(n)
    for i in range(0, n, BATCH):
        idx = order[i:i + BATCH]
        yield X[idx].to(DEVICE), y[idx].to(DEVICE)


@torch.no_grad()
def evaluate(model, Xte, yte, *, shuffle_ctx=False, collect=False):
    model.eval()
    set_cca(model, collect=collect, shuffle_ctx=shuffle_ctx)
    gen = torch.Generator().manual_seed(SEED)
    correct, total, t0 = 0, 0, time.time()
    for xb, yb in iterate(Xte, yte, gen, False):
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += len(yb)
    dt = time.time() - t0
    set_cca(model, collect=False, shuffle_ctx=False)
    return {"acc": correct / total, "throughput": total / dt,
            "coeff_var": (mean_coeff_variance(model) if collect else None)}


def train(mode, data):
    (Xtr, ytr), (Xte, yte), _ = data
    torch.manual_seed(SEED)
    model = SmallCNN(mode).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in iterate(Xtr, ytr, gen, True):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model, time.time() - t0, n_par


def run(multi, tag):
    global SEED
    raw = {"silu": [], "cca": [], "shuffle": []}
    coeff_vars, n_par_ref, n_reg = [], {}, None
    for sd in SEEDS:
        SEED = sd
        data = build_data(multi)
        n_reg = data[2]
        models = {}
        for mode in ("silu", "cca"):
            model, tt, n_par = train(mode, data)
            ev = evaluate(model, data[1][0], data[1][1], collect=(mode == "cca"))
            models[mode] = model
            n_par_ref[mode] = n_par
            raw[mode].append(ev["acc"])
            if ev["coeff_var"] is not None:
                coeff_vars.append(ev["coeff_var"])
        sh = evaluate(models["cca"], data[1][0], data[1][1], shuffle_ctx=True)
        raw["shuffle"].append(sh["acc"])
    print(f"\n[{tag}] regimes={n_reg} seeds={SEEDS}")
    out = {"task": "image_cls", "tag": tag, "n_regimes": n_reg,
           "dataset": "MNIST", "seeds": SEEDS, "models": {}}
    for mode in ("silu", "cca", "shuffle"):
        m, s = mstd(raw[mode])
        out["models"][mode] = {"acc": m, "acc_std": s,
                               "params": n_par_ref.get(mode)}
        print(f"  {mode:7s} acc {m*100:.2f}% +/- {s*100:.2f}")
    if coeff_vars:
        out["coeff_var"] = torch.stack(coeff_vars).mean(0).tolist()
    bm, bs = mstd(raw["silu"])
    cm, cs = mstd(raw["cca"])
    out["err_reduction_pct"] = 100 * ((1 - bm) - (1 - cm)) / (1 - bm) if bm < 1 else 0
    out["acc_gain_pp"] = 100 * (cm - bm)
    print(f"  -> error -{out['err_reduction_pct']:.1f}%  "
          f"(+{out['acc_gain_pp']:.2f}pp)")
    measure_cost(out, models, data)
    return out


def measure_cost(out, models, data):
    """Attach a cost block per model + the CCA viability ratio (last-seed models)."""
    xb = data[1][0][:BATCH].to(DEVICE)
    yb = data[1][1][:BATCH].to(DEVICE)
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
            n_items=len(xb), baseline_params=base_par)
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
    multi = run(True, "multi-regime")
    with open("results/image_cls.json", "w") as f:
        json.dump(multi, f, indent=2)
    single = run(False, "single-regime")
    with open("results/image_cls_single.json", "w") as f:
        json.dump(single, f, indent=2)
    print("\nsaved results/image_cls.json (+_single)")


if __name__ == "__main__":
    main()

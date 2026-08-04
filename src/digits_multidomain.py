"""
digits_multidomain.py — real multi-domain digit recognition (MNIST+SVHN+USPS).
=============================================================================

The synthetic experiments use clean, controllable regimes (ciphers, pixel
transforms). This task replaces them with a **real domain shift**: the three
classic digit datasets MNIST (clean handwritten), SVHN (street-view house
numbers, cluttered/colourful) and USPS (low-resolution handwritten) share the
same 10-class label space but have very different pixel statistics. The
**dataset of origin is the regime** — a genuine distribution shift rather than a
synthetic transform. A single fixed activation must serve all three domains;
CCA reads a global summary of the feature map (which carries the domain
signature) and adapts its curve per image.

This is the "genuinely multi-domain real benchmark" flagged as future work in
the report (PAPER.md §7). Everything is greyscaled and resized to 28x28 so a
small CNN fits comfortably on 8 GB.

Controls: single-regime null (MNIST only) + context-shuffle ablation.
Writes results/digits_multidomain.json (+_single).   Run: python digits_multidomain.py
"""

import json
import os
import ssl
import statistics
import time

# USPS is hosted on a server whose TLS chain fails verification on some setups;
# fall back to an unverified context so torchvision can fetch it (datasets only).
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import metrics
from cca_core import CCA2D, set_cca, mean_coeff_variance

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 28
BATCH = 256
EPOCHS = int(os.environ.get("EPOCHS", 4))
LR = 2e-3
N_PER_DOMAIN = int(os.environ.get("NTRAIN", 6000))   # train images per domain
N_TEST_PER = int(os.environ.get("NTEST", 2000))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1337,1338,1339").split(",")]
os.makedirs("results", exist_ok=True)


def mstd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


# --------------------------------------------------------------------------- #
#  Load + normalise the three domains to [N,1,28,28] greyscale in [0,1]
# --------------------------------------------------------------------------- #
def _grey_resize(X):
    if X.dim() == 3:
        X = X.unsqueeze(1)
    if X.size(1) == 3:
        X = (0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]).unsqueeze(1)
    if X.shape[-1] != SIZE:
        X = F.interpolate(X, size=SIZE, mode="bilinear", align_corners=False)
    return X.clamp(0, 1)


def _domain(name, train):
    if name == "mnist":
        ds = datasets.MNIST("./data", train=train, download=True)
        return ds.data.float() / 255.0, ds.targets
    if name == "usps":
        ds = datasets.USPS("./data", train=train, download=True)
        X = torch.stack([torch.as_tensor(_to_arr(img), dtype=torch.float32)
                         for img in ds.data]) / 255.0
        return X, torch.as_tensor(ds.targets)
    # svhn
    ds = datasets.SVHN("./data", split="train" if train else "test", download=True)
    X = torch.tensor(ds.data, dtype=torch.float32) / 255.0       # [N,3,32,32]
    return X, torch.as_tensor(ds.labels) % 10                    # '10' label -> 0


def _to_arr(img):
    import numpy as np
    return np.asarray(img)


DOMAINS = ["mnist", "svhn", "usps"]


def build_data(multi):
    doms = DOMAINS if multi else ["mnist"]

    def collect(train, n_per):
        Xs, ys, rs = [], [], []
        for ri, name in enumerate(doms):
            X, y = _domain(name, train)
            X = _grey_resize(X)
            g = torch.Generator().manual_seed(SEED + ri)
            sel = torch.randperm(len(X), generator=g)[:n_per]
            Xs.append(X[sel])
            ys.append(y[sel].long())
            rs.append(torch.full((len(sel),), ri))
        return torch.cat(Xs), torch.cat(ys), torch.cat(rs)

    Xtr, ytr, _ = collect(True, N_PER_DOMAIN)
    Xte, yte, rte = collect(False, N_TEST_PER)
    return (Xtr, ytr), (Xte, yte), len(doms)


# --------------------------------------------------------------------------- #
#  Small CNN (same shape as image_cls); CCA2D reads GAP context internally
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        if mode == "cca":
            self.a1, self.a2 = CCA2D(16), CCA2D(32)
        else:
            self.a1 = self.a2 = nn.SiLU()

    def forward(self, x):
        x = F.max_pool2d(self.a1(self.c1(x)), 2)
        x = F.max_pool2d(self.a2(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def iterate(X, y, gen, train):
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
    for xb, yb in iterate(Xte, yte, gen, False):
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    dt = time.time() - t0
    set_cca(model, collect=False, shuffle_ctx=False)
    return {"acc": correct / total, "throughput": total / dt,
            "coeff_var": (mean_coeff_variance(model) if collect else None)}


def train(mode, data):
    (Xtr, ytr), _, _ = data
    torch.manual_seed(SEED)
    model = SmallCNN(mode).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in iterate(Xtr, ytr, gen, True):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model, n_par


def run(multi, tag):
    global SEED
    raw = {"silu": [], "cca": [], "shuffle": []}
    coeff_vars, n_par_ref, n_reg = [], {}, None
    models, data = {}, None
    for sd in SEEDS:
        SEED = sd
        data = build_data(multi)
        n_reg = data[2]
        for mode in ("silu", "cca"):
            model, n_par = train(mode, data)
            ev = evaluate(model, data[1][0], data[1][1], collect=(mode == "cca"))
            models[mode] = model
            n_par_ref[mode] = n_par
            raw[mode].append(ev["acc"])
            if ev["coeff_var"] is not None:
                coeff_vars.append(ev["coeff_var"])
        sh = evaluate(models["cca"], data[1][0], data[1][1], shuffle_ctx=True)
        raw["shuffle"].append(sh["acc"])
    print(f"\n[{tag}] domains={DOMAINS if multi else ['mnist']} "
          f"regimes={n_reg} seeds={SEEDS}")
    out = {"task": "digits_multidomain", "tag": tag, "n_regimes": n_reg,
           "domains": DOMAINS if multi else ["mnist"], "seeds": SEEDS, "models": {}}
    for mode in ("silu", "cca", "shuffle"):
        m, s = mstd(raw[mode])
        out["models"][mode] = {"acc": m, "acc_std": s, "params": n_par_ref.get(mode)}
        print(f"  {mode:7s} acc {m*100:.2f}% +/- {s*100:.2f}")
    if coeff_vars:
        out["coeff_var"] = torch.stack(coeff_vars).mean(0).tolist()
    bm, _ = mstd(raw["silu"])
    cm, _ = mstd(raw["cca"])
    out["err_reduction_pct"] = 100 * ((1 - bm) - (1 - cm)) / (1 - bm) if bm < 1 else 0
    out["acc_gain_pp"] = 100 * (cm - bm)
    print(f"  -> error -{out['err_reduction_pct']:.1f}%  (+{out['acc_gain_pp']:.2f}pp)")
    measure_cost(out, models, data)
    return out


def measure_cost(out, models, data):
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
    multi = run(True, "multi-domain")
    with open("results/digits_multidomain.json", "w") as f:
        json.dump(multi, f, indent=2)
    single = run(False, "single-domain")
    with open("results/digits_multidomain_single.json", "w") as f:
        json.dump(single, f, indent=2)
    print("\nsaved results/digits_multidomain.json (+_single)")


if __name__ == "__main__":
    main()

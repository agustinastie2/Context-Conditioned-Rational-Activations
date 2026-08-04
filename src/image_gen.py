"""
Image generation — autoregressive over multi-regime MNIST (myopic backbone).
============================================================================

A VAE hides the regime inside its latent, and a *full-context* Transformer can
identify the regime itself via attention; in both cases a fixed-activation
baseline catches up, so the activation is never the bottleneck. We therefore use
the design that made the text-generation experiment work: make the backbone
*blind* to the regime and give the context-conditioned activation a privileged,
strictly-causal summary that reveals it.

Setup: MNIST downsampled to 16x16, quantised to Q=8 grey levels, raster-ordered.
R "regimes" permute the Q levels (a per-pixel value cipher). The backbone is a
**myopic local** model — a small causal depthwise conv (kernel K) + per-position
MLP, *no attention* — so it sees only the last K pixels and cannot compute the
global level histogram that identifies the permutation. The CCA activation is
conditioned on the **strictly-past running level histogram** (normalised counts
of the Q levels among pixels seen so far), which does identify the regime.
Metric: test NLL in bits/dim (lower is better).

Controls:
  * single-regime null (R=1)
  * context-shuffle ablation (wrong histogram -> wrong regime)
  * `silu_hist`: B0H-style control that concatenates the same histogram to the
    input, so the baseline gets the regime info too — isolating whether routing
    the activation beats merely injecting the context.

Writes results/image_gen.json (+_single).   Run: python image_gen.py
"""

import json
import math
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import metrics
from cca_core import CCA1D, set_cca, mean_coeff_variance

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DATASET selects the image source; CIFAR is real natural images (greyscaled,
# quantised) to keep the autoregressive model small enough for 8 GB.
DATASET = os.environ.get("DATASET", "mnist").lower()
GRID = int(os.environ.get("GRID", 24 if DATASET == "cifar" else 16))
Q = 8                          # grey levels / vocab
SEQ = GRID * GRID              # pixels
DMODEL = 64
KERNEL = int(os.environ.get("KERNEL", 3))   # myopic receptive field
NLAYERS = 2
# CIFAR's longer sequences (24x24=576) need a smaller batch to fit 8 GB VRAM.
BATCH = int(os.environ.get("BATCH", 64 if DATASET == "cifar" else 128))
STEPS = int(os.environ.get("STEPS", 1500))
LR = 3e-3
N_PER_CLASS = int(os.environ.get("NTRAIN", 2500))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1337,1338,1339").split(",")]
os.makedirs("results", exist_ok=True)


def mstd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


# --------------------------------------------------------------------------- #
#  Data: quantised, regime-permuted MNIST as token sequences
# --------------------------------------------------------------------------- #
def level_perms(R, seed):
    g = torch.Generator().manual_seed(seed)
    perms = [torch.arange(Q)]
    for _ in range(R - 1):
        perms.append(torch.randperm(Q, generator=g))
    return perms


def _load_raw(train):
    """Return (images[N,1,H,W] in [0,1], labels) for the selected DATASET.
    CIFAR-10 is converted to greyscale so the autoregressive model stays small."""
    if DATASET == "fashion":
        ds = datasets.FashionMNIST("./data", train=train, download=True)
        return ds.data.float().unsqueeze(1) / 255.0, ds.targets
    if DATASET == "cifar":
        ds = datasets.CIFAR10("./data", train=train, download=True)
        X = torch.tensor(ds.data, dtype=torch.float32) / 255.0   # [N,H,W,3]
        grey = (0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2])
        return grey.unsqueeze(1), torch.tensor(ds.targets)
    ds = datasets.MNIST("./data", train=train, download=False)
    return ds.data.float().unsqueeze(1) / 255.0, ds.targets


def build_data(multi):
    R = int(os.environ.get("NREG", 16)) if multi else 1
    tr_X, tr_y = _load_raw(True)
    te_X, te_y = _load_raw(False)

    def quant(X, y, n_per):
        idx, g = [], torch.Generator().manual_seed(SEED)
        for c in range(10):
            ci = (y == c).nonzero(as_tuple=True)[0]
            idx.append(ci[torch.randperm(len(ci), generator=g)[:n_per]])
        X = X[torch.cat(idx)]
        X = F.adaptive_avg_pool2d(X, GRID)[:, 0]         # -> [N, GRID, GRID]
        X = (X * (Q - 1)).round().long().clamp(0, Q - 1)
        return X.reshape(len(X), SEQ)                     # raster tokens

    perms = level_perms(R, SEED + 5)

    def apply(tok, gen):
        r = torch.randint(len(perms), (len(tok),), generator=gen)
        out = tok.clone()
        for ri, p in enumerate(perms):
            m = r == ri
            if m.any():
                out[m] = p[tok[m]]
        return out

    g = torch.Generator().manual_seed(SEED)
    Xtr = apply(quant(tr_X, tr_y, N_PER_CLASS), g)
    Xte = apply(quant(te_X, te_y, 400), g)
    return Xtr, Xte, R


# --------------------------------------------------------------------------- #
#  Strictly-causal running level histogram (reveals the level-permutation)
# --------------------------------------------------------------------------- #
def running_hist(idx):
    """idx [B,S] -> [B,S,Q] normalised counts of each level among tokens <= t."""
    oneh = F.one_hot(idx, Q).float()                  # [B,S,Q]
    counts = oneh.cumsum(1)                            # inclusive counts
    denom = torch.arange(1, idx.size(1) + 1, device=idx.device).view(1, -1, 1)
    return counts / denom


# --------------------------------------------------------------------------- #
#  Model: myopic local backbone (causal depthwise conv + MLP, NO attention)
# --------------------------------------------------------------------------- #
class LocalBlock(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.ln = nn.LayerNorm(DMODEL)
        # depthwise causal conv: sees only the last KERNEL positions
        self.conv = nn.Conv1d(DMODEL, DMODEL, KERNEL, padding=KERNEL - 1,
                              groups=DMODEL)
        self.fc1 = nn.Linear(DMODEL, 4 * DMODEL)
        self.fc2 = nn.Linear(4 * DMODEL, DMODEL)
        self.mode = mode
        if mode == "cca":
            self.act = CCA1D(Q, 4 * DMODEL)            # context = histogram (Q)
        else:
            self.act = nn.SiLU()

    def forward(self, x, hist):
        h = self.ln(x)
        c = self.conv(h.transpose(1, 2))[:, :, :h.size(1)].transpose(1, 2)
        a = self.fc1(c)
        a = self.act(a, hist) if self.mode == "cca" else self.act(a)
        return x + self.fc2(a)


class LocalAR(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.tok = nn.Embedding(Q, DMODEL)
        self.pos = nn.Embedding(SEQ, DMODEL)
        # B0H control: project the histogram and add it to the input embeddings
        self.hist_in = nn.Linear(Q, DMODEL) if mode == "silu_hist" else None
        self.blocks = nn.ModuleList([LocalBlock(mode) for _ in range(NLAYERS)])
        self.ln = nn.LayerNorm(DMODEL)
        self.head = nn.Linear(DMODEL, Q)

    def forward(self, idx):
        B, S = idx.shape
        hist = running_hist(idx)                       # [B,S,Q] strictly causal
        x = self.tok(idx) + self.pos(torch.arange(S, device=idx.device))
        if self.hist_in is not None:                   # B0H injection control
            x = x + self.hist_in(hist)
        for b in self.blocks:
            x = b(x, hist)
        return self.head(self.ln(x))

    def loss(self, idx):
        logits = self(idx[:, :-1])
        return F.cross_entropy(logits.reshape(-1, Q), idx[:, 1:].reshape(-1))


def batches(X, gen, train):
    order = torch.randperm(len(X), generator=gen) if train else torch.arange(len(X))
    for i in range(0, len(X), BATCH):
        yield X[order[i:i + BATCH]].to(DEVICE)


@torch.no_grad()
def evaluate(model, Xte, *, shuffle_ctx=False, collect=False):
    model.eval()
    set_cca(model, collect=collect, shuffle_ctx=shuffle_ctx)
    gen = torch.Generator().manual_seed(SEED)
    nll, n, t0 = 0.0, 0, time.time()
    for xb in batches(Xte, gen, False):
        nll += model.loss(xb).item() * len(xb)
        n += len(xb)
    dt = time.time() - t0
    set_cca(model, collect=False, shuffle_ctx=False)
    nats = nll / n
    return {"nll": nats, "bpd": nats / math.log(2), "throughput": n * SEQ / dt,
            "coeff_var": (mean_coeff_variance(model) if collect else None)}


def train(mode, Xtr):
    if DEVICE == "cuda":
        torch.cuda.empty_cache()          # reduce fragmentation across runs
    torch.manual_seed(SEED)
    model = LocalAR(mode).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    t0, step = time.time(), 0
    while step < STEPS:
        model.train()
        for xb in batches(Xtr, gen, True):
            loss = model.loss(xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step >= STEPS:
                break
    return model, time.time() - t0, n_par


@torch.no_grad()
def sample(model, n, temp=1.0):
    """Autoregressively generate n images -> [n, GRID, GRID] long tensors."""
    model.eval()
    set_cca(model, collect=False, shuffle_ctx=False)
    torch.manual_seed(SEED)
    idx = torch.randint(Q, (n, 1), device=DEVICE)     # seed pixel 0
    for _ in range(1, SEQ):
        logits = model(idx)[:, -1] / temp
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx.view(n, GRID, GRID).cpu()


def save_comparison(models, Xte, dataset, n=8):
    """Side-by-side inference grid: real data vs SiLU vs CCA samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.manual_seed(SEED)
    real = Xte[torch.randperm(len(Xte))[:n]].view(n, GRID, GRID).cpu()
    rows = [("Real data", real),
            ("SiLU\n(baseline)", sample(models["silu"], n)),
            ("CCA\n(ours)", sample(models["cca"], n))]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.05, 3.4))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=Q - 1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right",
                              va="center", labelpad=22)
    fig.suptitle(f"Autoregressive inference samples: {dataset} "
                 f"({GRID}x{GRID}, q{Q}, 16-regime mixture)\n"
                 "CCA vs the fixed-SiLU baseline", fontsize=10)
    fig.tight_layout(rect=(0.04, 0, 1, 0.92))
    fig.savefig(f"figs/samples_{dataset}.png")
    plt.close(fig)
    print(f"  wrote figs/samples_{dataset}.png")


def measure_cost(out, models, Xte):
    """Attach a cost block per model + the CCA viability ratio (last-seed models)."""
    xb = Xte[:BATCH].to(DEVICE)
    base_par = out["models"]["silu"]["params"]
    for mode in ("silu", "cca"):
        model = models[mode]

        def fwd(model=model):
            return model(xb[:, :-1])

        def step(model=model):
            loss = model.loss(xb)
            model.zero_grad(set_to_none=True)
            loss.backward()

        out["models"][mode]["cost"] = metrics.model_cost(
            model, DEVICE, forward_fn=fwd, train_step_fn=step,
            n_items=xb.numel(), baseline_params=base_par)
    # quality gain for image-gen is the bits/dim reduction (lower is better)
    bm, cm = out["models"]["silu"]["bpd"], out["models"]["cca"]["bpd"]
    out["viability"] = metrics.viability(
        100 * (bm - cm) / bm, out["models"]["silu"]["cost"]["throughput"],
        out["models"]["cca"]["cost"]["throughput"])
    v = out["viability"]
    print(f"  cost: CCA {out['models']['cca']['cost']['params']/1e3:.0f}k par "
          f"({out['models']['cca']['cost']['params_overhead_pct']:+.0f}%), "
          f"throughput cost {v['throughput_cost_pct']:.0f}% -> "
          f"R={v['viability_ratio']:.2f} "
          f"({'rentable' if v['rentable'] else 'NOT rentable'})")


def run(multi, tag):
    global SEED
    train_modes = ("silu", "silu_hist", "cca")
    raw = {m: [] for m in train_modes + ("shuffle",)}
    coeff_vars, n_par_ref, n_reg = [], {}, None
    for sd in SEEDS:
        SEED = sd
        Xtr, Xte, n_reg = build_data(multi)
        models = {}
        for mode in train_modes:
            model, tt, n_par = train(mode, Xtr)
            ev = evaluate(model, Xte, collect=(mode == "cca"))
            models[mode] = model
            n_par_ref[mode] = n_par
            raw[mode].append(ev["bpd"])
            if ev["coeff_var"] is not None:
                coeff_vars.append(ev["coeff_var"])
        sh = evaluate(models["cca"], Xte, shuffle_ctx=True)
        raw["shuffle"].append(sh["bpd"])
    print(f"\n[{tag}] dataset={DATASET} regimes={n_reg} seeds={SEEDS}")
    out = {"task": "image_gen", "tag": tag, "n_regimes": n_reg,
           "dataset": f"{DATASET} {GRID}x{GRID} q{Q} (myopic autoregressive)",
           "dataset_name": DATASET, "seeds": SEEDS,
           "metric": "bits/dim", "models": {}}
    for mode in train_modes + ("shuffle",):
        m, s = mstd(raw[mode])
        out["models"][mode] = {"bpd": m, "bpd_std": s, "params": n_par_ref.get(mode)}
        print(f"  {mode:9s} bpd {m:.4f} +/- {s:.4f}")
    if coeff_vars:
        out["coeff_var"] = torch.stack(coeff_vars).mean(0).tolist()
    bm, cm = out["models"]["silu"]["bpd"], out["models"]["cca"]["bpd"]
    hm = out["models"]["silu_hist"]["bpd"]
    out["improvement_pct"] = 100 * (bm - cm) / bm
    out["improvement_vs_hist_pct"] = 100 * (hm - cm) / hm
    print(f"  -> vs SiLU: {out['improvement_pct']:.1f}%   "
          f"vs SiLU+hist(B0H): {out['improvement_vs_hist_pct']:.1f}%")
    measure_cost(out, models, Xte)
    return out, models, Xte


def main():
    # mnist keeps the canonical filenames (paper headline); other datasets are
    # tagged so the three image-gen sources coexist in results/.
    suffix = "" if DATASET == "mnist" else f"_{DATASET}"
    multi, models, Xte = run(True, "multi")
    with open(f"results/image_gen{suffix}.json", "w") as f:
        json.dump(multi, f, indent=2)
    save_comparison(models, Xte, DATASET)
    single, _, _ = run(False, "single")
    with open(f"results/image_gen{suffix}_single.json", "w") as f:
        json.dump(single, f, indent=2)
    print(f"\nsaved results/image_gen{suffix}.json (+_single) + samples")


if __name__ == "__main__":
    main()

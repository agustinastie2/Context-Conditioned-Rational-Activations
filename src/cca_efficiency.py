"""
cca_efficiency.py — is CCA worth its cost, and can we make it cheaper?
=====================================================================

CCA buys quality with compute. This script measures *both* sides on a single
representative multi-regime task (the text-classification benchmark, which uses
the canonical `CCA1D` and trains in seconds) and sweeps a family of progressively
cheaper CCA generators to trace the **cost/quality Pareto frontier**:

    full        groups=64, g_hidden=64, affine on      (the paper's tuned CCA)
    g16 / g8    fewer channel-groups   -> smaller generator output
    ghid16      narrower generator hidden
    lowrank-r   low-rank factorisation of the generator's output projection
    lite        the cheap combination (few groups + narrow hidden)
    no-affine   v1 curve (5 coeffs, no re-centring)

For every variant we report the quality gain (error reduction vs a SiLU
baseline) and the cost (params, GFLOPs, throughput, peak memory), and the
*viability ratio* R = gain% / throughput-cost% (R>1 = the gain outweighs the
slowdown). We then pick **CCA-lite**: the cheapest variant that keeps most of
full-CCA's quality and is rentable, to report alongside CCA-full in the paper.

Writes results/cca_efficiency.json and figs/fig_pareto.png.   Run: python cca_efficiency.py
"""

import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import metrics
import text_cls as tc           # reuse the multi-regime data pipeline
from cca_core import CCA1D

try:                            # keep torch.compile's autotuner quiet
    torch._logging.set_logs(inductor=logging.ERROR, dynamo=logging.ERROR)
except Exception:
    pass
torch.set_float32_matmul_precision("high")

SEED = int(os.environ.get("SEED", 1337))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = int(os.environ.get("STEPS", 1500))
R_REGIMES = 8
os.makedirs("results", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# Variant catalogue: name -> CCA kwargs (None mode = SiLU baseline)
VARIANTS = {
    "SiLU":          None,
    "CCA-full":      dict(groups=64, g_hidden=64),
    "CCA-g16":       dict(groups=16, g_hidden=64),
    "CCA-g8":        dict(groups=8,  g_hidden=64),
    "CCA-ghid16":    dict(groups=64, g_hidden=16),
    "CCA-lowrank8":  dict(groups=64, g_hidden=64, lowrank=8),
    "CCA-lite":      dict(groups=16, g_hidden=16),
    "CCA-noaffine":  dict(groups=64, g_hidden=64, affine=False),
}


class BoE(nn.Module):
    """text_cls bag-of-embeddings classifier with a configurable CCA (or SiLU)."""

    def __init__(self, cca_kwargs):
        super().__init__()
        self.cca = cca_kwargs is not None
        self.emb = nn.Embedding(tc.VOCAB, tc.EMB)
        self.fc1 = nn.Linear(tc.EMB, tc.HID)
        self.fc2 = nn.Linear(tc.HID, tc.HID)
        self.head = nn.Linear(tc.HID, tc.N_CLASSES)
        if self.cca:
            self.a1 = CCA1D(tc.EMB, tc.HID, **cca_kwargs)
            self.a2 = CCA1D(tc.EMB, tc.HID, **cca_kwargs)
        else:
            self.a1 = self.a2 = nn.SiLU()

    def forward(self, idx):
        e = self.emb(idx)
        ctx = e.mean(1)
        h = self.a1(self.fc1(ctx), ctx) if self.cca else self.a1(self.fc1(ctx))
        h = self.a2(self.fc2(h), ctx) if self.cca else self.a2(self.fc2(h))
        return self.head(h)


def train_eval(cca_kwargs, data):
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(SEED)
    model = BoE(cca_kwargs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.LR)
    gen = torch.Generator().manual_seed(SEED)
    step = 0
    model.train()
    while step < STEPS:
        for xb, yb in tc.batches(Xtr, ytr, gen, True):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= STEPS:
                break
    model.eval()
    gen = torch.Generator().manual_seed(SEED)
    correct = total = 0
    with torch.no_grad():
        for xb, yb in tc.batches(Xte, yte, gen, False):
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return model, correct / total


def _min_latency(fn, iters=80):
    """Min wall-time over many runs — robust to jitter for sub-ms kernels."""
    import time
    for _ in range(15):
        fn()
    metrics._sync(DEVICE)
    best = float("inf")
    use_cuda = torch.device(DEVICE).type == "cuda"
    for _ in range(iters):
        if use_cuda:
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); fn(); best = min(best, (time.perf_counter()-t0)*1e3)
    return best


def compiled_throughput(model, xb, n_items):
    """Throughput (items/s) of the model after torch.compile fuses its kernels.
    A fresh dynamo state per call avoids cross-variant compile contamination.
    Falls back to eager timing if compilation is unavailable on this platform."""
    try:
        torch._dynamo.reset()
        cm = torch.compile(model)
        with torch.no_grad():
            lat = _min_latency(lambda: cm(xb))
        return n_items / (lat / 1e3), lat
    except Exception:
        lat = _min_latency(lambda: model(xb))
        return n_items / (lat / 1e3), lat


EFF_SEEDS = [int(s) for s in os.environ.get("EFF_SEEDS", "1337,1338,1339").split(",")]


def main():
    global SEED
    # quality is averaged over a few seeds (single-seed accuracy is noisy);
    # the data is regenerated per seed exactly as the text-cls benchmark does.
    accs = {name: [] for name in VARIANTS}
    last_model = {}
    for sd in EFF_SEEDS:
        SEED = sd
        sources = tc.make_sources(tc.N_CLASSES, sd)
        ciphers = tc.make_ciphers(R_REGIMES, sd + 1)
        Xtr, ytr = tc.gen_dataset(tc.N_TRAIN, sources, ciphers, sd)
        Xte, yte = tc.gen_dataset(tc.N_TEST, sources, ciphers, sd + 99)
        for name, kw in VARIANTS.items():
            model, acc = train_eval(kw, (Xtr, ytr, Xte, yte))
            accs[name].append(acc)
            last_model[name] = model
            last_batch = (Xte[:tc.BATCH].to(DEVICE), yte[:tc.BATCH].to(DEVICE))

    xb, yb = last_batch
    n_items = len(xb) * tc.LEN
    rows = []
    base = {"acc": None, "par": None, "tput": None, "tput_c": None}
    print(f"\nvariant sweep on text-cls (R={R_REGIMES}, steps={STEPS}, "
          f"seeds={EFF_SEEDS}) - quality averaged, cost = exact params/FLOPs")
    for name in VARIANTS:
        model = last_model[name]
        acc = sum(accs[name]) / len(accs[name])

        def fwd(model=model):
            return model(xb)

        def step(model=model):
            loss = F.cross_entropy(model(xb), yb)
            model.zero_grad(set_to_none=True)
            loss.backward()

        cost = metrics.model_cost(model, DEVICE, forward_fn=fwd,
                                  train_step_fn=step, n_items=n_items,
                                  baseline_params=base["par"])
        with metrics._eval_mode(model), torch.no_grad():
            cost["throughput"] = n_items / (_min_latency(fwd) / 1e3)
            tput_c, _ = compiled_throughput(model, xb, n_items)
        cost["throughput_compiled"] = tput_c
        if name == "SiLU":
            base.update(acc=acc, par=cost["params"],
                        tput=cost["throughput"], tput_c=tput_c)
            cost["params_overhead_pct"] = 0.0
            cost["flops_overhead_pct"] = 0.0
        gain = (100 * ((1 - base["acc"]) - (1 - acc)) / (1 - base["acc"])
                if base["acc"] is not None and base["acc"] < 1 else 0.0)
        # FLOP overhead vs SiLU is exact and deterministic — the primary cost axis
        base_flops = rows[0]["cost"]["gflops"] if rows else cost["gflops"]
        cost["flops_overhead_pct"] = metrics.overhead_pct(cost["gflops"], base_flops)
        # viability: gain vs FLOP overhead (primary), plus throughput (secondary)
        flop_cost = cost["flops_overhead_pct"]
        R_flops = gain / flop_cost if flop_cost > 0 else float("inf")
        viab_tput = metrics.viability(gain, base["tput"], cost["throughput"])
        row = {"name": name, "acc": acc, "err_reduction_pct": gain, "cost": cost,
               "flops_cost_pct": flop_cost, "viability_ratio_flops": R_flops,
               "rentable_flops": gain > flop_cost,
               "throughput_cost_pct": viab_tput["throughput_cost_pct"],
               "viability_ratio_tput": viab_tput["viability_ratio"]}
        rows.append(row)
        print(f"  {name:13s} acc {acc*100:5.2f}%  errRed {gain:5.1f}%  "
              f"params {cost['params']/1e3:4.0f}k ({cost['params_overhead_pct']:+4.0f}%)  "
              f"FLOPs {cost['gflops']*1e3:5.1f}M ({flop_cost:+4.0f}%)  "
              f"R_flops={R_flops:4.1f}  tputCost {row['throughput_cost_pct']:4.0f}%")

    pick = choose_lite(rows)
    # headline: clean compiled-vs-eager speedup of CCA-full (fusion lever)
    full = next(r for r in rows if r["name"] == "CCA-full")
    fusion_speedup = full["cost"]["throughput_compiled"] / full["cost"]["throughput"]
    out = {"task": "cca_efficiency_text_cls", "seeds": EFF_SEEDS, "steps": STEPS,
           "regimes": R_REGIMES, "variants": rows, "recommended_lite": pick,
           "fusion_speedup_cca_full": fusion_speedup}
    with open("results/cca_efficiency.json", "w") as f:
        json.dump(out, f, indent=2)
    plot_pareto(rows, pick)
    print(f"\nkernel fusion (torch.compile) speeds up CCA-full {fusion_speedup:.1f}x")
    print(f"recommended CCA-lite: {pick}")
    print("saved results/cca_efficiency.json + figs/fig_pareto.png")


def choose_lite(rows):
    """Cheapest variant by FLOP overhead that keeps >=90% of CCA-full's error
    reduction (the prudent, rentable configuration)."""
    full = next(r for r in rows if r["name"] == "CCA-full")
    cands = [r for r in rows if r["name"] not in ("SiLU", "CCA-full")
             and r["err_reduction_pct"] >= 0.90 * full["err_reduction_pct"]]
    if not cands:
        cands = [r for r in rows if r["name"] != "SiLU"]
    return min(cands, key=lambda r: r["cost"]["flops_overhead_pct"])["name"]


def plot_pareto(rows, pick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
                         "font.family": "serif", "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    cca = [r for r in rows if r["name"] != "SiLU"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (xk, xl) in zip(
            axes, [("flops_overhead_pct", "Forward-FLOPs overhead vs SiLU (%)"),
                   ("params_overhead_pct", "Parameter overhead vs SiLU (%)")]):
        xs = [r["cost"][xk] for r in cca]
        ys = [r["err_reduction_pct"] for r in cca]
        for r, x, y in zip(cca, xs, ys):
            is_pick = r["name"] == pick
            is_full = r["name"] == "CCA-full"
            ax.scatter([x], [y], s=130 if (is_pick or is_full) else 60,
                       color="#C44E52" if is_full else
                       ("#55A868" if is_pick else "#4C72B0"),
                       zorder=3, edgecolor="k", linewidth=0.6)
            ax.annotate(r["name"].replace("CCA-", ""), (x, y), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        lim = max(xs) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5,
                label="gain = cost (R=1)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_xlabel(xl)
        ax.set_ylabel("Error reduction vs SiLU (%)")
    fig.suptitle("CCA cost/quality Pareto frontier (text classification)\n"
                 "points above the dashed line are rentable (gain > cost) — "
                 "green = recommended CCA-lite, red = CCA-full", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("figs/fig_pareto.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

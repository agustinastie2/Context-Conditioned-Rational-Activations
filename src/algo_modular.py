"""
algo_modular.py — algorithmic reasoning: keyed symbol substitution + modular sum.
================================================================================

A symbolic reasoning task in the strict no-leak frame of the generation
benchmarks. Each example is K symbols x_1..x_K over an alphabet of size M. A
per-example **substitution key** (a permutation pi of the alphabet) is the
regime; the target is (sum_i pi(x_i)) mod M — i.e. decode every symbol through
the key, then add modulo M. The raw symbols span the whole alphabet regardless
of the key, so the backbone **cannot infer the key from the symbols**; the key
is supplied only through a privileged context channel. We compare:

  * B0  (SiLU)         : symbols only — blind to the key, so it can only guess.
  * B0H (SiLU + key)   : the key embedding added to the pooled features.
  * CCA (key-routed)   : the key embedding drives the activation's shape.

Applying a key-specific symbol remap is precisely the kind of regime-dependent
input->output mapping a single fixed activation cannot serve. As in the
generation tasks, the decisive comparison is CCA vs the B0H injection control:
does *routing* the key through the nonlinearity beat merely *adding* it?

Controls: single-regime null (one key) + context-shuffle ablation (wrong key).
Writes results/algo_modular.json (+_single).   Run: python algo_modular.py
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
M = 20                                        # alphabet size = #classes = modulus
N_KEYS = int(os.environ.get("NKEYS", 8))      # substitution keys = regimes (R)
K = 4                                         # symbols per example
EMB = 32
HID = 256
BATCH = 256
STEPS = int(os.environ.get("STEPS", 3000))
LR = 2e-3
N_TRAIN = 60000
N_TEST = 10000
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1337,1338,1339").split(",")]
os.makedirs("results", exist_ok=True)


def mstd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


def make_keys(n_keys, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(M, generator=g) for _ in range(n_keys)])


def gen_dataset(n, keys, seed):
    """K symbols in [0,M); regime = a random key; label = (sum key(symbol)) mod M.
    Symbols span [0,M) for every key, so the key is hidden from the symbols."""
    g = torch.Generator().manual_seed(seed)
    R = len(keys)
    regs = torch.randint(R, (n,), generator=g)
    X = torch.randint(M, (n, K), generator=g)
    decoded = keys[regs.unsqueeze(1).expand(-1, K), X]       # [n, K]
    y = decoded.sum(1) % M
    return X, y, regs


class KeyedNet(nn.Module):
    """Sum-of-embeddings backbone (blind to the key) + a key context channel."""

    def __init__(self, mode, n_keys):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(M, EMB)
        self.key_emb = nn.Embedding(n_keys, EMB)            # key -> context vector
        self.enc = nn.Sequential(nn.Linear(EMB, EMB), nn.SiLU())
        self.fc1 = nn.Linear(EMB, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.head = nn.Linear(HID, M)
        if mode == "cca":
            self.a1, self.a2 = CCA1D(EMB, HID), CCA1D(EMB, HID)
        else:
            self.a1 = self.a2 = nn.SiLU()

    def forward(self, idx, reg):
        s = self.emb(idx).sum(1)                            # additive aggregate
        c = self.enc(self.key_emb(reg))                     # key context [B, EMB]
        x = s + c if self.mode == "silu_hist" else s        # B0H injects the key
        h = self.a1(self.fc1(x), c) if self.mode == "cca" else self.a1(self.fc1(x))
        h = self.a2(self.fc2(h), c) if self.mode == "cca" else self.a2(self.fc2(h))
        return self.head(h)


def batches(X, y, r, gen, train):
    order = torch.randperm(len(X), generator=gen) if train else torch.arange(len(X))
    for i in range(0, len(X), BATCH):
        idx = order[i:i + BATCH]
        yield X[idx].to(DEVICE), y[idx].to(DEVICE), r[idx].to(DEVICE)


@torch.no_grad()
def evaluate(model, data_te, *, shuffle_ctx=False, collect=False):
    Xte, yte, rte = data_te
    model.eval()
    set_cca(model, collect=collect, shuffle_ctx=shuffle_ctx)
    gen = torch.Generator().manual_seed(SEED)
    correct, total, t0 = 0, 0, time.time()
    for xb, yb, rb in batches(Xte, yte, rte, gen, False):
        if shuffle_ctx:                                # wrong key context (B0H/SiLU only)
            rb = rb[torch.randperm(len(rb), device=rb.device)]
        correct += (model(xb, rb).argmax(1) == yb).sum().item()
        total += len(yb)
    dt = time.time() - t0
    set_cca(model, collect=False, shuffle_ctx=False)
    return {"acc": correct / total, "throughput": total * K / dt,
            "coeff_var": (mean_coeff_variance(model) if collect else None)}


def train(mode, tr, n_keys):
    Xtr, ytr, rtr = tr
    torch.manual_seed(SEED)
    model = KeyedNet(mode, n_keys).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    step = 0
    while step < STEPS:
        for xb, yb, rb in batches(Xtr, ytr, rtr, gen, True):
            loss = F.cross_entropy(model(xb, rb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= STEPS:
                break
    return model, n_par


def run(n_keys, tag):
    global SEED
    modes = ("silu", "silu_hist", "cca")
    raw = {m: [] for m in modes + ("shuffle",)}
    coeff_vars, n_par_ref = [], {}
    models, data_te = {}, None
    for sd in SEEDS:
        SEED = sd
        keys = make_keys(n_keys, sd + 5)
        tr = gen_dataset(N_TRAIN, keys, sd)
        data_te = gen_dataset(N_TEST, keys, sd + 99)
        for mode in modes:
            model, n_par = train(mode, tr, n_keys)
            ev = evaluate(model, data_te, collect=(mode == "cca"))
            models[mode] = model
            n_par_ref[mode] = n_par
            raw[mode].append(ev["acc"])
            if ev["coeff_var"] is not None:
                coeff_vars.append(ev["coeff_var"])
        sh = evaluate(models["cca"], data_te, shuffle_ctx=True)
        raw["shuffle"].append(sh["acc"])
    print(f"\n[{tag}] keys={n_keys} regimes={n_keys} seeds={SEEDS}")
    out = {"task": "algo_modular", "tag": tag, "n_regimes": n_keys,
           "alphabet": M, "n_keys": n_keys, "seeds": SEEDS, "models": {}}
    for mode in modes + ("shuffle",):
        m, s = mstd(raw[mode])
        out["models"][mode] = {"acc": m, "acc_std": s, "params": n_par_ref.get(mode)}
        print(f"  {mode:9s} acc {m*100:.2f}% +/- {s*100:.2f}")
    if coeff_vars:
        out["coeff_var"] = torch.stack(coeff_vars).mean(0).tolist()
    bm, _ = mstd(raw["silu"])
    hm, _ = mstd(raw["silu_hist"])
    cm, _ = mstd(raw["cca"])
    out["err_reduction_pct"] = 100 * ((1 - bm) - (1 - cm)) / (1 - bm) if bm < 1 else 0
    out["acc_gain_pp"] = 100 * (cm - bm)
    out["improvement_vs_hist_pct"] = (100 * ((1 - hm) - (1 - cm)) / (1 - hm)
                                      if hm < 1 else 0)
    print(f"  -> error -{out['err_reduction_pct']:.1f}%  (+{out['acc_gain_pp']:.2f}pp)  "
          f"vs B0H: {out['improvement_vs_hist_pct']:+.1f}%")
    measure_cost(out, models, data_te)
    return out


def measure_cost(out, models, data_te):
    xb = data_te[0][:BATCH].to(DEVICE)
    yb = data_te[1][:BATCH].to(DEVICE)
    rb = data_te[2][:BATCH].to(DEVICE)
    base_par = out["models"]["silu"]["params"]
    for mode in ("silu", "cca"):
        model = models[mode]

        def fwd(model=model):
            return model(xb, rb)

        def step(model=model):
            loss = F.cross_entropy(model(xb, rb), yb)
            model.zero_grad(set_to_none=True)
            loss.backward()

        out["models"][mode]["cost"] = metrics.model_cost(
            model, DEVICE, forward_fn=fwd, train_step_fn=step,
            n_items=len(xb) * K, baseline_params=base_par)
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
    multi = run(N_KEYS, "multi-regime")
    with open("results/algo_modular.json", "w") as f:
        json.dump(multi, f, indent=2)
    single = run(1, "single-regime")
    with open("results/algo_modular_single.json", "w") as f:
        json.dump(single, f, indent=2)
    print("\nsaved results/algo_modular.json (+_single)")


if __name__ == "__main__":
    main()

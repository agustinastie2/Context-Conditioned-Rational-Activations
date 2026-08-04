"""
text_sentiment_multidomain.py — real multi-domain sentiment classification.
===========================================================================

Replaces the synthetic text-classification regimes with a **real domain shift**:
Blitzer's Multi-Domain Sentiment Dataset (Amazon reviews over four product
domains — books, DVD, electronics, kitchen; processed bag-of-words, ~8 MB,
fetched automatically). The label is sentiment (positive/negative); the
**product domain is the regime**. Domains differ sharply in vocabulary and in
which words signal sentiment ("unreadable" vs "defective" vs "stale"), so a
single fixed activation must compromise across them. The domain is identifiable
from the review's vocabulary signature (the mean feature embedding), which CCA
reads to route its nonlinearity per domain.

Model: a bag-of-embeddings MLP (mirrors text_cls.py) over the top-V features,
with CCA1D conditioned on the pooled feature embedding.

Controls: single-regime null (one domain) + context-shuffle ablation.
Writes results/text_sentiment.json (+_single).   Run: python text_sentiment_multidomain.py
"""

import io
import json
import os
import statistics
import tarfile
import time
import urllib.request
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import metrics
from cca_core import CCA1D, set_cca, mean_coeff_variance

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOMAINS = ["books", "dvd", "electronics", "kitchen"]
VOCAB = 4000                 # top-V bag-of-words features
EMB = 32
HID = 256
BATCH = 256
STEPS = int(os.environ.get("STEPS", 1500))
LR = 2e-3
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1337,1338,1339").split(",")]
URL = "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz"
DATA_DIR = "data/sentiment"
os.makedirs("results", exist_ok=True)


def mstd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


# --------------------------------------------------------------------------- #
#  Fetch + parse processed_acl (stdlib only)
# --------------------------------------------------------------------------- #
def ensure_data():
    marker = os.path.join(DATA_DIR, "processed_acl")
    if os.path.isdir(marker):
        return marker
    os.makedirs(DATA_DIR, exist_ok=True)
    print("downloading Multi-Domain Sentiment Dataset (~8 MB)...")
    raw = urllib.request.urlopen(URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
        t.extractall(DATA_DIR)
    return marker


def parse_review_line(line):
    """'feat:count ... #label#:positive' -> (dict feat->count, label int)."""
    feats, label = {}, None
    for tok in line.split():
        key, _, val = tok.rpartition(":")
        if key == "#label#":
            label = 1 if val == "positive" else 0
        elif key:
            feats[key] = feats.get(key, 0) + int(val)
    return feats, label


def load_domain(root, domain):
    rows = []
    for pol in ("positive.review", "negative.review"):
        path = os.path.join(root, domain, pol)
        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                feats, label = parse_review_line(line.strip())
                if label is not None and feats:
                    rows.append((feats, label))
    return rows


def build_corpus():
    root = ensure_data()
    per_domain = {d: load_domain(root, d) for d in DOMAINS}
    freq = Counter()
    for rows in per_domain.values():
        for feats, _ in rows:
            freq.update(feats.keys())
    vocab = {w: i for i, (w, _) in enumerate(freq.most_common(VOCAB))}
    return per_domain, vocab


def vectorize(rows, vocab, domain_id):
    X = torch.zeros(len(rows), len(vocab), dtype=torch.float32)
    y = torch.zeros(len(rows), dtype=torch.long)
    r = torch.full((len(rows),), domain_id, dtype=torch.long)
    for i, (feats, label) in enumerate(rows):
        for w, c in feats.items():
            j = vocab.get(w)
            if j is not None:
                X[i, j] = float(c)
        y[i] = label
    X = (X > 0).float()                                # binary bag-of-words
    return X, y, r


def make_splits(per_domain, vocab, domains, seed):
    Xs, ys, rs = [], [], []
    for did, d in enumerate(domains):
        X, y, r = vectorize(per_domain[d], vocab, did)
        Xs.append(X); ys.append(y); rs.append(r)
    X, y, r = torch.cat(Xs), torch.cat(ys), torch.cat(rs)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    X, y, r = X[perm], y[perm], r[perm]
    n_te = len(X) // 5
    return (X[n_te:], y[n_te:], r[n_te:]), (X[:n_te], y[:n_te], r[:n_te])


# --------------------------------------------------------------------------- #
#  Bag-of-embeddings classifier (mean feature embedding reveals the domain)
# --------------------------------------------------------------------------- #
class BoWClassifier(nn.Module):
    def __init__(self, mode, vocab_size):
        super().__init__()
        self.mode = mode
        self.emb = nn.Linear(vocab_size, EMB, bias=False)   # bag -> embedding
        self.fc1 = nn.Linear(EMB, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.head = nn.Linear(HID, 2)
        if mode == "cca":
            self.a1, self.a2 = CCA1D(EMB, HID), CCA1D(EMB, HID)
        else:
            self.a1 = self.a2 = nn.SiLU()

    def forward(self, bag):
        ctx = self.emb(bag) / (bag.sum(1, keepdim=True) + 1.0)   # mean-ish signature
        h = self.a1(self.fc1(ctx), ctx) if self.mode == "cca" else self.a1(self.fc1(ctx))
        h = self.a2(self.fc2(h), ctx) if self.mode == "cca" else self.a2(self.fc2(h))
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
    return {"acc": correct / total, "throughput": total / dt,
            "coeff_var": (mean_coeff_variance(model) if collect else None)}


def train(mode, data, vocab_size):
    Xtr, ytr = data[0][0], data[0][1]
    torch.manual_seed(SEED)
    model = BoWClassifier(mode, vocab_size).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    step = 0
    while step < STEPS:
        for xb, yb in batches(Xtr, ytr, gen, True):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= STEPS:
                break
    return model, n_par


def run(domains, tag, per_domain, vocab):
    global SEED
    raw = {"silu": [], "cca": [], "shuffle": []}
    coeff_vars, n_par_ref = [], {}
    models, te = {}, None
    for sd in SEEDS:
        SEED = sd
        tr, te = make_splits(per_domain, vocab, domains, sd)
        for mode in ("silu", "cca"):
            model, n_par = train(mode, (tr, te), len(vocab))
            ev = evaluate(model, te[0], te[1], collect=(mode == "cca"))
            models[mode] = model
            n_par_ref[mode] = n_par
            raw[mode].append(ev["acc"])
            if ev["coeff_var"] is not None:
                coeff_vars.append(ev["coeff_var"])
        sh = evaluate(models["cca"], te[0], te[1], shuffle_ctx=True)
        raw["shuffle"].append(sh["acc"])
    print(f"\n[{tag}] domains={domains} regimes={len(domains)} seeds={SEEDS}")
    out = {"task": "text_sentiment", "tag": tag, "n_regimes": len(domains),
           "domains": domains, "seeds": SEEDS, "models": {}}
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
    measure_cost(out, models, te, len(vocab))
    return out


def measure_cost(out, models, te, vocab_size):
    xb = te[0][:BATCH].to(DEVICE)
    yb = te[1][:BATCH].to(DEVICE)
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
    per_domain, vocab = build_corpus()
    print(f"domains={DOMAINS} reviews="
          f"{ {d: len(r) for d, r in per_domain.items()} } vocab={len(vocab)}")
    multi = run(DOMAINS, "multi-domain", per_domain, vocab)
    with open("results/text_sentiment.json", "w") as f:
        json.dump(multi, f, indent=2)
    single = run(DOMAINS[:1], "single-domain", per_domain, vocab)
    with open("results/text_sentiment_single.json", "w") as f:
        json.dump(single, f, indent=2)
    print("\nsaved results/text_sentiment.json (+_single)")


if __name__ == "__main__":
    main()

"""
text_gen_bpe.py — leak-free text generation with small BPE tokenizers.
======================================================================

The original text-generation benchmark (`cca_leakfree.py`) is char-level. Real
language models use sub-word tokenizers, so we repeat the *same* leak-free
protocol over **small BPE vocabularies** (2k / 4k / 8k) to check the CCA gain is
not an artefact of character tokenization.

Protocol (identical in spirit to the char-level run):
  * 32-regime cipher mixture of Tiny Shakespeare (4 fixed + 27 substitution
    ciphers + identity). Ciphers are applied at the **character** level (the
    regime), then each enciphered corpus is tokenized with a shared BPE.
  * A tiny attention window (BLOCK tokens) cannot identify the cipher; a
    strictly-past history of HIST tokens (mean token embedding) can — and only
    the CCA activation receives it. No future information is ever used.
  * Models: B0 (SiLU), B0H (SiLU + history added to the input), CCA
    (history-conditioned). Metric: validation perplexity (lower is better).

For every vocabulary we report quality (loss/PPL) and the cost block + viability
ratio, reusing the tested model/training in `sweep_variants.py`.

Writes results/text_gen_bpe.json.   Run: python text_gen_bpe.py
"""

import json
import os
import statistics

import torch
import torch.nn.functional as F

# leak-free configuration (token units); set before importing sweep_variants
os.environ.setdefault("NCIPH", "27")        # 5 fixed + 27 ciphers = 32 regimes
os.environ.setdefault("BLOCK", "16")        # 16-token attention window
os.environ.setdefault("HIST_LEN", "128")    # 128-token strictly-past history
os.environ.setdefault("ITERS", "3000")
os.environ.setdefault("DIM", "128")
os.environ.setdefault("LAYERS", "3")

import sweep_variants as sv  # noqa: E402
import metrics  # noqa: E402
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders  # noqa: E402

VOCABS = [int(v) for v in os.environ.get("VOCABS", "2000,4000,8000").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1337,1338,1339").split(",")]
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

CFGS = {
    "B0": {"act": "silu"},
    "B0H": {"act": "silu", "hist_add": True},
    "CCA": {"act": "cca", "ctx": "hist", "groups": 64,
            "learn_alpha": True, "deep_g": True, "alpha": 1.0, "gamma": 2.0},
}


def mstd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


def get_bpe(text, vocab_size):
    """Train (and cache) a byte-level BPE of the given size on the raw corpus."""
    path = f"data/bpe_{vocab_size}.json"
    if os.path.exists(path):
        return Tokenizer.from_file(path)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                  special_tokens=["[UNK]"], min_frequency=2)
    tok.train_from_iterator([text], trainer)
    tok.save(path)
    return tok


class BPEDataset(sv.CharDataset):
    """Multi-regime cipher mixture, tokenized with a shared BPE. Reuses
    CharDataset.get_batch (token-id pools + strictly-past history)."""

    def __init__(self, text, tok, mix=True):
        self.mix, self.tok = mix, tok
        texts = [f(text) for f in sv.REGIMES] if mix else [text]
        self.vocab_size = tok.get_vocab_size()
        self.stoi = None
        self.train, self.val = [], []
        for t in texts:
            ids = torch.tensor(tok.encode(t).ids, dtype=torch.long)
            n = int(0.9 * len(ids))
            self.train.append(ids[:n])
            self.val.append(ids[n:])


def cost_for(model, ds):
    """Cost block for a trained model on a validation batch."""
    gen = torch.Generator().manual_seed(sv.SEED)
    x, y, h = ds.get_batch("val", gen)
    n_par = sum(p.numel() for p in model.parameters())

    def fwd():
        return model(x, None, h)

    def step():
        _, loss = model(x, y, h)
        model.zero_grad(set_to_none=True)
        loss.backward()

    c = metrics.model_cost(model, sv.DEVICE, forward_fn=fwd, train_step_fn=step,
                           n_items=x.numel())
    c["params"] = n_par
    return c


def run_vocab(vocab_size, text):
    tok = get_bpe(text, vocab_size)
    print(f"\n=== BPE vocab {vocab_size} (actual {tok.get_vocab_size()}) ===")
    acc = {k: {"loss": [], "ppl": []} for k in ["B0", "B0H", "CCA", "SHUF"]}
    params, costs = {}, {}
    for seed in SEEDS:
        sv.SEED = seed
        ds = BPEDataset(text, tok, mix=True)
        for name, cfg in CFGS.items():
            model, ev, sh, n_par, tt = sv.run(f"{name}-v{vocab_size}-s{seed}", cfg, ds)
            acc[name]["loss"].append(ev["val_loss"])
            acc[name]["ppl"].append(ev["ppl"])
            params[name] = n_par
            if name == "CCA":
                acc["SHUF"]["loss"].append(sh["val_loss"])
                acc["SHUF"]["ppl"].append(sh["ppl"])
            if seed == SEEDS[0]:
                costs[name] = cost_for(model, ds)
    stats = {k: {m: mstd(v[m]) for m in v} for k, v in acc.items()}
    b, c = stats["B0"], stats["CCA"]
    gp = 100 * (b["ppl"][0] - c["ppl"][0]) / b["ppl"][0]
    base_par = costs["B0"]["params"]
    for name in costs:
        costs[name]["params_overhead_pct"] = metrics.overhead_pct(
            costs[name]["params"], base_par)
    viab = metrics.viability(gp, costs["B0"]["throughput"], costs["CCA"]["throughput"])
    out = {
        "vocab_size": tok.get_vocab_size(), "seeds": SEEDS,
        "block": sv.BLOCK_SIZE, "hist": sv.HIST_LEN, "iters": sv.MAX_ITERS,
        "models": {
            "silu": {"val_loss": b["loss"][0], "val_loss_std": b["loss"][1],
                     "ppl": b["ppl"][0], "ppl_std": b["ppl"][1],
                     "cost": costs["B0"]},
            "silu_hist": {"val_loss": stats["B0H"]["loss"][0],
                          "ppl": stats["B0H"]["ppl"][0], "cost": costs["B0H"]},
            "cca": {"val_loss": c["loss"][0], "val_loss_std": c["loss"][1],
                    "ppl": c["ppl"][0], "ppl_std": c["ppl"][1],
                    "cost": costs["CCA"]},
            "shuffle": {"val_loss": stats["SHUF"]["loss"][0],
                        "ppl": stats["SHUF"]["ppl"][0]},
        },
        "improvement_pct": gp, "viability": viab,
    }
    print(f"  -> vocab {tok.get_vocab_size()}: PPL {b['ppl'][0]:.2f} -> "
          f"{c['ppl'][0]:.2f} ({gp:+.1f}%)  R={viab['viability_ratio']:.2f} "
          f"({'rentable' if viab['rentable'] else 'not rentable'})")
    return out


def main():
    text = sv.load_text()
    results = {"task": "text_gen_bpe",
               "dataset": "Tiny Shakespeare (32-regime cipher mixture, BPE)",
               "note": "leak-free; small BPE vocabularies", "by_vocab": {}}
    for v in VOCABS:
        results["by_vocab"][str(v)] = run_vocab(v, text)
    with open("results/text_gen_bpe.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/text_gen_bpe.json")


if __name__ == "__main__":
    main()

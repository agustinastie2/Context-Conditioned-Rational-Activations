"""
run_all.py — reproduce the entire CCA multi-modality study end to end.
======================================================================

Runs the four experiments (each multi-seed) and regenerates every figure and
table, then leaves results/ and figs/ populated for PAPER.md.

    python run_all.py            # full study, regenerates results/ and figs/
    FAST=1 python run_all.py     # 1 seed, fewer steps (quick smoke test)

Scripts are resolved relative to this file (so it works whether it lives in
src/ or the repo root), and every experiment runs with the repo root as its
working directory so results/, figs/ and data/ always resolve there.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# repo root = the directory that contains PAPER.md (HERE if run flat, else parent).
# PAPER.md only ever lives at the repo root, so this is unambiguous even if a
# stray results/ dir was created next to the scripts.
ROOT = HERE if (HERE / "PAPER.md").exists() else HERE.parent

FAST = os.environ.get("FAST", "0") == "1"
# (label, script, extra-env). image generation runs three datasets.
STEPS = [
    ("Text generation (leak-free, char)", "cca_leakfree.py", {}),
    ("Text generation (BPE 2k/4k/8k)", "text_gen_bpe.py", {}),
    ("Image classification", "image_cls.py", {}),
    ("Text classification", "text_cls.py", {}),
    ("Image generation (MNIST)", "image_gen.py", {"DATASET": "mnist"}),
    ("Image generation (Fashion-MNIST)", "image_gen.py", {"DATASET": "fashion"}),
    ("Image generation (CIFAR-10 grey)", "image_gen.py", {"DATASET": "cifar"}),
    ("Digits multi-domain (MNIST+SVHN+USPS)", "digits_multidomain.py", {}),
    ("Algorithmic reasoning (keyed remap)", "algo_modular.py", {}),
    ("Audio spoken digits (FSDD)", "audio_digits.py", {}),
    ("Text sentiment (multi-domain)", "text_sentiment_multidomain.py", {}),
    ("CCA efficiency sweep (Pareto)", "cca_efficiency.py", {}),
    ("Figures + tables", "make_figures.py", {}),
]


def main():
    base = dict(os.environ)
    if FAST:
        base["SEEDS"] = "1337"
        base["EPOCHS"] = "3"
        base["STEPS"] = "800"
        base["VOCABS"] = "2000"
    for name, script, extra in STEPS:
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
        env = {**base, **extra}
        r = subprocess.run([sys.executable, str(HERE / script)], env=env, cwd=ROOT)
        if r.returncode != 0:
            print(f"FAILED: {name}")
            sys.exit(r.returncode)
    print("\nAll experiments complete. See results/ and figs/, then PAPER.md.")


if __name__ == "__main__":
    main()

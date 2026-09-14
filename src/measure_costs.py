"""Re-measure the cost side of the confirmatory comparison on the local GPU.

`cca_fresh_confirmation.py` trains *and* profiles in one pass.  Accuracy is a
property of the experiment and is already settled over seeds 3001--3010; the
cost block, however, is a property of the *machine*, and the recorded run left
one cell empty: `digits/cca_hinge` could not be compiled because the Windows
Triton launcher failed on its generated C wrapper.

This script re-runs only `profile_costs()` -- a pure microbenchmark, no
training -- against the already-resolved budgets, so the whole cost table comes
from one homogeneous machine and the missing compiled measurement is filled in.
Output goes to a new directory; the published confirmation directory and its
manifest are left untouched.

Run:  python src/measure_costs.py [--budgets ...] [--output results/cca_cost_l4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from cca_fresh_confirmation import METHODS, atomic_json
from extended_comparison import profile_costs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="results/cca_hinge_confirmation/budgets.json",
                        help="Budgets resolved by the confirmation run; reused verbatim so the "
                             "profiled configurations are exactly the published ones.")
    parser.add_argument("--output", default="results/cca_cost_l4")
    args = parser.parse_args()

    budgets = json.loads((ROOT / args.budgets).read_text(encoding="utf-8"))
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    print(f"profiling {len(METHODS)} methods x 2 tasks on "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    costs = profile_costs(budgets, methods=METHODS)
    atomic_json(output / "costs.json", costs)

    for task, block in costs.items():
        print(f"\n{task}")
        for method, value in block.items():
            compiled = value["compiled_forward_backward_ms"]
            speedup = value["compiled_speedup"]
            print(f"  {method:12s} P={value['params']:6d} F={value['analytical_flops_per_sample']:8d} "
                  f"eager={value['eager_forward_backward_ms']:7.3f}ms "
                  f"compiled={'FAILED' if compiled is None else f'{compiled:7.3f}ms'} "
                  f"speedup={'n/a' if speedup is None else f'{speedup:.2f}x'}")
            if value["compile_error"]:
                print(f"    compile_error: {value['compile_error'][:120]}")

    atomic_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "budgets_source": args.budgets, "methods": list(METHODS),
        "note": ("Cost only. Accuracy for these configurations comes from "
                 "results/cca_hinge_confirmation (seeds 3001-3010) and is not re-run here, "
                 "because accuracy is hardware-independent and cost is not."),
    })
    names = ("costs.json", "environment.json")
    manifest = [f"{hashlib.sha256((output/name).read_bytes()).hexdigest()}  {name}" for name in names]
    (output / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="ascii")
    failures = [f"{task}/{method}" for task, block in costs.items()
                for method, value in block.items() if value["compile_error"]]
    print(f"\nwrote {output}; compile failures: {failures or 'none'}")


if __name__ == "__main__":
    main()

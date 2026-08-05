"""Paired, resumable comparison over every quantitative condition in main.tex.

The task adapters call the original paper implementations directly.  Thus data,
splits, architectures, optimizers, context construction, and metrics stay
identical; only the activation family changes.  The parent process launches one
isolated worker per task/seed, checkpoints every method immediately, and can be
resumed after a timeout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from extended_comparison import bootstrap_ci, exact_paired_permutation, holm_adjust


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = tuple(range(4001, 4011))
CORE_METHODS = ("silu", "rational", "dyrelu", "meta_acon", "cca_lite", "cca_hinge")
NAMES = {
    "silu": "SiLU", "b0h": "SiLU+Context (B0H)",
    "rational": "Static Rational", "dyrelu": "Dynamic ReLU-B",
    "meta_acon": "Meta-ACON", "cca_lite": "CCA-Lite", "cca_hinge": "CCA-Hinge",
}

# These are the quantitative conditions actually used by main.tex: eight main
# tasks, the three image sources, three BPE robustness settings, and two explicit
# single-regime null controls used by Fig. 2.
TASKS: dict[str, dict[str, Any]] = {
    "text_char": {"metric": "ppl", "higher": False, "b0h": True,
                  "context": "mean embedding of strictly-past 512-char history"},
    "text_bpe_2000": {"metric": "ppl", "higher": False, "b0h": True,
                      "context": "mean embedding of strictly-past 128-token history"},
    "text_bpe_4000": {"metric": "ppl", "higher": False, "b0h": True,
                      "context": "mean embedding of strictly-past 128-token history"},
    "text_bpe_8000": {"metric": "ppl", "higher": False, "b0h": True,
                      "context": "mean embedding of strictly-past 128-token history"},
    "image_gen_mnist": {"metric": "bits_per_dim", "higher": False, "b0h": True,
                        "context": "strictly-causal running grey-level histogram"},
    "image_gen_fashion": {"metric": "bits_per_dim", "higher": False, "b0h": True,
                          "context": "strictly-causal running grey-level histogram"},
    "image_gen_cifar": {"metric": "bits_per_dim", "higher": False, "b0h": True,
                        "context": "strictly-causal running grey-level histogram"},
    "image_cls": {"metric": "accuracy", "higher": True, "b0h": False,
                  "context": "global-average pooling of each convolution preactivation"},
    "image_cls_single": {"metric": "accuracy", "higher": True, "b0h": False,
                         "context": "global-average pooling; identity regime only"},
    "text_cls": {"metric": "accuracy", "higher": True, "b0h": False,
                 "context": "mean embedding of the complete input window"},
    "text_cls_single": {"metric": "accuracy", "higher": True, "b0h": False,
                        "context": "mean embedding; one cipher only"},
    "digits_multidomain": {"metric": "accuracy", "higher": True, "b0h": False,
                           "context": "global-average pooling; no domain identity"},
    "algo_modular": {"metric": "accuracy", "higher": True, "b0h": True,
                     "context": "encoded substitution key"},
    "audio_digits": {"metric": "accuracy", "higher": True, "b0h": False,
                     "context": "global-average pooling of log-STFT preactivation"},
    "text_sentiment": {"metric": "accuracy", "higher": True, "b0h": False,
                       "context": "normalized bag-of-words embedding; no domain identity"},
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def methods_for(task: str) -> tuple[str, ...]:
    return CORE_METHODS + (("b0h",) if TASKS[task]["b0h"] else ())


def _mode(method: str) -> str:
    if method == "silu":
        return "silu"
    if method == "b0h":
        return "silu_hist"
    os.environ["CCA_VARIANT"] = method
    os.environ.setdefault("ACT_GROUPS", "16")
    os.environ.setdefault("ACT_HIDDEN", "16")
    return "cca"


def _classification_data(task: str, seed: int):
    """Return module, train args, evaluation args, and metadata."""
    if task.startswith("text_cls"):
        import text_cls as m
        m.SEED = seed
        regimes = 1 if task.endswith("single") else 8
        sources = m.make_sources(m.N_CLASSES, seed)
        ciphers = m.make_ciphers(regimes, seed + 1)
        xtr, ytr = m.gen_dataset(m.N_TRAIN, sources, ciphers, seed)
        xte, yte = m.gen_dataset(m.N_TEST, sources, ciphers, seed + 99)
        data = (xtr, ytr, xte, yte)
        return m, (data,), (xte, yte), {"n_regimes": regimes}
    if task.startswith("image_cls"):
        import image_cls as m
        m.SEED = seed
        data = m.build_data(not task.endswith("single"))
        return m, (data,), data[1], {"n_regimes": data[2]}
    if task == "digits_multidomain":
        import digits_multidomain as m
        m.SEED = seed
        data = m.build_data(True)
        return m, (data,), data[1], {"n_regimes": data[2], "domains": m.DOMAINS}
    if task == "algo_modular":
        import algo_modular as m
        m.SEED = seed
        keys = m.make_keys(m.N_KEYS, seed + 5)
        train = m.gen_dataset(m.N_TRAIN, keys, seed)
        test = m.gen_dataset(m.N_TEST, keys, seed + 99)
        return m, (train, m.N_KEYS), (test,), {"n_regimes": m.N_KEYS}
    if task == "audio_digits":
        import audio_digits as m
        m.SEED = seed
        base = m.load_dataset()
        data = m.build_data(True, *base)
        return m, (data,), data[1], {"n_regimes": data[2]}
    if task == "text_sentiment":
        import text_sentiment_multidomain as m
        m.SEED = seed
        per_domain, vocab = m.build_corpus()
        train, test = m.make_splits(per_domain, vocab, m.DOMAINS, seed)
        return m, ((train, test), len(vocab)), (test[0], test[1]), {
            "n_regimes": len(m.DOMAINS), "domains": m.DOMAINS,
        }
    raise KeyError(task)


def _run_classification(task: str, method: str, seed: int, cached):
    module, train_args, eval_args, metadata = cached
    mode = _mode(method)
    started = time.monotonic()
    trained = module.train(mode, *train_args)
    model = trained[0]
    params = trained[-1]
    if task == "algo_modular":
        evaluation = module.evaluate(model, *eval_args, collect=method.startswith("cca_"))
    else:
        evaluation = module.evaluate(model, *eval_args, collect=method.startswith("cca_"))
    shuffle = None
    if method in ("cca_lite", "cca_hinge"):
        shuffle = module.evaluate(model, *eval_args, shuffle_ctx=True).get("acc")
    return {
        "value": evaluation["acc"], "shuffle_value": shuffle,
        "params": int(params), "train_seconds": time.monotonic() - started,
        "eval_throughput": evaluation.get("throughput"), **metadata,
    }


def _run_image_generation(task: str, method: str, seed: int, cached):
    module, data = cached
    module.SEED = seed
    mode = _mode(method)
    started = time.monotonic()
    model, _, params = module.train(mode, data[0])
    evaluation = module.evaluate(model, data[1], collect=method.startswith("cca_"))
    shuffle = None
    if method in ("cca_lite", "cca_hinge"):
        shuffle = module.evaluate(model, data[1], shuffle_ctx=True)["bpd"]
    return {
        "value": evaluation["bpd"], "shuffle_value": shuffle,
        "params": int(params), "train_seconds": time.monotonic() - started,
        "eval_throughput": evaluation.get("throughput"), "n_regimes": data[2],
        "training_steps": module.STEPS, "mixed_precision": module.USE_AMP,
    }


def _language_cfg(method: str) -> dict[str, Any]:
    if method == "silu":
        return {"act": "silu"}
    if method == "b0h":
        return {"act": "silu", "hist_add": True}
    os.environ["CCA_VARIANT"] = method
    return {"act": "cca", "ctx": "hist", "groups": 64,
            "learn_alpha": True, "deep_g": True, "alpha": 1.0, "gamma": 2.0}


def _run_language(task: str, method: str, seed: int, cached):
    sv, dataset = cached
    sv.SEED = seed
    started = time.monotonic()
    model, evaluation, shuffled, params, _ = sv.run(
        f"{task}-{method}-s{seed}", _language_cfg(method), dataset)
    shuffle = (shuffled["ppl"] if shuffled is not None and
               method in ("cca_lite", "cca_hinge") else None)
    return {
        "value": evaluation["ppl"], "shuffle_value": shuffle,
        "params": int(params), "train_seconds": time.monotonic() - started,
        "eval_throughput": evaluation.get("tok_s"), "n_regimes": len(dataset.train),
    }


def worker(task: str, seed: int, methods: list[str], output: Path) -> None:
    existing = json.loads(output.read_text()) if output.exists() else {"runs": []}
    completed = {run["method"] for run in existing["runs"]}
    if task.startswith("image_gen_"):
        dataset_name = task.removeprefix("image_gen_")
        os.environ["DATASET"] = dataset_name
        import image_gen as module
        module.SEED = seed
        data = module.build_data(True)
        if module.DEVICE == "cuda" and os.environ.get("PAPER_STAGE_DATA", "0") == "1":
            data = (data[0].to(module.DEVICE), data[1].to(module.DEVICE), data[2])
        cached, runner = (module, data), _run_image_generation
    elif task == "text_char":
        os.environ.setdefault("NCIPH", "27")
        os.environ.setdefault("BLOCK", "16")
        os.environ.setdefault("HIST_LEN", "512")
        os.environ.setdefault("ITERS", "3000")
        import sweep_variants as sv
        cached, runner = (sv, sv.CharDataset(sv.load_text(), mix=True)), _run_language
    elif task.startswith("text_bpe_"):
        os.environ.setdefault("NCIPH", "27")
        os.environ.setdefault("BLOCK", "16")
        os.environ.setdefault("HIST_LEN", "128")
        os.environ.setdefault("ITERS", "3000")
        import text_gen_bpe as bpe
        vocab = int(task.rsplit("_", 1)[1])
        text = bpe.sv.load_text()
        cached = (bpe.sv, bpe.BPEDataset(text, bpe.get_bpe(text, vocab), mix=True))
        runner = _run_language
    else:
        cached, runner = _classification_data(task, seed), _run_classification

    for method in methods:
        if method in completed:
            continue
        print(f"WORKER {task} seed={seed} method={method}", flush=True)
        details = runner(task, method, seed, cached)
        run = {"task": task, "method": method, "seed": seed,
               "metric": TASKS[task]["metric"], "higher_is_better": TASKS[task]["higher"],
               "context_origin": TASKS[task]["context"], **details}
        existing["runs"].append(run)
        completed.add(method)
        atomic_json(output, existing)
        print(f"RESULT {task}/{method}/s{seed}: {details['value']:.6f}", flush=True)


def summarize(runs: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {"seeds": seeds, "tasks": {}}
    for task, spec in TASKS.items():
        task_runs = [run for run in runs if run["task"] == task]
        methods, by_method = {}, {}
        for method in methods_for(task):
            selected = sorted((r for r in task_runs if r["method"] == method),
                              key=lambda r: r["seed"])
            values = [r["value"] for r in selected]
            by_method[method] = {r["seed"]: r["value"] for r in selected}
            if values:
                hinge_params = [r["params"] for r in task_runs if r["method"] == "cca_hinge"]
                target = statistics.mean(hinge_params) if hinge_params else None
                methods[method] = {
                    "n": len(values), "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "ci95": bootstrap_ci(values),
                    "mean_params": statistics.mean(r["params"] for r in selected),
                    "params_vs_hinge_pct": (100 * (statistics.mean(r["params"] for r in selected)
                                                   / target - 1) if target else None),
                    "mean_train_seconds": statistics.mean(r["train_seconds"] for r in selected),
                    "mean_eval_throughput": statistics.mean(r["eval_throughput"] for r in selected),
                }
        comparisons, raw_p = {}, {}
        hinge = by_method.get("cca_hinge", {})
        for method, values in by_method.items():
            if method == "cca_hinge":
                continue
            common = sorted(set(hinge) & set(values))
            if not common:
                continue
            sign = 1.0 if spec["higher"] else -1.0
            gains = [sign * (hinge[seed] - values[seed]) for seed in common]
            sd = statistics.stdev(gains) if len(gains) > 1 else 0.0
            raw_p[method] = exact_paired_permutation(gains)
            comparisons[method] = {
                "n_pairs": len(common), "mean_gain": statistics.mean(gains),
                "ci95_gain": bootstrap_ci(gains), "p_exact_two_sided": raw_p[method],
                "cohens_dz": statistics.mean(gains) / sd if sd else 0.0,
            }
        adjusted = holm_adjust(raw_p)
        for method, pvalue in adjusted.items():
            comparisons[method]["p_holm"] = pvalue
            comparisons[method]["significant_0_05"] = pvalue < 0.05
        output["tasks"][task] = {"metric": spec["metric"], "higher": spec["higher"],
                                 "methods": methods, "cca_hinge_comparisons": comparisons}
    return output


def write_artifacts(directory: Path, runs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    atomic_json(directory / "summary.json", summary)
    fields = sorted({key for run in runs for key in run if not isinstance(run[key], (dict, list))})
    with (directory / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run.get(field) for field in fields})
    lines = ["# Complete paper experiment matrix", "",
             f"Paired seeds: {', '.join(map(str, summary['seeds']))}.", ""]
    generation_protocols = sorted({
        (run.get("training_steps"), run.get("mixed_precision"))
        for run in runs if run.get("training_steps") is not None
    })
    if generation_protocols:
        descriptions = [f"{steps} steps, " + ("BF16" if amp else "FP32")
                        for steps, amp in generation_protocols]
        lines += ["Image-generation protocol(s): " + "; ".join(descriptions) + ".", ""]
    for task, block in summary["tasks"].items():
        if not block["methods"] or max(value["n"] for value in block["methods"].values()) < 2:
            continue
        lines += [f"## {task}", "", f"Metric: `{block['metric']}` " +
                  ("(higher is better)." if block["higher"] else "(lower is better)."), "",
                  "| Method | n | Mean ± sd | 95% CI | Params vs Hinge |", "|---|---:|---:|---:|---:|"]
        for method in methods_for(task):
            if method not in block["methods"]:
                continue
            value = block["methods"][method]
            params = value["params_vs_hinge_pct"]
            lines.append(f"| {NAMES[method]} | {value['n']} | {value['mean']:.5f} ± "
                         f"{value['std']:.5f} | [{value['ci95'][0]:.5f}, "
                         f"{value['ci95'][1]:.5f}] | "
                         f"{params:+.2f}% |" if params is not None else "n/a |")
        lines += ["", "| Comparator | CCA-Hinge paired gain | p-Holm |", "|---|---:|---:|"]
        for method, value in block["cca_hinge_comparisons"].items():
            lines.append(f"| {NAMES[method]} | {value['mean_gain']:+.5f} | "
                         f"{value['p_holm']:.5f} |")
        lines.append("")
    (directory / "EVIDENCE.md").write_text("\n".join(lines), encoding="utf-8")
    environment = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": (torch.cuda.get_device_name(0)
                                             if torch.cuda.is_available() else None),
        "tasks": TASKS, "methods": list(CORE_METHODS),
    }
    atomic_json(directory / "environment.json", environment)
    names = ("checkpoint.json", "summary.json", "runs.csv", "EVIDENCE.md", "environment.json")
    manifest = [f"{hashlib.sha256((directory/name).read_bytes()).hexdigest()}  {name}"
                for name in names]
    (directory / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="ascii")


def ingest_workers(directory: Path, checkpoint: dict[str, Any]) -> None:
    indexed = {(r["task"], r["method"], r["seed"]): r for r in checkpoint["runs"]}
    for path in (directory / "workers").glob("*.json"):
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            indexed[(run["task"], run["method"], run["seed"])] = run
    checkpoint["runs"] = sorted(indexed.values(), key=lambda r: (r["task"], r["seed"], r["method"]))
    atomic_json(directory / "checkpoint.json", checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--methods", default=",".join(CORE_METHODS) + ",b0h")
    parser.add_argument("--output", default="results/paper_full_comparison")
    parser.add_argument("--timeout-minutes", type=float, default=110.0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.worker:
        worker(args.task, args.seed, args.methods.split(","), Path(args.worker_output))
        return

    seeds = [int(value) for value in args.seeds.split(",") if value]
    requested_tasks = [value for value in args.tasks.split(",") if value]
    requested_methods = set(value for value in args.methods.split(",") if value)
    directory = ROOT / args.output
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workers").mkdir(exist_ok=True)
    checkpoint_path = directory / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"runs": []}
    ingest_workers(directory, checkpoint)
    started = time.monotonic()
    for task in requested_tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task}")
        eligible = [m for m in methods_for(task) if m in requested_methods]
        for seed in seeds:
            ingest_workers(directory, checkpoint)
            complete = {(r["task"], r["method"], r["seed"]) for r in checkpoint["runs"]}
            missing = [m for m in eligible if (task, m, seed) not in complete]
            if not missing:
                continue
            if (time.monotonic() - started) / 60 >= args.timeout_minutes:
                print("TIMEOUT budget reached; state is resumable.", flush=True)
                summary = summarize(checkpoint["runs"], seeds)
                write_artifacts(directory, checkpoint["runs"], summary)
                return
            worker_path = directory / "workers" / f"{task}_s{seed}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--task", task, "--seed", str(seed), "--methods", ",".join(missing),
                       "--worker-output", str(worker_path)]
            print(f"LAUNCH {task} seed={seed} methods={','.join(missing)}", flush=True)
            completed_worker = subprocess.run(command, cwd=ROOT)
            ingest_workers(directory, checkpoint)
            if completed_worker.returncode:
                print(f"FAILED {task}/seed={seed}; partial worker state retained", flush=True)
                summary = summarize(checkpoint["runs"], seeds)
                write_artifacts(directory, checkpoint["runs"], summary)
                raise SystemExit(completed_worker.returncode)
            print(f"PROGRESS {len(checkpoint['runs'])} completed cells", flush=True)
    summary = summarize(checkpoint["runs"], seeds)
    write_artifacts(directory, checkpoint["runs"], summary)
    expected = sum(len([m for m in methods_for(t) if m in requested_methods])
                   for t in requested_tasks) * len(seeds)
    scoped = sum(r["task"] in requested_tasks and r["method"] in requested_methods and
                 r["seed"] in seeds for r in checkpoint["runs"])
    print(f"COMPLETE {scoped}/{expected} cells in "
          f"{(time.monotonic()-started)/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()

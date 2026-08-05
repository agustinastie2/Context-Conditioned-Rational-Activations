"""Fresh-seed confirmation of the validation-selected CCA-Hinge preset."""

from __future__ import annotations

import json
import hashlib
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from extended_comparison import (
    ModelConfig, _candidate_configs, analytical_flops, atomic_json, bootstrap_ci,
    count_parameters, exact_paired_permutation, holm_adjust, make_model,
    train_and_evaluate,
    profile_costs,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(3001, 3011))
METHODS = ("silu_budget", "rational", "dyrelu", "meta_acon", "cca_lite", "cca_hinge")
NAMES = {
    "silu_budget": "SiLU (budget-matched)", "rational": "Static Rational",
    "dyrelu": "Dynamic ReLU-B", "meta_acon": "Meta-ACON",
    "cca_lite": "CCA-Lite", "cca_hinge": "CCA-Hinge",
}


def hinge_config(task: str) -> ModelConfig:
    if task == "text":
        return ModelConfig("cca_hinge", hidden=256, groups=16, g_hidden=16)
    return ModelConfig("cca_hinge", channels1=16, channels2=32, groups=16, g_hidden=16)


def resolve_budgets():
    result = {}
    for task in ("text", "digits"):
        target = hinge_config(task)
        target_params = count_parameters(make_model(task, target))
        target_flops = analytical_flops(task, target)
        entries = {}
        for method in METHODS:
            configs = [target] if method == "cca_hinge" else list(_candidate_configs(task, method))
            feasible = []
            for config in configs:
                params = count_parameters(make_model(task, config))
                flops = analytical_flops(task, config)
                if params <= 1.05 * target_params and flops <= 1.10 * target_flops:
                    score = abs(1 - params / target_params) + abs(1 - flops / target_flops)
                    feasible.append((score, -params, config, params, flops))
            if not feasible:
                raise RuntimeError(f"No feasible configuration for {task}/{method}")
            _, _, config, params, flops = min(feasible, key=lambda value: value[:2])
            entries[method] = {
                "config": asdict(config), "params": params, "flops_per_sample": flops,
                "params_vs_target_pct": 100 * (params / target_params - 1),
                "flops_vs_target_pct": 100 * (flops / target_flops - 1),
            }
        result[task] = {"target": "cca_hinge", "target_params": target_params,
                        "target_flops_per_sample": target_flops, "methods": entries}
    return result


def summarize(runs):
    output = {"seeds": list(SEEDS), "tasks": {}}
    for task in ("text", "digits"):
        methods, comparisons, raw_p = {}, {}, {}
        indexed = {}
        for method in METHODS:
            selected = sorted((run for run in runs if run["task"] == task and run["method"] == method),
                              key=lambda run: run["seed"])
            values = [run["accuracy"] for run in selected]
            indexed[method] = {run["seed"]: run["accuracy"] for run in selected}
            methods[method] = {"n": len(values), "mean_accuracy": statistics.mean(values),
                               "std_accuracy": statistics.stdev(values),
                               "ci95_accuracy": bootstrap_ci(values)}
            if task == "digits":
                methods[method]["per_domain_mean"] = {
                    domain: statistics.mean(run["per_domain_accuracy"][domain] for run in selected)
                    for domain in ("mnist", "svhn", "usps")}
        hinge = indexed["cca_hinge"]
        for method in METHODS:
            if method == "cca_hinge":
                continue
            common = sorted(set(hinge) & set(indexed[method]))
            differences = [hinge[seed] - indexed[method][seed] for seed in common]
            sd = statistics.stdev(differences)
            p = exact_paired_permutation(differences); raw_p[method] = p
            comparisons[method] = {
                "mean_gain_accuracy_points": 100 * statistics.mean(differences),
                "ci95_gain_accuracy_points": [100 * value for value in bootstrap_ci(differences)],
                "cohens_dz": statistics.mean(differences) / sd if sd else 0.0,
                "p_exact_two_sided": p,
            }
        adjusted = holm_adjust(raw_p)
        for method, p in adjusted.items():
            comparisons[method]["p_holm"] = p
            comparisons[method]["significant_0_05"] = p < 0.05
        output["tasks"][task] = {"methods": methods, "cca_hinge_comparisons": comparisons}
    return output


def report(path: Path, summary, budgets, costs):
    lines = ["# Fresh-seed CCA-Hinge confirmation", "",
             "Preset selected on seeds 2001–2003; confirmation uses untouched seeds 3001–3010.", ""]
    for task in ("text", "digits"):
        lines += [f"## {task.title()}", "", "| Method | Accuracy mean ± sd | 95% CI |",
                  "|---|---:|---:|"]
        for method in METHODS:
            value = summary["tasks"][task]["methods"][method]
            lo, hi = value["ci95_accuracy"]
            lines.append(f"| {NAMES[method]} | {100*value['mean_accuracy']:.2f} ± "
                         f"{100*value['std_accuracy']:.2f} | [{100*lo:.2f}, {100*hi:.2f}] |")
        lines += ["", "CCA-Hinge paired gain:", "", "| Comparator | Gain pp | 95% CI | p-Holm |",
                  "|---|---:|---:|---:|"]
        for method, value in summary["tasks"][task]["cca_hinge_comparisons"].items():
            lo, hi = value["ci95_gain_accuracy_points"]
            lines.append(f"| {NAMES[method]} | {value['mean_gain_accuracy_points']:+.2f} | "
                         f"[{lo:+.2f}, {hi:+.2f}] | {value['p_holm']:.5f} |")
        lines += ["", "| Method | Params vs CCA-Hinge | FLOPs vs CCA-Hinge |",
                  "|---|---:|---:|"]
        for method in METHODS:
            value = budgets[task]["methods"][method]
            lines.append(f"| {NAMES[method]} | {value['params_vs_target_pct']:+.2f}% | "
                         f"{value['flops_vs_target_pct']:+.2f}% |")
        lines += ["", "| Method | Eager train ms | Compiled train ms | Compiled samples/s |",
                  "|---|---:|---:|---:|"]
        for method in METHODS:
            value = costs[task][method]
            compiled = value["compiled_forward_backward_ms"]
            compiled_text = "n/a" if compiled is None else f"{compiled:.3f}"
            throughput = value["compiled_samples_per_second"]
            throughput_text = "n/a" if throughput is None else f"{throughput:.0f}"
            lines.append(f"| {NAMES[method]} | {value['eager_forward_backward_ms']:.3f} | "
                         f"{compiled_text} | {throughput_text} |")
        failed_compiles = [NAMES[method] for method in METHODS if costs[task][method]["compile_error"]]
        if failed_compiles:
            lines += ["", "Compiled timing is unavailable for " + ", ".join(failed_compiles) +
                      " because the local Windows Triton launcher could not compile its generated C "
                      "wrapper. Eager execution and all correctness/training runs completed; `costs.json` "
                      "retains the full compiler error for reproducibility."]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    output = ROOT / "results" / "cca_hinge_confirmation"
    output.mkdir(parents=True, exist_ok=True)
    budgets = resolve_budgets(); atomic_json(output / "budgets.json", budgets)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"runs": []}
    runs = checkpoint["runs"]
    completed = {(run["task"], run["method"], run["seed"]) for run in runs}
    total = 2 * len(METHODS) * len(SEEDS); done = len(completed)
    started = time.monotonic()
    for task in ("text", "digits"):
        for seed in SEEDS:
            for method in METHODS:
                key = (task, method, seed)
                if key in completed:
                    continue
                config = ModelConfig(**budgets[task]["methods"][method]["config"])
                print(f"[{done+1}/{total}] {task}/{method}/seed={seed}", flush=True)
                run = train_and_evaluate(task, config, seed, False)
                runs.append(run); completed.add(key); done += 1
                atomic_json(checkpoint_path, checkpoint)
                print(f"  accuracy={100*run['accuracy']:.2f}%", flush=True)
    summary = summarize(runs)
    atomic_json(output / "summary.json", summary)
    costs = profile_costs(budgets, methods=METHODS)
    atomic_json(output / "costs.json", costs)
    report(output / "EVIDENCE.md", summary, budgets, costs)
    manifest_names = ("budgets.json", "checkpoint.json", "summary.json", "costs.json", "EVIDENCE.md")
    manifest = []
    for name in manifest_names:
        digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
        manifest.append(f"{digest}  {name}")
    (output / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="ascii")
    print(f"complete in {(time.monotonic()-started)/60:.1f} minutes")


if __name__ == "__main__":
    main()

"""Confirmatory, budget-controlled CCA-Lite comparison.

Runs a controlled synthetic text task and real multi-domain digit recognition
with paired seeds. Results are checkpointed after every method/seed and can be
resumed safely. This script never reads domain identity as model context.

Full run (10 seeds, 110-minute hard deadline):
    python src/extended_comparison.py
Fast verification:
    python src/extended_comparison.py --smoke --skip-cost
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Python installations on Windows may not inherit the OS certificate store.
# Use certifi's Mozilla CA bundle when available; certificate verification stays
# enabled and torchvision additionally verifies every dataset's published MD5.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import digits_multidomain as digits
import text_cls as text
from adaptive_activations import METHODS, ActivationConfig, build_activation


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEEDS = tuple(range(1337, 1347))
PRIMARY_METHODS = tuple(METHODS)
DISPLAY_NAMES = {
    "silu": "SiLU (original)",
    "silu_budget": "SiLU (budget-matched)",
    "rational": "Static Rational",
    "dyrelu": "Dynamic ReLU-B",
    "meta_acon": "Meta-ACON",
    "cca_lite": "CCA-Lite",
}


@dataclass(frozen=True)
class ModelConfig:
    method: str
    hidden: int | None = None
    channels1: int | None = None
    channels2: int | None = None
    groups: int = 16
    g_hidden: int = 16
    lowrank: int = 0


class TextModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = int(config.hidden)
        self.config = config
        self.emb = nn.Embedding(text.VOCAB, text.EMB)
        self.fc1 = nn.Linear(text.EMB, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, text.N_CLASSES)
        ac = ActivationConfig(config.method, config.groups, config.g_hidden, config.lowrank)
        self.a1 = build_activation(ac, text.EMB, hidden)
        self.a2 = build_activation(ac, text.EMB, hidden)

    def forward(self, tokens: Tensor) -> Tensor:
        context = self.emb(tokens).mean(1)
        hidden = self.a1(self.fc1(context), context)
        hidden = self.a2(self.fc2(hidden), context)
        return self.head(hidden)


class DigitModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c1, c2 = int(config.channels1), int(config.channels2)
        self.config = config
        self.c1 = nn.Conv2d(1, c1, 3, padding=1)
        self.c2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc = nn.Linear(c2 * 7 * 7, 10)
        ac = ActivationConfig(config.method, config.groups, config.g_hidden, config.lowrank)
        self.a1 = build_activation(ac, c1, c1, spatial=True)
        self.a2 = build_activation(ac, c2, c2, spatial=True)

    def forward(self, images: Tensor) -> Tensor:
        pre1 = self.c1(images)
        # Context is inferred only from the current feature map; domain labels
        # are never passed to the model.
        context1 = pre1.mean((2, 3))
        hidden = F.max_pool2d(self.a1(pre1, context1), 2)
        pre2 = self.c2(hidden)
        context2 = pre2.mean((2, 3))
        hidden = F.max_pool2d(self.a2(pre2, context2), 2)
        return self.fc(hidden.flatten(1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _activation_flops(method: str, elements: int, ctx: int, features: int,
                      groups: int, g_hidden: int, lowrank: int = 0) -> int:
    """Documented analytical operation estimate, including pointwise work."""
    groups = min(groups, features)
    while features % groups:
        groups -= 1
    pointwise = {
        "silu": 4, "silu_budget": 4, "rational": 20,
        "dyrelu": 8, "meta_acon": 9, "cca_lite": 22, "cca_hinge": 28,
    }[method] * elements
    outputs = {"dyrelu": 4, "meta_acon": 1, "cca_lite": 7, "cca_hinge": 9}.get(method, 0)
    if not outputs:
        generator = 0
    elif method == "cca_lite" and lowrank and lowrank < outputs * groups:
        generator = 2 * (ctx * g_hidden + g_hidden * lowrank + lowrank * outputs * groups)
    else:
        generator = 2 * (ctx * g_hidden + g_hidden * outputs * groups)
    pooling = elements if outputs else 0
    return pointwise + generator + pooling


def analytical_flops(task: str, config: ModelConfig) -> int:
    if task == "text":
        hidden = int(config.hidden)
        linear = 2 * (text.EMB * hidden + hidden * hidden + hidden * text.N_CLASSES)
        acts = 2 * _activation_flops(
            config.method, hidden, text.EMB, hidden, config.groups,
            config.g_hidden, config.lowrank
        )
        return linear + acts + text.LEN * text.EMB
    c1, c2 = int(config.channels1), int(config.channels2)
    backbone = (
        2 * 28 * 28 * c1 * 9
        + 2 * 14 * 14 * c1 * c2 * 9
        + 2 * c2 * 7 * 7 * 10
    )
    acts = _activation_flops(config.method, c1 * 28 * 28, c1, c1,
                             config.groups, config.g_hidden, config.lowrank)
    acts += _activation_flops(config.method, c2 * 14 * 14, c2, c2,
                              config.groups, config.g_hidden, config.lowrank)
    return backbone + acts


def make_model(task: str, config: ModelConfig) -> nn.Module:
    return TextModel(config) if task == "text" else DigitModel(config)


def _candidate_configs(task: str, method: str):
    if method == "silu":
        yield ModelConfig(method, hidden=256) if task == "text" else ModelConfig(
            method, channels1=16, channels2=32
        )
        return
    if method == "cca_lite":
        yield ModelConfig(method, hidden=256, groups=16, g_hidden=16) if task == "text" else ModelConfig(
            method, channels1=16, channels2=32, groups=16, g_hidden=16
        )
        return
    groups_values = (8, 16, 32, 64) if method in ("dyrelu", "meta_acon") else (16,)
    hidden_gen_values = (8, 16, 24, 32, 48, 64, 96) if method in ("dyrelu", "meta_acon") else (16,)
    if task == "text":
        for hidden in range(240, 305, 2):
            for groups, g_hidden in itertools.product(groups_values, hidden_gen_values):
                yield ModelConfig(method, hidden=hidden, groups=groups, g_hidden=g_hidden)
    else:
        for c1 in range(13, 23):
            for c2 in range(27, 43):
                for groups, g_hidden in itertools.product(groups_values, hidden_gen_values):
                    yield ModelConfig(method, channels1=c1, channels2=c2,
                                      groups=groups, g_hidden=g_hidden)


def resolve_budgets() -> dict[str, dict[str, Any]]:
    """Resolve configurations from structure/cost alone, without training data."""
    resolved: dict[str, dict[str, Any]] = {}
    for task in ("text", "digits"):
        target = next(_candidate_configs(task, "cca_lite"))
        target_params = count_parameters(make_model(task, target))
        target_flops = analytical_flops(task, target)
        param_cap, flop_cap = 1.05 * target_params, 1.10 * target_flops
        methods: dict[str, Any] = {}
        for method in PRIMARY_METHODS:
            candidates = []
            for config in _candidate_configs(task, method):
                params = count_parameters(make_model(task, config))
                flops = analytical_flops(task, config)
                if method == "silu" or (params <= param_cap and flops <= flop_cap):
                    # Closest joint utilization of the fixed resource caps.
                    score = abs(1 - params / target_params) + abs(1 - flops / target_flops)
                    candidates.append((score, -params, -flops, config, params, flops))
            if not candidates:
                raise RuntimeError(f"No budget-feasible configuration for {task}/{method}")
            _, _, _, config, params, flops = min(candidates, key=lambda item: item[:3])
            methods[method] = {
                "config": asdict(config), "params": params, "flops_per_sample": flops,
                "params_vs_target_pct": 100 * (params / target_params - 1),
                "flops_vs_target_pct": 100 * (flops / target_flops - 1),
                "within_budget": method == "silu" or (
                    params <= param_cap and flops <= flop_cap
                ),
            }
        resolved[task] = {
            "target_method": "cca_lite", "target_params": target_params,
            "target_flops_per_sample": target_flops, "parameter_cap_ratio": 1.05,
            "flop_cap_ratio": 1.10, "methods": methods,
        }
    return resolved


def configure_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


_LAST_TEXT_DATA: tuple[tuple[int, bool], Any] | None = None


def text_data(seed: int, smoke: bool):
    global _LAST_TEXT_DATA
    key = (seed, smoke)
    if _LAST_TEXT_DATA is not None and _LAST_TEXT_DATA[0] == key:
        return _LAST_TEXT_DATA[1]
    sources = text.make_sources(text.N_CLASSES, seed)
    ciphers = text.make_ciphers(8, seed + 1)
    n_train, n_test = (4096, 1024) if smoke else (text.N_TRAIN, text.N_TEST)
    train_x, train_y = text.gen_dataset(n_train, sources, ciphers, seed)
    test_x, test_y = text.gen_dataset(n_test, sources, ciphers, seed + 99)
    result = (train_x, train_y), (test_x, test_y)
    _LAST_TEXT_DATA = (key, result)
    return result


_DIGIT_DOMAIN_CACHE: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
_LAST_DIGIT_DATA: tuple[tuple[int, bool], Any] | None = None


def load_digit_domains() -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor]]:
    if _DIGIT_DOMAIN_CACHE:
        return _DIGIT_DOMAIN_CACHE
    try:
        for name in digits.DOMAINS:
            train_x, train_y = digits._domain(name, True)
            test_x, test_y = digits._domain(name, False)
            _DIGIT_DOMAIN_CACHE[name] = (
                digits._grey_resize(train_x), train_y.long(),
                digits._grey_resize(test_x), test_y.long(),
            )
    except Exception as error:
        raise RuntimeError(
            "Verified torchvision dataset download failed. No TLS verification "
            "bypass is used; fix the local CA/network and rerun."
        ) from error
    return _DIGIT_DOMAIN_CACHE


def digit_data(seed: int, smoke: bool):
    global _LAST_DIGIT_DATA
    key = (seed, smoke)
    if _LAST_DIGIT_DATA is not None and _LAST_DIGIT_DATA[0] == key:
        return _LAST_DIGIT_DATA[1]
    cached = load_digit_domains()
    n_train, n_test = (512, 256) if smoke else (digits.N_PER_DOMAIN, digits.N_TEST_PER)
    train_parts, test_parts, test_domains = [], [], []
    for domain_index, name in enumerate(digits.DOMAINS):
        train_x, train_y, test_x, test_y = cached[name]
        generator = torch.Generator().manual_seed(seed + domain_index)
        train_idx = torch.randperm(len(train_x), generator=generator)[:n_train]
        test_idx = torch.randperm(len(test_x), generator=generator)[:n_test]
        train_parts.append((train_x[train_idx], train_y[train_idx]))
        test_parts.append((test_x[test_idx], test_y[test_idx]))
        test_domains.append(torch.full((len(test_idx),), domain_index, dtype=torch.long))
    result = (
        (torch.cat([part[0] for part in train_parts]), torch.cat([part[1] for part in train_parts])),
        (torch.cat([part[0] for part in test_parts]), torch.cat([part[1] for part in test_parts]),
         torch.cat(test_domains)),
    )
    _LAST_DIGIT_DATA = (key, result)
    return result


def batches(x: Tensor, y: Tensor, seed: int, train: bool):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator) if train else torch.arange(len(x))
    for start in range(0, len(x), 256):
        index = order[start:start + 256]
        yield x[index].to(DEVICE), y[index].to(DEVICE)


def train_and_evaluate(task: str, config: ModelConfig, seed: int, smoke: bool) -> dict[str, Any]:
    configure_seed(seed)
    model = make_model(task, config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    started = time.perf_counter()
    if task == "text":
        train_set, test_set = text_data(seed, smoke)
        steps, step = (30 if smoke else 1500), 0
        while step < steps:
            model.train()
            for xb, yb in batches(*train_set, seed + step, True):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(xb), yb).backward()
                optimizer.step()
                step += 1
                if step >= steps:
                    break
        test_x, test_y = test_set
        domains = None
    else:
        train_set, test_set = digit_data(seed, smoke)
        epochs = 1 if smoke else 4
        for epoch in range(epochs):
            model.train()
            for xb, yb in batches(*train_set, seed + epoch, True):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(xb), yb).backward()
                optimizer.step()
        test_x, test_y, domains = test_set
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    model.eval()
    predictions = []
    with torch.no_grad():
        for xb, _ in batches(test_x, test_y, seed, False):
            predictions.append(model(xb).argmax(1).cpu())
    prediction = torch.cat(predictions)
    accuracy = (prediction == test_y).float().mean().item()
    per_domain = {}
    if domains is not None:
        for index, name in enumerate(digits.DOMAINS):
            mask = domains == index
            per_domain[name] = (prediction[mask] == test_y[mask]).float().mean().item()
    return {
        "task": task, "method": config.method, "seed": seed,
        "accuracy": accuracy, "per_domain_accuracy": per_domain,
        "train_seconds": train_seconds, "params": count_parameters(model),
    }


def bootstrap_ci(values: list[float], seed: int = 2026, samples: int = 10_000):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def exact_paired_permutation(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    observed = abs(values.mean())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    permuted = np.abs((signs * values).mean(1))
    return float(np.mean(permuted >= observed - 1e-15))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running, total = {}, 0.0, len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def summarize(runs: list[dict[str, Any]], expected_seeds: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"expected_seeds": expected_seeds, "tasks": {}}
    for task in ("text", "digits"):
        task_runs = [run for run in runs if run["task"] == task]
        summaries, by_method = {}, {}
        for method in PRIMARY_METHODS:
            method_runs = sorted(
                (run for run in task_runs if run["method"] == method),
                key=lambda run: run["seed"],
            )
            by_method[method] = {run["seed"]: run["accuracy"] for run in method_runs}
            values = [run["accuracy"] for run in method_runs]
            if values:
                summaries[method] = {
                    "n": len(values), "mean_accuracy": statistics.mean(values),
                    "std_accuracy": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "ci95_accuracy": bootstrap_ci(values),
                    "mean_train_seconds": statistics.mean(run["train_seconds"] for run in method_runs),
                }
                if task == "digits":
                    summaries[method]["per_domain_mean"] = {
                        domain: statistics.mean(run["per_domain_accuracy"][domain] for run in method_runs)
                        for domain in digits.DOMAINS
                    }
        comparisons, raw_p = {}, {}
        cca = by_method["cca_lite"]
        for method in PRIMARY_METHODS:
            if method == "cca_lite":
                continue
            common = sorted(set(cca) & set(by_method[method]))
            if not common:
                continue
            differences = [cca[seed] - by_method[method][seed] for seed in common]
            std = statistics.stdev(differences) if len(differences) > 1 else 0.0
            raw_p[method] = exact_paired_permutation(differences)
            comparisons[method] = {
                "n_pairs": len(common), "mean_gain_accuracy_points": 100 * statistics.mean(differences),
                "ci95_gain_accuracy_points": [100 * value for value in bootstrap_ci(differences)],
                "cohens_dz": statistics.mean(differences) / std if std else (
                    math.inf if statistics.mean(differences) > 0 else 0.0
                ),
                "p_exact_two_sided": raw_p[method],
            }
        adjusted = holm_adjust(raw_p)
        for method, value in adjusted.items():
            comparisons[method]["p_holm"] = value
            comparisons[method]["significant_0_05"] = value < 0.05
        result["tasks"][task] = {"methods": summaries, "cca_comparisons": comparisons}
    return result


def _latency(step, iterations=20, warmup=5) -> float:
    for _ in range(warmup):
        step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); step(); end.record(); end.synchronize()
            values.append(start.elapsed_time(end))
        else:
            start = time.perf_counter(); step(); values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values)


def profile_costs(budgets: dict[str, Any], methods=PRIMARY_METHODS) -> dict[str, Any]:
    costs: dict[str, Any] = {}
    for task in ("text", "digits"):
        costs[task] = {}
        if task == "text":
            x = torch.randint(0, text.VOCAB, (256, text.LEN), device=DEVICE)
            y = torch.randint(0, text.N_CLASSES, (256,), device=DEVICE)
        else:
            x = torch.randn(256, 1, 28, 28, device=DEVICE)
            y = torch.randint(0, 10, (256,), device=DEVICE)
        for method in methods:
            config = ModelConfig(**budgets[task]["methods"][method]["config"])
            model = make_model(task, config).to(DEVICE).train()
            def step(runner=model):
                model.zero_grad(set_to_none=True)
                F.cross_entropy(runner(x), y).backward()
            eager = _latency(step)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(); step(); torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1e6
            else:
                peak = 0.0
            compiled, compile_error = None, None
            try:
                # Isolate every compilation from stale/corrupted Inductor disk
                # entries (notably important on Windows after interrupted runs).
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
                    prefix=f"cca_inductor_{task}_{method}_"
                )
                torch._dynamo.reset()
                runner = torch.compile(model, fullgraph=True)
                compiled = _latency(lambda: step(runner), iterations=15, warmup=3)
            except Exception as error:
                compile_error = f"{type(error).__name__}: {error}"
            costs[task][method] = {
                "params": count_parameters(model),
                "analytical_flops_per_sample": analytical_flops(task, config),
                "eager_forward_backward_ms": eager,
                "compiled_forward_backward_ms": compiled,
                "compiled_speedup": eager / compiled if compiled else None,
                "eager_samples_per_second": 256 / (eager / 1000),
                "compiled_samples_per_second": 256 / (compiled / 1000) if compiled else None,
                "peak_memory_mb": peak, "compile_error": compile_error,
            }
    return costs


def environment_info(args) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        commit = None
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit, "command": sys.argv, "arguments": vars(args),
    }


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def write_csvs(output: Path, runs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "method", "seed", "accuracy", "train_seconds", "params",
                  "mnist_accuracy", "svhn_accuracy", "usps_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for run in sorted(runs, key=lambda value: (value["task"], value["method"], value["seed"])):
            row = {key: run.get(key) for key in fields}
            for domain in digits.DOMAINS:
                row[f"{domain}_accuracy"] = run["per_domain_accuracy"].get(domain)
            writer.writerow(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "method", "n", "mean_accuracy", "std_accuracy", "ci95_low", "ci95_high"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for task, task_summary in summary["tasks"].items():
            for method, values in task_summary["methods"].items():
                writer.writerow({"task": task, "method": method, "n": values["n"],
                                 "mean_accuracy": values["mean_accuracy"],
                                 "std_accuracy": values["std_accuracy"],
                                 "ci95_low": values["ci95_accuracy"][0],
                                 "ci95_high": values["ci95_accuracy"][1]})


def write_report(output: Path, summary: dict[str, Any], budgets: dict[str, Any], costs: Any) -> None:
    lines = ["# Confirmatory activation comparison", "",
             "This evidence file is generated from paired runs; it is not a paper source.", ""]
    for task in ("text", "digits"):
        lines += [f"## {task.title()}", "", "| Method | n | Accuracy (mean ± sd) | 95% CI |",
                  "|---|---:|---:|---:|"]
        for method in PRIMARY_METHODS:
            value = summary["tasks"][task]["methods"].get(method)
            if not value:
                continue
            lo, hi = value["ci95_accuracy"]
            lines.append(f"| {DISPLAY_NAMES[method]} | {value['n']} | "
                         f"{100*value['mean_accuracy']:.2f} ± {100*value['std_accuracy']:.2f} | "
                         f"[{100*lo:.2f}, {100*hi:.2f}] |")
        if task == "digits":
            lines += ["", "Per-domain mean accuracy:", "",
                      "| Method | MNIST | SVHN | USPS |", "|---|---:|---:|---:|"]
            for method in PRIMARY_METHODS:
                value = summary["tasks"][task]["methods"].get(method)
                if not value:
                    continue
                domain = value["per_domain_mean"]
                lines.append(f"| {DISPLAY_NAMES[method]} | {100*domain['mnist']:.2f} | "
                             f"{100*domain['svhn']:.2f} | {100*domain['usps']:.2f} |")
        lines += ["", "CCA-Lite paired comparisons (two-sided exact permutation; Holm corrected):", "",
                  "| Competitor | Gain (pp) | 95% CI | Cohen dz | p-Holm |", "|---|---:|---:|---:|---:|"]
        for method, value in summary["tasks"][task]["cca_comparisons"].items():
            lo, hi = value["ci95_gain_accuracy_points"]
            lines.append(f"| {DISPLAY_NAMES[method]} | {value['mean_gain_accuracy_points']:+.2f} | "
                         f"[{lo:+.2f}, {hi:+.2f}] | {value['cohens_dz']:.3f} | {value['p_holm']:.5f} |")
        lines += ["", "Budget utilization:", "", "| Method | Parameters vs target | FLOPs vs target |",
                  "|---|---:|---:|"]
        for method in PRIMARY_METHODS:
            value = budgets[task]["methods"][method]
            lines.append(f"| {DISPLAY_NAMES[method]} | {value['params_vs_target_pct']:+.2f}% | "
                         f"{value['flops_vs_target_pct']:+.2f}% |")
        if costs:
            lines += ["", "Measured forward+backward cost:", "",
                      "| Method | Eager ms | Compiled ms | Peak MB |", "|---|---:|---:|---:|"]
            for method in PRIMARY_METHODS:
                value = costs[task][method]
                compiled = "n/a" if value["compiled_forward_backward_ms"] is None else f"{value['compiled_forward_backward_ms']:.3f}"
                lines.append(f"| {DISPLAY_NAMES[method]} | {value['eager_forward_backward_ms']:.3f} | "
                             f"{compiled} | {value['peak_memory_mb']:.1f} |")
        lines.append("")
    (output / "EVIDENCE.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(output: Path) -> None:
    names = ("checkpoint.json", "summary.json", "runs.csv", "summary.csv", "EVIDENCE.md",
             "budgets.json", "costs.json", "environment.json")
    lines = []
    for name in names:
        path = output / name
        if path.exists():
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    (output / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-cost", action="store_true")
    parser.add_argument("--deadline-minutes", type=float, default=110.0)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--tasks", default="text,digits")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.smoke:
        seeds = seeds[:1]
    tasks = [value.strip() for value in args.tasks.split(",")]
    invalid = set(tasks) - {"text", "digits"}
    if invalid:
        raise SystemExit(f"Unknown tasks: {sorted(invalid)}")
    output = args.output or ROOT / "results" / (
        "extended_comparison_smoke" if args.smoke else "extended_comparison"
    )
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.deadline_minutes * 60
    torch.set_float32_matmul_precision("high")
    budgets = resolve_budgets()
    atomic_json(output / "budgets.json", budgets)
    atomic_json(output / "environment.json", environment_info(args))
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {
        "schema_version": 1, "seeds": seeds, "runs": []
    }
    runs = checkpoint["runs"]
    completed = {(run["task"], run["method"], run["seed"]) for run in runs}
    selected_keys = {(task, method, seed) for task in tasks
                     for method in PRIMARY_METHODS for seed in seeds}
    done_selected = len(selected_keys & completed)
    total = len(selected_keys)
    for task in tasks:
        for seed in seeds:
            for method in PRIMARY_METHODS:
                key = (task, method, seed)
                if key in completed:
                    print(f"resume: skipping {task}/{method}/seed={seed}", flush=True)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("110-minute study deadline reached; checkpoint is safe to resume")
                config = ModelConfig(**budgets[task]["methods"][method]["config"])
                print(f"[{done_selected+1}/{total}] {task}/{method}/seed={seed}", flush=True)
                run = train_and_evaluate(task, config, seed, args.smoke)
                runs.append(run); completed.add(key)
                done_selected += 1
                checkpoint["elapsed_seconds"] = time.monotonic() - started
                atomic_json(checkpoint_path, checkpoint)
                print(f"  accuracy={100*run['accuracy']:.2f}% train={run['train_seconds']:.1f}s", flush=True)
    summary = summarize(runs, seeds)
    costs = {} if args.skip_cost else profile_costs(budgets)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "costs.json", costs)
    write_csvs(output, runs, summary)
    write_report(output, summary, budgets, costs)
    write_manifest(output)
    print(f"complete in {(time.monotonic()-started)/60:.1f} min -> {output}")


if __name__ == "__main__":
    main()

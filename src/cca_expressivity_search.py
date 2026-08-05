"""Validation-only search for a more expressive, cost-matched CCA preset.

No confirmatory test split or seed is used for selection. Text validation uses
fresh synthetic samples (seed + 49); digit validation is held out from each
official training split. The official digit test sets remain untouched.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

import digits_multidomain as digits
import text_cls as text
from extended_comparison import (
    DEVICE, DigitModel, ModelConfig, TextModel, analytical_flops, atomic_json,
    batches, configure_seed, count_parameters, load_digit_domains,
)


ROOT = Path(__file__).resolve().parents[1]
DEV_SEEDS = (2001, 2002, 2003)
VARIANTS = {
    "lite_g16_h16": dict(method="cca_lite", groups=16, g_hidden=16, lowrank=0),
    "g32_lr8": dict(method="cca_lite", groups=32, g_hidden=16, lowrank=8),
    "g32_lr4": dict(method="cca_lite", groups=32, g_hidden=16, lowrank=4),
    "g64_lr4": dict(method="cca_lite", groups=64, g_hidden=16, lowrank=4),
    "g8_h32": dict(method="cca_lite", groups=8, g_hidden=32, lowrank=0),
    "g16_h24_lr8": dict(method="cca_lite", groups=16, g_hidden=24, lowrank=8),
    "g32_h24_lr4": dict(method="cca_lite", groups=32, g_hidden=24, lowrank=4),
    "hinge_g8_h32": dict(method="cca_hinge", groups=8, g_hidden=32, lowrank=0),
    "hinge_g8_h24": dict(method="cca_hinge", groups=8, g_hidden=24, lowrank=0),
    "hinge_g16_h16": dict(method="cca_hinge", groups=16, g_hidden=16, lowrank=0),
}


def config_for(task: str, variant: str) -> ModelConfig:
    kwargs = dict(VARIANTS[variant])
    method = kwargs.pop("method")
    if task == "text":
        return ModelConfig(method, hidden=256, **kwargs)
    return ModelConfig(method, channels1=16, channels2=32, **kwargs)


def validate_budgets() -> dict:
    result = {}
    for task in ("text", "digits"):
        baseline = config_for(task, "lite_g16_h16")
        target_params = count_parameters(TextModel(baseline) if task == "text" else DigitModel(baseline))
        target_flops = analytical_flops(task, baseline)
        result[task] = {}
        for name in VARIANTS:
            config = config_for(task, name)
            model = TextModel(config) if task == "text" else DigitModel(config)
            params, flops = count_parameters(model), analytical_flops(task, config)
            feasible = params <= 1.05 * target_params and flops <= 1.10 * target_flops
            result[task][name] = {
                "config": asdict(config), "params": params, "flops_per_sample": flops,
                "params_vs_lite_pct": 100 * (params / target_params - 1),
                "flops_vs_lite_pct": 100 * (flops / target_flops - 1),
                "feasible": feasible,
            }
    return result


def make_text_dev(seed: int, smoke: bool):
    sources = text.make_sources(text.N_CLASSES, seed)
    ciphers = text.make_ciphers(8, seed + 1)
    n_train, n_val = (4096, 1024) if smoke else (text.N_TRAIN, text.N_TEST)
    train_set = text.gen_dataset(n_train, sources, ciphers, seed)
    validation_set = text.gen_dataset(n_val, sources, ciphers, seed + 49)
    return train_set, validation_set


def make_digit_dev(seed: int, smoke: bool):
    cached = load_digit_domains()
    n_train, n_val = (512, 128) if smoke else (5000, 1000)
    train_x, train_y, val_x, val_y = [], [], [], []
    for domain_index, name in enumerate(digits.DOMAINS):
        domain_x, domain_y, _, _ = cached[name]
        generator = torch.Generator().manual_seed(seed + domain_index)
        order = torch.randperm(len(domain_x), generator=generator)
        train_index = order[:n_train]
        validation_index = order[n_train:n_train + n_val]
        train_x.append(domain_x[train_index]); train_y.append(domain_y[train_index])
        val_x.append(domain_x[validation_index]); val_y.append(domain_y[validation_index])
    return (torch.cat(train_x), torch.cat(train_y)), (torch.cat(val_x), torch.cat(val_y))


@torch.no_grad()
def accuracy(model, dataset, seed: int) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in batches(*dataset, seed, False):
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += len(yb)
    return correct / total


def train_validate(task: str, config: ModelConfig, seed: int, smoke: bool, dataset) -> dict:
    configure_seed(seed)
    model = (TextModel(config) if task == "text" else DigitModel(config)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    train_set, validation_set = dataset
    started = time.perf_counter()
    if task == "text":
        steps, step = (40 if smoke else 1000), 0
        while step < steps:
            model.train()
            for xb, yb in batches(*train_set, seed + step, True):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(xb), yb).backward(); optimizer.step()
                step += 1
                if step >= steps:
                    break
    else:
        for epoch in range(1 if smoke else 3):
            model.train()
            for xb, yb in batches(*train_set, seed + epoch, True):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(xb), yb).backward(); optimizer.step()
    return {"validation_accuracy": accuracy(model, validation_set, seed),
            "train_seconds": time.perf_counter() - started}


def choose_variant(runs: list[dict]) -> dict:
    aggregate = {}
    available = set(VARIANTS)
    for task in ("text", "digits"):
        aggregate[task] = {}
        for variant in VARIANTS:
            values = [run["validation_accuracy"] for run in runs
                      if run["task"] == task and run["variant"] == variant]
            if values:
                aggregate[task][variant] = statistics.mean(values)
            else:
                available.discard(variant)
    baseline = "lite_g16_h16"
    candidates = []
    for variant in sorted(available):
        gains = {task: aggregate[task][variant] - aggregate[task][baseline]
                 for task in ("text", "digits")}
        eligible = all(gain >= -0.002 for gain in gains.values())
        score = statistics.mean(gains.values()) if eligible else float("-inf")
        candidates.append((score, min(gains.values()), variant, gains))
    score, _, selected, gains = max(candidates)
    return {"selected": selected, "selection_rule": (
                "maximize mean validation accuracy gain across tasks; reject any candidate "
                "more than 0.2 percentage points below CCA-Lite on either task"),
            "score": score, "paired_task_gains": gains, "means": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "results" / (
        "cca_expressivity_search_smoke" if args.smoke else "cca_expressivity_search"
    )
    output.mkdir(parents=True, exist_ok=True)
    budgets = validate_budgets()
    feasible = [name for name in VARIANTS if all(budgets[task][name]["feasible"]
                                                  for task in ("text", "digits"))]
    if "lite_g16_h16" not in feasible:
        raise RuntimeError("Baseline CCA-Lite is unexpectedly infeasible")
    atomic_json(output / "budgets.json", budgets)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"runs": []}
    runs = checkpoint["runs"]
    seeds = DEV_SEEDS[:1] if args.smoke else DEV_SEEDS
    completed = {(run["task"], run["variant"], run["seed"]) for run in runs}
    total = 2 * len(feasible) * len(seeds)
    done = len(completed)
    for task in ("text", "digits"):
        for seed in seeds:
            dataset = make_text_dev(seed, args.smoke) if task == "text" else make_digit_dev(seed, args.smoke)
            for variant in feasible:
                key = (task, variant, seed)
                if key in completed:
                    continue
                print(f"[{done+1}/{total}] {task}/{variant}/seed={seed}", flush=True)
                result = train_validate(task, config_for(task, variant), seed, args.smoke, dataset)
                run = {"task": task, "variant": variant, "seed": seed, **result}
                runs.append(run); completed.add(key); done += 1
                atomic_json(checkpoint_path, checkpoint)
                print(f"  validation={100*result['validation_accuracy']:.2f}%", flush=True)
    selection = choose_variant(runs)
    atomic_json(output / "selection.json", selection)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

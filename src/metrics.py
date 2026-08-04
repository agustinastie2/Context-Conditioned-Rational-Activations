"""
metrics.py — shared cost / viability instrumentation for the CCA study.
========================================================================

The research question for this round is no longer "does CCA help?" but
"**is it worth it?**" — i.e. is the quality gain larger than the compute it
costs? This module is the single source of truth for the *cost* side of that
trade-off, so every experiment reports the same numbers in the same units.

It measures, for any model:

  * `params`            total parameter count
  * `params_overhead`   % extra parameters vs a baseline (SiLU) model
  * `gflops`            forward FLOPs (torch FlopCounterMode; counts the matmul/
                        conv/attention work — including CCA's generator `G` — but
                        not cheap pointwise ops, so it is a lower bound on CCA's
                        true overhead and an upper bound on its *relative* cost)
  * `latency_ms`        median wall-time of one forward (and fwd+bwd) step
  * `throughput`        items/s (tokens, pixels or samples — caller's unit)
  * `peak_mem_mb`       peak CUDA memory of a single step

and the *viability* of CCA against its baseline:

  gain%   = relative quality improvement (PPL/bits-dim/error reduction)
  cost%   = relative throughput drop  = 100 * (1 - tput_cca / tput_silu)
  R       = gain% / cost%             (the "intelligence per cost" ratio)
  rentable = gain% > cost%            (the prudent-use rule from the plan)

A config with R > 1 buys more quality than the speed it gives up.
"""

import statistics
import time
from contextlib import contextmanager

import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def overhead_pct(n, base):
    return 100.0 * (n - base) / base if base else 0.0


def measure_gflops(forward_fn):
    """Forward GFLOPs for a single call of `forward_fn` (a zero-arg closure).
    Returns None if the FLOP counter is unavailable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    fc = FlopCounterMode(display=False)
    try:
        with fc, torch.no_grad():
            forward_fn()
        return fc.get_total_flops() / 1e9
    except Exception:
        return None


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def measure_latency(step_fn, device, warmup=5, iters=20):
    """Median ms of one `step_fn()` call (forward, or forward+backward).
    Uses CUDA events on GPU and perf_counter on CPU. `step_fn` takes no args."""
    for _ in range(warmup):
        step_fn()
    _sync(device)
    times = []
    use_cuda = torch.device(device).type == "cuda"
    for _ in range(iters):
        if use_cuda:
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            step_fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            step_fn()
            times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def peak_mem_mb(fn, device):
    """Peak CUDA memory (MB) used while running `fn()` once. 0 on CPU."""
    if torch.device(device).type != "cuda":
        fn()
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


@contextmanager
def _eval_mode(model):
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


def model_cost(model, device, *, forward_fn, train_step_fn=None,
               throughput=None, baseline_params=None, n_items=None):
    """Assemble the standard cost block for one model.

    forward_fn     : zero-arg closure running ONE inference forward (no grad).
    train_step_fn  : optional zero-arg closure running ONE forward+backward.
    throughput     : optional items/s already measured by the caller.
    baseline_params: optional SiLU param count for the overhead %.
    n_items        : optional items processed per forward, to derive throughput
                     from latency when `throughput` is not supplied.
    """
    n_par = count_params(model)
    cost = {"params": n_par}
    if baseline_params is not None:
        cost["params_overhead_pct"] = overhead_pct(n_par, baseline_params)
    with _eval_mode(model), torch.no_grad():
        cost["gflops"] = measure_gflops(forward_fn)
        cost["fwd_latency_ms"] = measure_latency(forward_fn, device)
        cost["peak_mem_mb"] = peak_mem_mb(forward_fn, device)
    if train_step_fn is not None:
        cost["step_latency_ms"] = measure_latency(train_step_fn, device)
    if throughput is not None:
        cost["throughput"] = throughput
    elif n_items is not None and cost["fwd_latency_ms"] > 0:
        cost["throughput"] = n_items / (cost["fwd_latency_ms"] / 1e3)
    return cost


def viability(gain_pct, tput_silu, tput_cca):
    """Relate the quality gain (%) to the throughput it costs (%)."""
    cost = 100.0 * (1.0 - tput_cca / tput_silu) if tput_silu else 0.0
    if cost <= 0:                      # CCA same speed or faster: pure win
        return {"throughput_cost_pct": cost,
                "viability_ratio": float("inf"), "rentable": True}
    return {"throughput_cost_pct": cost,
            "viability_ratio": gain_pct / cost,
            "rentable": gain_pct > cost}

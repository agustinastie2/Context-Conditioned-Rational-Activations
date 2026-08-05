"""Reproducible CCA-lite forward/backward microbenchmark.

Run from the repository root with ``python benchmarks/benchmark_cca.py``.
Pass ``--compile`` to measure the fused Inductor/Triton training path.
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from cca_core import CCALite1D, CCALite2D


def benchmark(name, module, x, context, compile_model, warmup, iterations):
    module = module.cuda().train()
    x = x.cuda().requires_grad_(True)
    context = None if context is None else context.cuda()
    with torch.no_grad():
        module.G[-1].weight.normal_(0, 0.02)
        module.G[-1].bias.normal_(0, 0.02)
    runner = torch.compile(module, fullgraph=True) if compile_model else module

    def step():
        module.zero_grad(set_to_none=True)
        x.grad = None
        output = runner(x, context) if context is not None else runner(x)
        output.square().mean().backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        step()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"{name:12s} {statistics.median(timings):7.3f} ms  {peak:8.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=60)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark currently requires CUDA.")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    print(f"PyTorch {torch.__version__} | {torch.cuda.get_device_name()} | "
          f"{'compiled' if args.compile else 'eager'} training")
    benchmark("1d-batch", CCALite1D(128, 256), torch.randn(128, 256),
              torch.randn(128, 128), args.compile, args.warmup, args.iterations)
    benchmark("1d-token", CCALite1D(128, 512), torch.randn(16, 128, 512),
              torch.randn(16, 128, 128), args.compile, args.warmup, args.iterations)
    benchmark("2d", CCALite2D(64), torch.randn(32, 64, 28, 28), None,
              args.compile, args.warmup, args.iterations)


if __name__ == "__main__":
    main()

"""Bounded-concurrency launcher for the paper comparison matrix.

`paper_full_comparison.py` already isolates every (task, seed) cell in its own
worker process and checkpoints each method atomically; it just launches those
workers strictly one at a time.  This module reuses that worker CLI unchanged
and keeps up to `--jobs` of them alive at once, so the numbers are bit-identical
to a serial run while the wall clock shrinks on launch-bound workloads.

Two differences from the parent loop matter on rented hardware:

  * a worker that dies is recorded and reported, not allowed to abort the whole
    matrix -- every other cell still completes and the failure can be retried;
  * each worker gets ``OMP_NUM_THREADS=1``.  PyTorch otherwise defaults to one
    OpenMP thread per core *per process*, so N concurrent workers oversubscribe
    the host and can run slower than serial.  That is the single most common
    reason multi-process training "does not scale".

`--jobs 1` is exactly the serial path, through the same code.

Run:  python src/run_parallel.py --tasks text_char --seeds 4001,4002 \
          --output results/paper_textchar_l4 --jobs 4 --env ITERS=3000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from paper_full_comparison import TASKS, methods_for, atomic_json

ROOT = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "paper_full_comparison.py"


def completed_cells(directory: Path) -> set[tuple[str, str, int]]:
    """Every (task, method, seed) already on disk, from the checkpoint and the
    per-worker files.  Mirrors the parent's `ingest_workers` view so a rerun
    resumes instead of retraining."""
    done: set[tuple[str, str, int]] = set()
    sources = list((directory / "workers").glob("*.json"))
    checkpoint = directory / "checkpoint.json"
    if checkpoint.exists():
        sources.append(checkpoint)
    for path in sources:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for run in payload.get("runs", []):
            done.add((run["task"], run["method"], run["seed"]))
    return done


def plan_jobs(tasks, seeds, methods, directory):
    """One job per (task, seed) that still has missing methods."""
    done = completed_cells(directory)
    jobs = []
    for task in tasks:
        eligible = [m for m in methods_for(task) if m in methods]
        for seed in seeds:
            missing = [m for m in eligible if (task, m, seed) not in done]
            if missing:
                jobs.append((task, seed, missing))
    return jobs


def worker_env(base_env: dict[str, str], task: str, seed: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update(base_env)
    # Keep every worker single-threaded on the host: N processes x one OpenMP
    # thread per core would thrash an 8-vCPU box long before the GPU saturates.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if env.get("PAPER_COMPILE") == "1":
        # Private Inductor cache per worker: concurrent compiles otherwise race
        # on the same on-disk entries.
        env["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / ".inductor_cache" / f"{task}_s{seed}")
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--methods", default="silu,rational,dyrelu,meta_acon,cca_lite,cca_hinge,b0h")
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--env", action="append", default=[],
                        help="KEY=VALUE passed to every worker (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-summarize", action="store_true")
    args = parser.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]
    for task in tasks:
        if task not in TASKS:
            raise SystemExit(f"Unknown task {task}")
    seeds = [int(s) for s in args.seeds.split(",") if s]
    methods = {m for m in args.methods.split(",") if m}
    base_env = dict(pair.split("=", 1) for pair in args.env)

    directory = ROOT / args.output
    (directory / "workers").mkdir(parents=True, exist_ok=True)
    (directory / "logs").mkdir(exist_ok=True)

    jobs = plan_jobs(tasks, seeds, methods, directory)
    total_cells = sum(len(missing) for _, _, missing in jobs)
    print(f"PLAN {len(jobs)} workers, {total_cells} cells, jobs={args.jobs}, env={base_env}",
          flush=True)
    if args.dry_run:
        for task, seed, missing in jobs:
            print(f"  {task} s{seed}: {','.join(missing)}")
        return

    started = time.monotonic()
    running: list[tuple] = []
    pending = list(jobs)
    failures: list[tuple[str, int, int]] = []
    finished = 0

    def launch(job):
        task, seed, missing = job
        log_path = directory / "logs" / f"{task}_s{seed}.log"
        handle = log_path.open("w", encoding="utf-8")
        command = [sys.executable, str(HARNESS), "--worker", "--task", task,
                   "--seed", str(seed), "--methods", ",".join(missing),
                   "--worker-output", str(directory / "workers" / f"{task}_s{seed}.json")]
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT,
                                   env=worker_env(base_env, task, seed))
        print(f"LAUNCH {task} s{seed} [{len(missing)} methods] pid={process.pid}", flush=True)
        return (process, handle, task, seed, len(missing), time.monotonic())

    while pending or running:
        while pending and len(running) < args.jobs:
            running.append(launch(pending.pop(0)))
        time.sleep(2.0)
        for entry in list(running):
            process, handle, task, seed, n_cells, launched = entry
            if process.poll() is None:
                continue
            running.remove(entry)
            handle.close()
            finished += 1
            elapsed = time.monotonic() - launched
            if process.returncode:
                # Do not abort the matrix: record it, keep every other cell going.
                failures.append((task, seed, process.returncode))
                print(f"FAILED {task} s{seed} rc={process.returncode} after {elapsed:.0f}s "
                      f"(see logs/{task}_s{seed}.log)", flush=True)
            else:
                print(f"DONE {task} s{seed} in {elapsed:.0f}s "
                      f"[{finished}/{len(jobs)} workers, "
                      f"{(time.monotonic()-started)/60:.1f} min elapsed]", flush=True)

    wall_minutes = (time.monotonic() - started) / 60
    contended = args.jobs > 1
    atomic_json(directory / "parallel_manifest.json", {
        "jobs": args.jobs, "env": base_env, "tasks": tasks, "seeds": seeds,
        "workers_launched": len(jobs), "cells_requested": total_cells,
        "failures": [{"task": t, "seed": s, "returncode": r} for t, s, r in failures],
        "wall_minutes": wall_minutes,
        "cells_per_minute": total_cells / wall_minutes if wall_minutes else None,
        "host": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timings_contended": contended,
        "timings_note": (
            "train_seconds and eval_throughput in the run records were measured with "
            f"{args.jobs} concurrent workers and are inflated by contention; they are NOT "
            "cost evidence. Use profile_costs() output instead."
        ) if contended else "Workers ran serially; per-run timings are uncontended.",
    })
    print(f"WALL {wall_minutes:.1f} min, {total_cells/wall_minutes if wall_minutes else 0:.2f} "
          f"cells/min, {len(failures)} failed workers", flush=True)

    if failures:
        print("RETRY needed for: " + ", ".join(f"{t}/s{s}" for t, s, _ in failures), flush=True)
    if args.no_summarize:
        return
    # Every cell is on disk now; the parent only ingests and writes artifacts.
    subprocess.run([sys.executable, str(HARNESS), "--tasks", ",".join(tasks),
                    "--seeds", ",".join(map(str, seeds)),
                    "--methods", ",".join(sorted(methods)),
                    "--output", args.output], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()

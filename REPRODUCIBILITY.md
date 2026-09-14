# Reproducing the confirmatory activation comparison

The confirmatory study compares CCA-Lite with Dynamic ReLU-B, Meta-ACON, a
static rational activation, and SiLU under fixed total parameter and analytical
compute budgets. It does not modify or regenerate the paper.

## Environment

The recorded run uses Python 3.14.4, PyTorch 2.11.0, CUDA 13.0, and an NVIDIA
RTX 4070 Laptop GPU. Install the tested top-level packages with:

```powershell
python -m pip install -r requirements-repro.txt
```

CUDA wheels may require selecting the wheel index appropriate for the local
driver from the official PyTorch installer. `requirements-repro.txt` records
package versions; `results/extended_comparison/environment.json` records the
actual runtime and command.

## Verification and execution

```powershell
python -m unittest discover -s tests -v
python src/extended_comparison.py --smoke --skip-cost
python src/extended_comparison.py
python src/cca_expressivity_search.py
python src/cca_fresh_confirmation.py
```

The full command uses paired seeds 1337–1346 and stops before beginning a new
run once its 110-minute deadline is reached. Every completed method/seed is
atomically written to `results/extended_comparison/checkpoint.json`; rerunning
the same command resumes rather than retraining completed entries.

`cca_expressivity_search.py` selects CCA-Hinge using development seeds
2001–2003 and held-out validation data only. `cca_fresh_confirmation.py` then
compares the locked preset on untouched seeds 3001–3010. To run the selected
preset throughout the original suite, set `CCA_VARIANT=hinge` before invoking
`src/run_all.py`; leaving it unset preserves the original full CCA.

For a statistics-only rebuild, rerun the command after all checkpoints exist;
completed training runs are skipped. Dataset files remain under ignored
`data/` and are never redistributed.

## Full paper comparison (character-level only)

`paper_full_comparison.py` applies the common activation interface to the
experiments defined by the manuscript. It runs SiLU, Static Rational, Dynamic
ReLU-B, Meta-ACON, CCA-Lite, CCA-Hinge, and B0H where the paper defines that
ablation. Each task/seed/method result is written atomically, so interrupted
commands can be rerun without repeating completed cells.

The completed confirmatory matrix used ten paired seeds for the main tasks and
three paired seeds for the slower character language model:

```powershell
python src/paper_full_comparison.py --tasks image_cls,image_cls_single,text_cls,text_cls_single,digits_multidomain,algo_modular,audio_digits,text_sentiment --seeds 4001,4002,4003,4004,4005,4006,4007,4008,4009,4010 --output results/paper_full_comparison
python src/paper_full_comparison.py --tasks text_char --seeds 4001,4002,4003 --output results/paper_full_comparison
```

The BPE language-model experiment is deliberately omitted from this fast
evidence pass. No BPE result is mixed into the character-level statistics.

## Short image-generation protocol

The short protocol keeps the paper's autoregressive image task and evaluates
MNIST, Fashion-MNIST, and CIFAR-10 for 500 optimization steps. It uses BF16
autocast and stages each dataset on the GPU to remove host-transfer overhead:

```powershell
$env:STEPS='500'
$env:PAPER_AMP='1'
$env:PAPER_STAGE_DATA='1'
python src/paper_full_comparison.py --tasks image_gen_mnist,image_gen_fashion,image_gen_cifar --seeds 4101,4102,4103 --output results/paper_imagegen_short
```

The CIFAR-10 loader first uses the official archive. If that source is
unavailable, it downloads checksum-pinned Parquet files from the Hugging Face
dataset mirror. Both paths preserve the standard train/test examples and fail
closed when a digest does not match.

## Ten-seed generation pass on an NVIDIA L4 (Linux)

The character language model and the three image-generation sources were
originally run for three seeds only, for time. They were re-run for the full ten
paired seeds on a rented NVIDIA L4 (23 GB), Linux, Python 3.12, PyTorch
2.8.0+cu128, Triton 3.4.0. All ten seeds of a block come from this one machine,
so no block mixes hardware.

`src/run_parallel.py` keeps up to `--jobs` of the existing
`paper_full_comparison.py --worker` processes alive at once. Each worker is an
independent, seeded process writing its own atomic checkpoint, so concurrency
does not change any result; it only changes wall-clock. Unlike the serial parent
loop, a worker that dies is recorded and reported instead of aborting the matrix.

```bash
python src/run_parallel.py --tasks text_char \
  --seeds 4001,4002,4003,4004,4005,4006,4007,4008,4009,4010 \
  --output results/paper_textchar_l4 --jobs 6 --env ITERS=3000

python src/run_parallel.py --tasks image_gen_mnist,image_gen_fashion,image_gen_cifar \
  --seeds 4101,4102,4103,4104,4105,4106,4107,4108,4109,4110 \
  --output results/paper_imagegen_short_l4 --jobs 1 \
  --env STEPS=500 --env PAPER_AMP=1 --env PAPER_STAGE_DATA=1
```

The image-generation protocol (500 steps, BF16, staged data) is unchanged from
the three-seed pass, so the two are directly comparable.

### Choosing the execution configuration

`results/perf_calibration_l4.json` records a measured comparison on a fixed short
workload rather than an assumed one. Three findings:

- **Concurrency only pays off with single-threaded workers.** PyTorch defaults to
  one OpenMP thread per core *per process*; six workers times eight threads
  oversubscribes an 8-vCPU host and makes concurrent runs slower than serial.
  `run_parallel.py` sets `OMP_NUM_THREADS=MKL_NUM_THREADS=1` per worker.
- **And only for the task that cannot fill the GPU.** With that fixed, the two
  blocks behave in opposite ways, so they are run with different concurrency:

  | Block | Per-step work | jobs | Result |
  |---|---|---:|---|
  | `text_char` | batch 64, 16-token window, d=128 | 6 | 70 cells in 15.7 min vs 56.4 min serial, **3.6x** |
  | image generation | batch 128, 256-pixel sequence, BF16 | 1 | **flat**: 4.88 / 4.77 / 4.83 cells/min at jobs=1 / 2 / 4 |

  The character LM is launch-bound and leaves the L4 mostly idle, so extra
  processes fill it. Image generation already saturates the same GPU with one
  process, so concurrency buys nothing and only adds contention. An initial
  calibration that mixed both tasks hid this, because the character LM's gain
  dominated the aggregate -- per-task calibration is the one to trust.
- **CUDA MPS does not help.** Measured 17.74 vs 18.52 cells/min, i.e. 4% *slower*.
  At this model size the bottleneck is host-side kernel launch, not GPU context
  switching, so MPS only adds indirection.
- **`torch.compile` is not a training-time optimisation here.** It raised
  `BackendCompilerFailed` on Dynamic ReLU-B (Inductor pattern matcher, torch 2.8),
  and where it did compile, 20-27 s of warm-up bought 3-4 s of training.
  Compilation remains the right measurement for *steady-state* cost, which is what
  `costs.json` reports; it does not speed up these short runs.

### Cost re-measurement

`src/measure_costs.py` re-runs only `profile_costs()` against the already-resolved
`results/cca_hinge_confirmation/budgets.json`, writing `results/cca_cost_l4/`.
It retrains nothing: accuracy over seeds 3001--3010 is unchanged and remains the
published result, while cost is a property of the machine and is re-measured here.
This fills the one cell the Windows run could not produce -- compiled
`digits/cca_hinge`, which failed on the Triton launcher's generated C wrapper.

```bash
python src/measure_costs.py --output results/cca_cost_l4
```

### A caveat on timings inside the run records

When `--jobs > 1`, the `train_seconds` and `eval_throughput` fields of each run
record are inflated by contention between workers. They are **not** cost evidence
and no table in the paper uses them; `parallel_manifest.json` in each output
directory flags this explicitly. All reported cost numbers come from
`profile_costs()`, which runs alone.

## Dataset protocol

- Synthetic text: 40,000 train and 8,000 test sequences, length 48, vocabulary
  27, four Markov sources, and eight substitution-cipher regimes. Test data use
  seed `train_seed + 99`.
- Real digits: official MNIST, SVHN, and USPS train/test splits. Each paired run
  samples 6,000 train and 2,000 test examples per domain, converts images to
  grayscale, and resizes them to 28×28.

Adaptive text activations receive the mean token embedding. Adaptive digit
activations receive global-average-pooled preactivation features at each layer.
The dataset/domain identity is retained only for reporting per-domain accuracy
and is never provided to any model.

Downloads use torchvision's normal certificate verification and the Mozilla CA
bundle distributed by `certifi`. If the official USPS host's legacy certificate
is rejected, the loader uses the `flwrlabs/usps` HTTPS mirror and checks pinned
SHA-256 digests before parsing it. The code never disables TLS verification.

## Output integrity

**Line endings.** The manifests under `results/extended_comparison`,
`results/paper_full_comparison`, `results/paper_imagegen_short` and
`results/cca_hinge_confirmation` were written on Windows, over CRLF files. On a
checkout that normalises to LF every one of them reports a mismatch even though
the artifacts are intact: re-hashing the same bytes with `\n` replaced by `\r\n`
reproduces each recorded digest exactly. Verify those four directories either on
Windows or after converting to CRLF. The directories produced on Linux
(`paper_textchar_l4`, `paper_imagegen_short_l4`, `cca_cost_l4`) verify directly
with `sha256sum -c`.


The output directory contains raw runs, summaries, costs, budget configurations,
CSV tables, an evidence report, and `manifest.sha256`. Verify it in PowerShell:

```powershell
Get-Content results/extended_comparison/manifest.sha256
Get-FileHash results/extended_comparison/*.json -Algorithm SHA256
```

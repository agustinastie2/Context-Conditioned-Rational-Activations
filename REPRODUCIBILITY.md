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

The output directory contains raw runs, summaries, costs, budget configurations,
CSV tables, an evidence report, and `manifest.sha256`. Verify it in PowerShell:

```powershell
Get-Content results/extended_comparison/manifest.sha256
Get-FileHash results/extended_comparison/*.json -Algorithm SHA256
```

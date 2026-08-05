# Complete paper experiment matrix

Paired seeds: 4999.

## text_char

Metric: `ppl` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## text_bpe_2000

Metric: `ppl` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## text_bpe_4000

Metric: `ppl` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## text_bpe_8000

Metric: `ppl` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## image_gen_mnist

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 1 | 1.03250 ± 0.00000 | [1.03250, 1.03250] | -5.98% |
| Static Rational | 1 | 1.03208 ± 0.00000 | [1.03208, 1.03208] | -5.70% |
| Dynamic ReLU-B | 1 | 0.91418 ± 0.00000 | [0.91418, 0.91418] | -3.24% |
| Meta-ACON | 1 | 0.91034 ± 0.00000 | [0.91034, 0.91034] | -4.99% |
| CCA-Lite | 1 | 0.89541 ± 0.00000 | [0.89541, 0.89541] | -1.21% |
| CCA-Hinge | 1 | 0.89380 ± 0.00000 | [0.89380, 0.89380] | +0.00% |
| SiLU+Context (B0H) | 1 | 0.91713 ± 0.00000 | [0.91713, 0.91713] | -5.34% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.13870 | 1.00000 |
| Static Rational | +0.13828 | 1.00000 |
| Dynamic ReLU-B | +0.02038 | 1.00000 |
| Meta-ACON | +0.01653 | 1.00000 |
| CCA-Lite | +0.00160 | 1.00000 |
| SiLU+Context (B0H) | +0.02333 | 1.00000 |

## image_gen_fashion

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## image_gen_cifar

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## image_cls

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## image_cls_single

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## text_cls

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## text_cls_single

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## digits_multidomain

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## algo_modular

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## audio_digits

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

## text_sentiment

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|

# Complete paper experiment matrix

Paired seeds: 4998.

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
| SiLU | 1 | 1.01882 ± 0.00000 | [1.01882, 1.01882] | -5.98% |
| Static Rational | 1 | 1.01895 ± 0.00000 | [1.01895, 1.01895] | -5.70% |
| Dynamic ReLU-B | 1 | 0.88488 ± 0.00000 | [0.88488, 0.88488] | -3.24% |
| Meta-ACON | 1 | 0.88260 ± 0.00000 | [0.88260, 0.88260] | -4.99% |
| CCA-Lite | 1 | 0.86591 ± 0.00000 | [0.86591, 0.86591] | -1.21% |
| CCA-Hinge | 1 | 0.86505 ± 0.00000 | [0.86505, 0.86505] | +0.00% |
| SiLU+Context (B0H) | 1 | 0.88477 ± 0.00000 | [0.88477, 0.88477] | -5.34% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.15377 | 1.00000 |
| Static Rational | +0.15390 | 1.00000 |
| Dynamic ReLU-B | +0.01983 | 1.00000 |
| Meta-ACON | +0.01755 | 1.00000 |
| CCA-Lite | +0.00087 | 1.00000 |
| SiLU+Context (B0H) | +0.01973 | 1.00000 |

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

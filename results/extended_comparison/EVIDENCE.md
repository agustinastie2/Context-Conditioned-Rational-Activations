# Confirmatory activation comparison

This evidence file is generated from paired runs; it is not a paper source.

## Text

| Method | n | Accuracy (mean ± sd) | 95% CI |
|---|---:|---:|---:|
| SiLU (original) | 10 | 92.27 ± 2.50 | [90.82, 93.78] |
| SiLU (budget-matched) | 10 | 92.40 ± 2.43 | [91.00, 93.87] |
| Static Rational | 10 | 95.55 ± 1.65 | [94.58, 96.52] |
| Dynamic ReLU-B | 10 | 96.01 ± 1.53 | [95.12, 96.91] |
| Meta-ACON | 10 | 95.68 ± 1.61 | [94.73, 96.64] |
| CCA-Lite | 10 | 95.80 ± 1.49 | [94.93, 96.69] |

CCA-Lite paired comparisons (two-sided exact permutation; Holm corrected):

| Competitor | Gain (pp) | 95% CI | Cohen dz | p-Holm |
|---|---:|---:|---:|---:|
| SiLU (original) | +3.52 | [+2.88, +4.14] | 3.318 | 0.00977 |
| SiLU (budget-matched) | +3.40 | [+2.79, +3.96] | 3.434 | 0.00977 |
| Static Rational | +0.24 | [+0.06, +0.49] | 0.649 | 0.07617 |
| Dynamic ReLU-B | -0.21 | [-0.37, -0.05] | -0.787 | 0.07812 |
| Meta-ACON | +0.12 | [-0.04, +0.28] | 0.425 | 0.21289 |

Budget utilization:

| Method | Parameters vs target | FLOPs vs target |
|---|---:|---:|
| SiLU (original) | -6.23% | -11.01% |
| SiLU (budget-matched) | +0.67% | -4.48% |
| Static Rational | -0.49% | -0.89% |
| Dynamic ReLU-B | +0.33% | -3.36% |
| Meta-ACON | +0.14% | -3.22% |
| CCA-Lite | +0.00% | +0.00% |

Measured forward+backward cost:

| Method | Eager ms | Compiled ms | Peak MB |
|---|---:|---:|---:|
| SiLU (original) | 2.569 | 2.077 | 20.9 |
| SiLU (budget-matched) | 2.430 | 2.124 | 21.5 |
| Static Rational | 5.436 | 3.324 | 25.2 |
| Dynamic ReLU-B | 6.514 | 4.958 | 22.4 |
| Meta-ACON | 6.454 | 3.898 | 22.3 |
| CCA-Lite | 9.959 | 4.087 | 25.9 |

## Digits

| Method | n | Accuracy (mean ± sd) | 95% CI |
|---|---:|---:|---:|
| SiLU (original) | 10 | 80.76 ± 1.30 | [79.98, 81.50] |
| SiLU (budget-matched) | 10 | 81.28 ± 1.11 | [80.61, 81.91] |
| Static Rational | 10 | 83.82 ± 1.00 | [83.20, 84.38] |
| Dynamic ReLU-B | 10 | 87.81 ± 0.54 | [87.49, 88.12] |
| Meta-ACON | 10 | 83.44 ± 0.69 | [83.05, 83.87] |
| CCA-Lite | 10 | 87.86 ± 0.51 | [87.54, 88.14] |

Per-domain mean accuracy:

| Method | MNIST | SVHN | USPS |
|---|---:|---:|---:|
| SiLU (original) | 95.66 | 53.13 | 93.50 |
| SiLU (budget-matched) | 95.75 | 54.42 | 93.66 |
| Static Rational | 96.06 | 61.41 | 93.99 |
| Dynamic ReLU-B | 96.54 | 72.65 | 94.23 |
| Meta-ACON | 95.85 | 60.52 | 93.94 |
| CCA-Lite | 96.62 | 72.74 | 94.21 |

CCA-Lite paired comparisons (two-sided exact permutation; Holm corrected):

| Competitor | Gain (pp) | 95% CI | Cohen dz | p-Holm |
|---|---:|---:|---:|---:|
| SiLU (original) | +7.10 | [+6.46, +7.73] | 6.530 | 0.00977 |
| SiLU (budget-matched) | +6.58 | [+5.99, +7.19] | 6.349 | 0.00977 |
| Static Rational | +4.03 | [+3.58, +4.47] | 5.331 | 0.00977 |
| Dynamic ReLU-B | +0.05 | [-0.11, +0.20] | 0.188 | 0.58594 |
| Meta-ACON | +4.42 | [+3.97, +4.89] | 5.673 | 0.00977 |

Budget utilization:

| Method | Parameters vs target | FLOPs vs target |
|---|---:|---:|
| SiLU (original) | -18.98% | -14.62% |
| SiLU (budget-matched) | +2.12% | -0.17% |
| Static Rational | -0.43% | -1.45% |
| Dynamic ReLU-B | +0.11% | +0.16% |
| Meta-ACON | +0.16% | +0.17% |
| CCA-Lite | +0.00% | +0.00% |

Measured forward+backward cost:

| Method | Eager ms | Compiled ms | Peak MB |
|---|---:|---:|---:|
| SiLU (original) | 2.957 | 3.337 | 79.9 |
| SiLU (budget-matched) | 2.883 | 3.433 | 84.7 |
| Static Rational | 9.205 | 3.790 | 209.7 |
| Dynamic ReLU-B | 8.146 | 5.291 | 137.7 |
| Meta-ACON | 7.379 | 4.609 | 132.9 |
| CCA-Lite | 12.355 | 5.438 | 216.4 |

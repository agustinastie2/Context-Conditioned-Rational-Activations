# Confirmatory activation comparison

This evidence file is generated from paired runs; it is not a paper source.

## Text

| Method | n | Accuracy (mean ± sd) | 95% CI |
|---|---:|---:|---:|
| SiLU (original) | 1 | 71.78 ± 0.00 | [71.78, 71.78] |
| SiLU (budget-matched) | 1 | 72.07 ± 0.00 | [72.07, 72.07] |
| Static Rational | 1 | 75.49 ± 0.00 | [75.49, 75.49] |
| Dynamic ReLU-B | 1 | 77.64 ± 0.00 | [77.64, 77.64] |
| Meta-ACON | 1 | 74.12 ± 0.00 | [74.12, 74.12] |
| CCA-Lite | 1 | 75.20 ± 0.00 | [75.20, 75.20] |

CCA-Lite paired comparisons (two-sided exact permutation; Holm corrected):

| Competitor | Gain (pp) | 95% CI | Cohen dz | p-Holm |
|---|---:|---:|---:|---:|
| SiLU (original) | +3.42 | [+3.42, +3.42] | inf | 1.00000 |
| SiLU (budget-matched) | +3.12 | [+3.12, +3.12] | inf | 1.00000 |
| Static Rational | -0.29 | [-0.29, -0.29] | 0.000 | 1.00000 |
| Dynamic ReLU-B | -2.44 | [-2.44, -2.44] | 0.000 | 1.00000 |
| Meta-ACON | +1.07 | [+1.07, +1.07] | inf | 1.00000 |

Budget utilization:

| Method | Parameters vs target | FLOPs vs target |
|---|---:|---:|
| SiLU (original) | -6.23% | -11.01% |
| SiLU (budget-matched) | +0.67% | -4.48% |
| Static Rational | -0.49% | -0.89% |
| Dynamic ReLU-B | +0.33% | -3.36% |
| Meta-ACON | +0.14% | -3.22% |
| CCA-Lite | +0.00% | +0.00% |

## Digits

| Method | n | Accuracy (mean ± sd) | 95% CI |
|---|---:|---:|---:|
| SiLU (original) | 1 | 20.31 ± 0.00 | [20.31, 20.31] |
| SiLU (budget-matched) | 1 | 21.48 ± 0.00 | [21.48, 21.48] |
| Static Rational | 1 | 20.96 ± 0.00 | [20.96, 20.96] |
| Dynamic ReLU-B | 1 | 28.26 ± 0.00 | [28.26, 28.26] |
| Meta-ACON | 1 | 21.09 ± 0.00 | [21.09, 21.09] |
| CCA-Lite | 1 | 20.96 ± 0.00 | [20.96, 20.96] |

CCA-Lite paired comparisons (two-sided exact permutation; Holm corrected):

| Competitor | Gain (pp) | 95% CI | Cohen dz | p-Holm |
|---|---:|---:|---:|---:|
| SiLU (original) | +0.65 | [+0.65, +0.65] | inf | 1.00000 |
| SiLU (budget-matched) | -0.52 | [-0.52, -0.52] | 0.000 | 1.00000 |
| Static Rational | +0.00 | [+0.00, +0.00] | 0.000 | 1.00000 |
| Dynamic ReLU-B | -7.29 | [-7.29, -7.29] | 0.000 | 1.00000 |
| Meta-ACON | -0.13 | [-0.13, -0.13] | 0.000 | 1.00000 |

Budget utilization:

| Method | Parameters vs target | FLOPs vs target |
|---|---:|---:|
| SiLU (original) | -18.98% | -14.62% |
| SiLU (budget-matched) | +2.12% | -0.17% |
| Static Rational | -0.43% | -1.45% |
| Dynamic ReLU-B | +0.11% | +0.16% |
| Meta-ACON | +0.16% | +0.17% |
| CCA-Lite | +0.00% | +0.00% |

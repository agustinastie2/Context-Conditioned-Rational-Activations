# Fresh-seed CCA-Hinge confirmation

Preset selected on seeds 2001–2003; confirmation uses untouched seeds 3001–3010.

## Text

| Method | Accuracy mean ± sd | 95% CI |
|---|---:|---:|
| SiLU (budget-matched) | 90.58 ± 3.33 | [88.59, 92.49] |
| Static Rational | 94.51 ± 2.36 | [93.08, 95.85] |
| Dynamic ReLU-B | 94.87 ± 1.97 | [93.68, 95.98] |
| Meta-ACON | 94.53 ± 2.30 | [93.12, 95.80] |
| CCA-Lite | 94.75 ± 2.01 | [93.55, 95.89] |
| CCA-Hinge | 94.75 ± 2.13 | [93.50, 95.98] |

CCA-Hinge paired gain:

| Comparator | Gain pp | 95% CI | p-Holm |
|---|---:|---:|---:|
| SiLU (budget-matched) | +4.17 | [+3.38, +4.98] | 0.00977 |
| Static Rational | +0.24 | [-0.00, +0.56] | 0.49219 |
| Dynamic ReLU-B | -0.12 | [-0.36, +0.10] | 0.77344 |
| Meta-ACON | +0.22 | [-0.01, +0.49] | 0.49219 |
| CCA-Lite | -0.00 | [-0.14, +0.14] | 0.97461 |

| Method | Params vs CCA-Hinge | FLOPs vs CCA-Hinge |
|---|---:|---:|
| SiLU (budget-matched) | +0.73% | -5.94% |
| Static Rational | +0.81% | -1.10% |
| Dynamic ReLU-B | +0.59% | -4.65% |
| Meta-ACON | +0.17% | -4.73% |
| CCA-Lite | -1.32% | -2.89% |
| CCA-Hinge | +0.00% | +0.00% |

| Method | Eager train ms | Compiled train ms | Compiled samples/s |
|---|---:|---:|---:|
| SiLU (budget-matched) | 2.503 | 2.091 | 122429 |
| Static Rational | 6.044 | 2.876 | 89000 |
| Dynamic ReLU-B | 6.359 | 3.656 | 70028 |
| Meta-ACON | 5.966 | 3.681 | 69541 |
| CCA-Lite | 9.052 | 3.747 | 68326 |
| CCA-Hinge | 10.519 | 3.823 | 66970 |

## Digits

| Method | Accuracy mean ± sd | 95% CI |
|---|---:|---:|
| SiLU (budget-matched) | 81.14 ± 1.17 | [80.41, 81.81] |
| Static Rational | 83.14 ± 0.79 | [82.68, 83.60] |
| Dynamic ReLU-B | 88.06 ± 0.98 | [87.47, 88.63] |
| Meta-ACON | 83.58 ± 1.03 | [83.01, 84.20] |
| CCA-Lite | 88.11 ± 0.68 | [87.74, 88.54] |
| CCA-Hinge | 88.68 ± 0.50 | [88.38, 88.98] |

CCA-Hinge paired gain:

| Comparator | Gain pp | 95% CI | p-Holm |
|---|---:|---:|---:|
| SiLU (budget-matched) | +7.54 | [+6.65, +8.48] | 0.00977 |
| Static Rational | +5.54 | [+5.03, +6.01] | 0.00977 |
| Dynamic ReLU-B | +0.62 | [+0.21, +1.11] | 0.02344 |
| Meta-ACON | +5.10 | [+4.40, +5.77] | 0.00977 |
| CCA-Lite | +0.57 | [+0.30, +0.86] | 0.01172 |

| Method | Params vs CCA-Hinge | FLOPs vs CCA-Hinge |
|---|---:|---:|
| SiLU (budget-matched) | -0.66% | +1.63% |
| Static Rational | -0.34% | +2.96% |
| Dynamic ReLU-B | +0.93% | +0.15% |
| Meta-ACON | -0.08% | +0.02% |
| CCA-Lite | -4.12% | -4.39% |
| CCA-Hinge | +0.00% | +0.00% |

| Method | Eager train ms | Compiled train ms | Compiled samples/s |
|---|---:|---:|---:|
| SiLU (budget-matched) | 3.164 | 3.020 | 84774 |
| Static Rational | 10.064 | 4.267 | 59995 |
| Dynamic ReLU-B | 7.804 | 4.709 | 54360 |
| Meta-ACON | 6.473 | 4.797 | 53362 |
| CCA-Lite | 12.341 | 5.665 | 45192 |
| CCA-Hinge | 13.553 | n/a | n/a |

Compiled timing is unavailable for CCA-Hinge because the local Windows Triton launcher could not compile its generated C wrapper. Eager execution and all correctness/training runs completed; `costs.json` retains the full compiler error for reproducibility.

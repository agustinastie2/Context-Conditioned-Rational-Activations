# Complete paper experiment matrix

Paired seeds: 4101, 4102, 4103.

Image-generation protocol(s): 500 steps, BF16.

## image_gen_mnist

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 3 | 1.07827 ± 0.00651 | [1.07077, 1.08246] | -5.98% |
| Static Rational | 3 | 1.06553 ± 0.00611 | [1.05887, 1.07085] | -5.70% |
| Dynamic ReLU-B | 3 | 0.94892 ± 0.00707 | [0.94473, 0.95708] | -3.24% |
| Meta-ACON | 3 | 0.95240 ± 0.00357 | [0.94953, 0.95640] | -4.99% |
| CCA-Lite | 3 | 0.93949 ± 0.00272 | [0.93781, 0.94262] | -1.21% |
| CCA-Hinge | 3 | 0.93898 ± 0.00309 | [0.93613, 0.94227] | +0.00% |
| SiLU+Context (B0H) | 3 | 0.95616 ± 0.00595 | [0.95093, 0.96263] | -5.34% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.13929 | 1.00000 |
| Static Rational | +0.12656 | 1.00000 |
| Dynamic ReLU-B | +0.00994 | 1.00000 |
| Meta-ACON | +0.01342 | 1.00000 |
| CCA-Lite | +0.00051 | 1.00000 |
| SiLU+Context (B0H) | +0.01718 | 1.00000 |

## image_gen_fashion

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 3 | 1.59416 ± 0.00889 | [1.58392, 1.59986] | -5.98% |
| Static Rational | 3 | 1.56526 ± 0.01092 | [1.55341, 1.57491] | -5.70% |
| Dynamic ReLU-B | 3 | 1.32868 ± 0.00103 | [1.32785, 1.32983] | -3.24% |
| Meta-ACON | 3 | 1.33542 ± 0.00368 | [1.33163, 1.33897] | -4.99% |
| CCA-Lite | 3 | 1.28984 ± 0.00407 | [1.28544, 1.29347] | -1.21% |
| CCA-Hinge | 3 | 1.28639 ± 0.00233 | [1.28394, 1.28859] | +0.00% |
| SiLU+Context (B0H) | 3 | 1.35360 ± 0.00492 | [1.34817, 1.35778] | -5.34% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.30778 | 1.00000 |
| Static Rational | +0.27887 | 1.00000 |
| Dynamic ReLU-B | +0.04229 | 1.00000 |
| Meta-ACON | +0.04903 | 1.00000 |
| CCA-Lite | +0.00346 | 1.00000 |
| SiLU+Context (B0H) | +0.06721 | 1.00000 |

## image_gen_cifar

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 3 | 1.58789 ± 0.00495 | [1.58289, 1.59279] | -4.87% |
| Static Rational | 3 | 1.58098 ± 0.00478 | [1.57679, 1.58619] | -4.64% |
| Dynamic ReLU-B | 3 | 1.45516 ± 0.00591 | [1.44836, 1.45902] | -2.64% |
| Meta-ACON | 3 | 1.46125 ± 0.00410 | [1.45733, 1.46551] | -4.06% |
| CCA-Lite | 3 | 1.42999 ± 0.00602 | [1.42629, 1.43694] | -0.99% |
| CCA-Hinge | 3 | 1.42692 ± 0.00236 | [1.42452, 1.42923] | +0.00% |
| SiLU+Context (B0H) | 3 | 1.46534 ± 0.00119 | [1.46426, 1.46662] | -4.35% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.16096 | 1.00000 |
| Static Rational | +0.15405 | 1.00000 |
| Dynamic ReLU-B | +0.02824 | 1.00000 |
| Meta-ACON | +0.03432 | 1.00000 |
| CCA-Lite | +0.00306 | 1.00000 |
| SiLU+Context (B0H) | +0.03842 | 1.00000 |

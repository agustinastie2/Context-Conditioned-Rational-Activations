# Complete paper experiment matrix

Paired seeds: 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110.

Image-generation protocol(s): 500 steps, BF16.

## image_gen_mnist

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 1.08261 ± 0.00742 | [1.07828, 1.08698] | -5.98% |
| Static Rational | 10 | 1.06890 ± 0.00751 | [1.06443, 1.07329] | -5.70% |
| Dynamic ReLU-B | 10 | 0.94856 ± 0.00606 | [0.94518, 0.95227] | -3.24% |
| Meta-ACON | 10 | 0.95129 ± 0.00567 | [0.94810, 0.95472] | -4.99% |
| CCA-Lite | 10 | 0.93783 ± 0.00564 | [0.93434, 0.94085] | -1.21% |
| CCA-Hinge | 10 | 0.93729 ± 0.00674 | [0.93347, 0.94128] | +0.00% |
| SiLU+Context (B0H) | 10 | 0.95536 ± 0.00692 | [0.95152, 0.95958] | -5.34% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.14533 | 0.01172 |
| Static Rational | +0.13162 | 0.01172 |
| Dynamic ReLU-B | +0.01128 | 0.01172 |
| Meta-ACON | +0.01400 | 0.01172 |
| CCA-Lite | +0.00054 | 0.55664 |
| SiLU+Context (B0H) | +0.01807 | 0.01172 |

## image_gen_fashion

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 1.59652 ± 0.00899 | [1.59087, 1.60140] | -5.98% |
| Static Rational | 10 | 1.56652 ± 0.00825 | [1.56133, 1.57091] | -5.70% |
| Dynamic ReLU-B | 10 | 1.32736 ± 0.00935 | [1.32175, 1.33281] | -3.24% |
| Meta-ACON | 10 | 1.33513 ± 0.00727 | [1.33088, 1.33936] | -4.99% |
| CCA-Lite | 10 | 1.28633 ± 0.00977 | [1.28031, 1.29177] | -1.21% |
| CCA-Hinge | 10 | 1.28423 ± 0.00879 | [1.27907, 1.28925] | +0.00% |
| SiLU+Context (B0H) | 10 | 1.35039 ± 0.00828 | [1.34563, 1.35526] | -5.34% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.31229 | 0.01172 |
| Static Rational | +0.28228 | 0.01172 |
| Dynamic ReLU-B | +0.04313 | 0.01172 |
| Meta-ACON | +0.05090 | 0.01172 |
| CCA-Lite | +0.00210 | 0.14648 |
| SiLU+Context (B0H) | +0.06616 | 0.01172 |

## image_gen_cifar

Metric: `bits_per_dim` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 1.58777 ± 0.00782 | [1.58324, 1.59253] | -4.87% |
| Static Rational | 10 | 1.58107 ± 0.00707 | [1.57695, 1.58529] | -4.64% |
| Dynamic ReLU-B | 10 | 1.45629 ± 0.00469 | [1.45359, 1.45911] | -2.64% |
| Meta-ACON | 10 | 1.46890 ± 0.00721 | [1.46480, 1.47331] | -4.06% |
| CCA-Lite | 10 | 1.43034 ± 0.00467 | [1.42761, 1.43306] | -0.99% |
| CCA-Hinge | 10 | 1.43018 ± 0.00658 | [1.42648, 1.43420] | +0.00% |
| SiLU+Context (B0H) | 10 | 1.47035 ± 0.00713 | [1.46620, 1.47457] | -4.35% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.15758 | 0.01172 |
| Static Rational | +0.15089 | 0.01172 |
| Dynamic ReLU-B | +0.02611 | 0.01172 |
| Meta-ACON | +0.03872 | 0.01172 |
| CCA-Lite | +0.00016 | 0.91797 |
| SiLU+Context (B0H) | +0.04017 | 0.01172 |

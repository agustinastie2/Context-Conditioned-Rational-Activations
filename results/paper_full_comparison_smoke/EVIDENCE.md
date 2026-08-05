# Complete paper experiment matrix

Paired seeds: 994.

## text_char

Metric: `ppl` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 1 | 53.90868 ± 0.00000 | [53.90868, 53.90868] | -4.77% |
| Static Rational | 1 | 50.33888 ± 0.00000 | [50.33888, 50.33888] | -2.11% |
| Dynamic ReLU-B | 1 | 49.17587 ± 0.00000 | [49.17587, 49.17587] | -0.69% |
| Meta-ACON | 1 | 62.24439 ± 0.00000 | [62.24439, 62.24439] | -1.06% |
| CCA-Lite | 1 | 50.03499 ± 0.00000 | [50.03499, 50.03499] | -0.26% |
| CCA-Hinge | 1 | 55.71010 ± 0.00000 | [55.71010, 55.71010] | +0.00% |
| SiLU+Context (B0H) | 1 | 50.36950 ± 0.00000 | [50.36950, 50.36950] | -2.17% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | -1.80143 | 1.00000 |
| Static Rational | -5.37122 | 1.00000 |
| Dynamic ReLU-B | -6.53423 | 1.00000 |
| Meta-ACON | +6.53429 | 1.00000 |
| CCA-Lite | -5.67511 | 1.00000 |
| SiLU+Context (B0H) | -5.34060 | 1.00000 |

## text_bpe_2000

Metric: `ppl` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 1 | 1687.69473 ± 0.00000 | [1687.69473, 1687.69473] | -3.43% |
| Static Rational | 1 | 1408.49153 ± 0.00000 | [1408.49153, 1408.49153] | -1.52% |
| Dynamic ReLU-B | 1 | 1488.64701 ± 0.00000 | [1488.64701, 1488.64701] | -0.49% |
| Meta-ACON | 1 | 1487.05700 ± 0.00000 | [1487.05700, 1487.05700] | -0.76% |
| CCA-Lite | 1 | 1560.27927 ± 0.00000 | [1560.27927, 1560.27927] | -0.18% |
| CCA-Hinge | 1 | 1663.29189 ± 0.00000 | [1663.29189, 1663.29189] | +0.00% |
| SiLU+Context (B0H) | 1 | 1409.28695 ± 0.00000 | [1409.28695, 1409.28695] | -1.56% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +24.40284 | 1.00000 |
| Static Rational | -254.80036 | 1.00000 |
| Dynamic ReLU-B | -174.64488 | 1.00000 |
| Meta-ACON | -176.23488 | 1.00000 |
| CCA-Lite | -103.01262 | 1.00000 |
| SiLU+Context (B0H) | -254.00494 | 1.00000 |

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
| SiLU | 1 | 3.05407 ± 0.00000 | [3.05407, 3.05407] | -7.22% |
| Static Rational | 1 | 3.05398 ± 0.00000 | [3.05398, 3.05398] | -6.87% |
| Dynamic ReLU-B | 1 | 3.10827 ± 0.00000 | [3.10827, 3.10827] | -3.91% |
| Meta-ACON | 1 | 3.15854 ± 0.00000 | [3.15854, 3.15854] | -6.01% |
| CCA-Lite | 1 | 2.99682 ± 0.00000 | [2.99682, 2.99682] | -1.46% |
| CCA-Hinge | 1 | 3.10279 ± 0.00000 | [3.10279, 3.10279] | +0.00% |
| SiLU+Context (B0H) | 1 | 3.08759 ± 0.00000 | [3.08759, 3.08759] | -6.44% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | -0.04872 | 1.00000 |
| Static Rational | -0.04881 | 1.00000 |
| Dynamic ReLU-B | +0.00548 | 1.00000 |
| Meta-ACON | +0.05574 | 1.00000 |
| CCA-Lite | -0.10597 | 1.00000 |
| SiLU+Context (B0H) | -0.01520 | 1.00000 |

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
| SiLU | 1 | 0.09250 ± 0.00000 | [0.09250, 0.09250] | -22.32% |
| Static Rational | 1 | 0.09670 ± 0.00000 | [0.09670, 0.09670] | -21.35% |
| Dynamic ReLU-B | 1 | 0.11740 ± 0.00000 | [0.11740, 0.11740] | -11.04% |
| Meta-ACON | 1 | 0.09250 ± 0.00000 | [0.09250, 0.09250] | -16.98% |
| CCA-Lite | 1 | 0.10360 ± 0.00000 | [0.10360, 0.10360] | -4.12% |
| CCA-Hinge | 1 | 0.10400 ± 0.00000 | [0.10400, 0.10400] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.01150 | 1.00000 |
| Static Rational | +0.00730 | 1.00000 |
| Dynamic ReLU-B | -0.01340 | 1.00000 |
| Meta-ACON | +0.01150 | 1.00000 |
| CCA-Lite | +0.00040 | 1.00000 |

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
| SiLU | 1 | 0.29187 ± 0.00000 | [0.29187, 0.29187] | -7.47% |
| Static Rational | 1 | 0.28900 ± 0.00000 | [0.28900, 0.28900] | -7.16% |
| Dynamic ReLU-B | 1 | 0.30025 ± 0.00000 | [0.30025, 0.30025] | -3.54% |
| Meta-ACON | 1 | 0.29175 ± 0.00000 | [0.29175, 0.29175] | -5.45% |
| CCA-Lite | 1 | 0.28925 ± 0.00000 | [0.28925, 0.28925] | -1.32% |
| CCA-Hinge | 1 | 0.28950 ± 0.00000 | [0.28950, 0.28950] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | -0.00238 | 1.00000 |
| Static Rational | +0.00050 | 1.00000 |
| Dynamic ReLU-B | -0.01075 | 1.00000 |
| Meta-ACON | -0.00225 | 1.00000 |
| CCA-Lite | +0.00025 | 1.00000 |

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
| SiLU | 1 | 0.12017 ± 0.00000 | [0.12017, 0.12017] | -22.32% |
| Static Rational | 1 | 0.11733 ± 0.00000 | [0.11733, 0.11733] | -21.35% |
| Dynamic ReLU-B | 1 | 0.09600 ± 0.00000 | [0.09600, 0.09600] | -11.04% |
| Meta-ACON | 1 | 0.12017 ± 0.00000 | [0.12017, 0.12017] | -16.98% |
| CCA-Lite | 1 | 0.10933 ± 0.00000 | [0.10933, 0.10933] | -4.12% |
| CCA-Hinge | 1 | 0.10917 ± 0.00000 | [0.10917, 0.10917] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | -0.01100 | 1.00000 |
| Static Rational | -0.00817 | 1.00000 |
| Dynamic ReLU-B | +0.01317 | 1.00000 |
| Meta-ACON | -0.01100 | 1.00000 |
| CCA-Lite | -0.00017 | 1.00000 |

## algo_modular

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 1 | 0.05120 ± 0.00000 | [0.05120, 0.05120] | -7.02% |
| Static Rational | 1 | 0.05150 ± 0.00000 | [0.05150, 0.05150] | -6.73% |
| Dynamic ReLU-B | 1 | 0.05030 ± 0.00000 | [0.05030, 0.05030] | -3.33% |
| Meta-ACON | 1 | 0.05110 ± 0.00000 | [0.05110, 0.05110] | -5.12% |
| CCA-Lite | 1 | 0.05050 ± 0.00000 | [0.05050, 0.05050] | -1.24% |
| CCA-Hinge | 1 | 0.05030 ± 0.00000 | [0.05030, 0.05030] | +0.00% |
| SiLU+Context (B0H) | 1 | 0.05160 ± 0.00000 | [0.05160, 0.05160] | -7.02% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | -0.00090 | 1.00000 |
| Static Rational | -0.00120 | 1.00000 |
| Dynamic ReLU-B | +0.00000 | 1.00000 |
| Meta-ACON | -0.00080 | 1.00000 |
| CCA-Lite | -0.00020 | 1.00000 |
| SiLU+Context (B0H) | -0.00130 | 1.00000 |

## audio_digits

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 1 | 0.35667 ± 0.00000 | [0.35667, 0.35667] | -18.89% |
| Static Rational | 1 | 0.35167 ± 0.00000 | [0.35167, 0.35167] | -18.06% |
| Dynamic ReLU-B | 1 | 0.26667 ± 0.00000 | [0.26667, 0.26667] | -9.34% |
| Meta-ACON | 1 | 0.35833 ± 0.00000 | [0.35833, 0.35833] | -14.37% |
| CCA-Lite | 1 | 0.36000 ± 0.00000 | [0.36000, 0.36000] | -3.49% |
| CCA-Hinge | 1 | 0.36167 ± 0.00000 | [0.36167, 0.36167] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.00500 | 1.00000 |
| Static Rational | +0.01000 | 1.00000 |
| Dynamic ReLU-B | +0.09500 | 1.00000 |
| Meta-ACON | +0.00333 | 1.00000 |
| CCA-Lite | +0.00167 | 1.00000 |

## text_sentiment

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 1 | 0.49062 ± 0.00000 | [0.49062, 0.49062] | -2.94% |
| Static Rational | 1 | 0.49062 ± 0.00000 | [0.49062, 0.49062] | -2.82% |
| Dynamic ReLU-B | 1 | 0.49062 ± 0.00000 | [0.49062, 0.49062] | -1.39% |
| Meta-ACON | 1 | 0.49062 ± 0.00000 | [0.49062, 0.49062] | -2.14% |
| CCA-Lite | 1 | 0.49062 ± 0.00000 | [0.49062, 0.49062] | -0.52% |
| CCA-Hinge | 1 | 0.49062 ± 0.00000 | [0.49062, 0.49062] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.00000 | 1.00000 |
| Static Rational | +0.00000 | 1.00000 |
| Dynamic ReLU-B | +0.00000 | 1.00000 |
| Meta-ACON | +0.00000 | 1.00000 |
| CCA-Lite | +0.00000 | 1.00000 |

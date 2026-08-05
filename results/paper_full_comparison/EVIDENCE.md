# Complete paper experiment matrix

Paired seeds: 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010.

## text_char

Metric: `ppl` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 3 | 21.56471 ± 0.09781 | [21.48137, 21.67239] | -4.77% |
| Static Rational | 3 | 21.06986 ± 0.37088 | [20.64242, 21.30653] | -2.11% |
| Dynamic ReLU-B | 3 | 11.81198 ± 0.14078 | [11.65952, 11.93705] | -0.69% |
| Meta-ACON | 3 | 12.61112 ± 0.34776 | [12.36171, 13.00837] | -1.06% |
| CCA-Lite | 3 | 11.89184 ± 0.27045 | [11.62908, 12.16938] | -0.26% |
| CCA-Hinge | 3 | 11.48015 ± 0.09926 | [11.38944, 11.58618] | +0.00% |
| SiLU+Context (B0H) | 3 | 13.65642 ± 0.88990 | [12.66032, 14.37304] | -2.17% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +10.08456 | 1.00000 |
| Static Rational | +9.58970 | 1.00000 |
| Dynamic ReLU-B | +0.33183 | 1.00000 |
| Meta-ACON | +1.13096 | 1.00000 |
| CCA-Lite | +0.41169 | 1.00000 |
| SiLU+Context (B0H) | +2.17627 | 1.00000 |

## image_cls

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 0.85131 ± 0.00459 | [0.84847, 0.85386] | -22.32% |
| Static Rational | 10 | 0.86614 ± 0.00395 | [0.86382, 0.86848] | -21.35% |
| Dynamic ReLU-B | 10 | 0.87936 ± 0.00366 | [0.87714, 0.88141] | -11.04% |
| Meta-ACON | 10 | 0.86485 ± 0.00514 | [0.86183, 0.86784] | -16.98% |
| CCA-Lite | 10 | 0.87866 ± 0.00728 | [0.87438, 0.88298] | -4.12% |
| CCA-Hinge | 10 | 0.88025 ± 0.00701 | [0.87659, 0.88471] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.02894 | 0.00977 |
| Static Rational | +0.01411 | 0.00977 |
| Dynamic ReLU-B | +0.00089 | 0.85156 |
| Meta-ACON | +0.01540 | 0.00977 |
| CCA-Lite | +0.00159 | 0.30469 |

## image_cls_single

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 0.98248 ± 0.00100 | [0.98191, 0.98309] | -22.32% |
| Static Rational | 10 | 0.98292 ± 0.00261 | [0.98135, 0.98437] | -21.35% |
| Dynamic ReLU-B | 10 | 0.98410 ± 0.00199 | [0.98291, 0.98528] | -11.04% |
| Meta-ACON | 10 | 0.98331 ± 0.00158 | [0.98236, 0.98421] | -16.98% |
| CCA-Lite | 10 | 0.98446 ± 0.00241 | [0.98305, 0.98585] | -4.12% |
| CCA-Hinge | 10 | 0.98488 ± 0.00233 | [0.98350, 0.98621] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.00240 | 0.02344 |
| Static Rational | +0.00196 | 0.01953 |
| Dynamic ReLU-B | +0.00078 | 0.35156 |
| Meta-ACON | +0.00157 | 0.08789 |
| CCA-Lite | +0.00042 | 0.35156 |

## text_cls

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 0.93265 ± 0.01685 | [0.92226, 0.94235] | -7.47% |
| Static Rational | 10 | 0.96215 ± 0.01120 | [0.95569, 0.96894] | -7.16% |
| Dynamic ReLU-B | 10 | 0.96556 ± 0.01049 | [0.95927, 0.97169] | -3.54% |
| Meta-ACON | 10 | 0.96225 ± 0.01026 | [0.95629, 0.96841] | -5.45% |
| CCA-Lite | 10 | 0.96359 ± 0.00983 | [0.95798, 0.96949] | -1.32% |
| CCA-Hinge | 10 | 0.96451 ± 0.01034 | [0.95830, 0.97058] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.03186 | 0.00977 |
| Static Rational | +0.00236 | 0.07031 |
| Dynamic ReLU-B | -0.00105 | 0.53125 |
| Meta-ACON | +0.00226 | 0.08203 |
| CCA-Lite | +0.00092 | 0.59375 |

## text_cls_single

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 0.99419 ± 0.00341 | [0.99196, 0.99599] | -7.47% |
| Static Rational | 10 | 0.99551 ± 0.00177 | [0.99443, 0.99650] | -7.16% |
| Dynamic ReLU-B | 10 | 0.99639 ± 0.00162 | [0.99540, 0.99729] | -3.54% |
| Meta-ACON | 10 | 0.99526 ± 0.00192 | [0.99409, 0.99635] | -5.45% |
| CCA-Lite | 10 | 0.99594 ± 0.00165 | [0.99492, 0.99687] | -1.32% |
| CCA-Hinge | 10 | 0.99546 ± 0.00224 | [0.99414, 0.99678] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.00128 | 0.37500 |
| Static Rational | -0.00005 | 1.00000 |
| Dynamic ReLU-B | -0.00092 | 0.26367 |
| Meta-ACON | +0.00020 | 1.00000 |
| CCA-Lite | -0.00047 | 0.93750 |

## digits_multidomain

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 0.81248 ± 0.01120 | [0.80562, 0.81875] | -22.32% |
| Static Rational | 10 | 0.83417 ± 0.00984 | [0.82875, 0.84028] | -21.35% |
| Dynamic ReLU-B | 10 | 0.87523 ± 0.00888 | [0.87017, 0.88062] | -11.04% |
| Meta-ACON | 10 | 0.83223 ± 0.01740 | [0.82143, 0.84177] | -16.98% |
| CCA-Lite | 10 | 0.88537 ± 0.00548 | [0.88225, 0.88862] | -4.12% |
| CCA-Hinge | 10 | 0.88958 ± 0.00721 | [0.88522, 0.89362] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.07710 | 0.00977 |
| Static Rational | +0.05542 | 0.00977 |
| Dynamic ReLU-B | +0.01435 | 0.00977 |
| Meta-ACON | +0.05735 | 0.00977 |
| CCA-Lite | +0.00422 | 0.01172 |

## algo_modular

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 0.08209 ± 0.00512 | [0.07896, 0.08495] | -7.02% |
| Static Rational | 10 | 0.16091 ± 0.00523 | [0.15789, 0.16401] | -6.73% |
| Dynamic ReLU-B | 10 | 0.29768 ± 0.08213 | [0.25149, 0.34803] | -3.33% |
| Meta-ACON | 10 | 0.23239 ± 0.03560 | [0.21502, 0.25655] | -5.12% |
| CCA-Lite | 10 | 0.47098 ± 0.10083 | [0.41487, 0.53470] | -1.24% |
| CCA-Hinge | 10 | 0.49877 ± 0.09453 | [0.44760, 0.55840] | +0.00% |
| SiLU+Context (B0H) | 10 | 0.12280 ± 0.01413 | [0.11442, 0.13113] | -7.02% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.41668 | 0.01172 |
| Static Rational | +0.33786 | 0.01172 |
| Dynamic ReLU-B | +0.20109 | 0.01172 |
| Meta-ACON | +0.26638 | 0.01172 |
| CCA-Lite | +0.02779 | 0.51172 |
| SiLU+Context (B0H) | +0.37597 | 0.01172 |

## audio_digits

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 0.72767 ± 0.02493 | [0.71316, 0.74250] | -18.89% |
| Static Rational | 10 | 0.74733 ± 0.02003 | [0.73533, 0.75883] | -18.06% |
| Dynamic ReLU-B | 10 | 0.73700 ± 0.02497 | [0.72167, 0.75117] | -9.34% |
| Meta-ACON | 10 | 0.73950 ± 0.02650 | [0.72383, 0.75500] | -14.37% |
| CCA-Lite | 10 | 0.73900 ± 0.02245 | [0.72600, 0.75250] | -3.49% |
| CCA-Hinge | 10 | 0.75700 ± 0.02041 | [0.74567, 0.76950] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +0.02933 | 0.00977 |
| Static Rational | +0.00967 | 0.06250 |
| Dynamic ReLU-B | +0.02000 | 0.06250 |
| Meta-ACON | +0.01750 | 0.06250 |
| CCA-Lite | +0.01800 | 0.06250 |

## text_sentiment

Metric: `accuracy` (higher is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 0.82794 ± 0.00582 | [0.82444, 0.83131] | -2.94% |
| Static Rational | 10 | 0.82431 ± 0.00487 | [0.82144, 0.82719] | -2.82% |
| Dynamic ReLU-B | 10 | 0.82312 ± 0.00836 | [0.81781, 0.82775] | -1.39% |
| Meta-ACON | 10 | 0.82300 ± 0.00479 | [0.82050, 0.82612] | -2.14% |
| CCA-Lite | 10 | 0.82037 ± 0.00696 | [0.81650, 0.82462] | -0.52% |
| CCA-Hinge | 10 | 0.82475 ± 0.00484 | [0.82188, 0.82756] | +0.00% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | -0.00319 | 0.17578 |
| Static Rational | +0.00044 | 1.00000 |
| Dynamic ReLU-B | +0.00163 | 1.00000 |
| Meta-ACON | +0.00175 | 1.00000 |
| CCA-Lite | +0.00438 | 0.26562 |

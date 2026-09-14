# Complete paper experiment matrix

Paired seeds: 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010.

## text_char

Metric: `ppl` (lower is better).

| Method | n | Mean ± sd | 95% CI | Params vs Hinge |
|---|---:|---:|---:|---:|
| SiLU | 10 | 21.70276 ± 0.31797 | [21.51903, 21.88939] | -4.77% |
| Static Rational | 10 | 21.64410 ± 0.49571 | [21.34476, 21.93186] | -2.11% |
| Dynamic ReLU-B | 10 | 11.98454 ± 0.21097 | [11.86366, 12.11012] | -0.69% |
| Meta-ACON | 10 | 12.88530 ± 0.55133 | [12.60330, 13.23930] | -1.06% |
| CCA-Lite | 10 | 11.76336 ± 0.19749 | [11.64827, 11.87987] | -0.26% |
| CCA-Hinge | 10 | 11.72328 ± 0.35974 | [11.53663, 11.96019] | +0.00% |
| SiLU+Context (B0H) | 10 | 13.88119 ± 0.85183 | [13.39943, 14.40167] | -2.17% |

| Comparator | CCA-Hinge paired gain | p-Holm |
|---|---:|---:|
| SiLU | +9.97949 | 0.01172 |
| Static Rational | +9.92083 | 0.01172 |
| Dynamic ReLU-B | +0.26127 | 0.11328 |
| Meta-ACON | +1.16202 | 0.01172 |
| CCA-Lite | +0.04008 | 0.81055 |
| SiLU+Context (B0H) | +2.15792 | 0.01172 |

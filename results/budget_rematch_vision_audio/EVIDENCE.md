# Re-corrida con presupuesto igualado (filas de imagen y audio de la Tabla 2)

Protocolo idéntico al de la Tabla 1 y al de `results/budget_rematch`: CCA-Hinge define
el objetivo de recursos y una búsqueda discreta ajusta la red anfitriona de cada
competidor hasta quedar dentro del 105 % de sus parámetros y del 110 % de sus
operaciones por muestra. La fila *SiLU (ancho original)* se informa sin igualar,
como referencia.

Semillas: 4001–4010 (10 emparejadas).

## Imagen, 8 regímenes  (`image_cls`)

Familia de Holm: 5 comparaciones (piso corregido 0.0098).

| Método | Exactitud media ± sd | IC95 | ΔP | ΔF | Barajado |
|---|---:|---:|---:|---:|---:|
| SiLU (ancho original) | 85.13 ± 0.46 | [84.85, 85.39] | -22.32% | -18.37% | — |
| SiLU igualada | 85.56 ± 0.51 | [85.27, 85.87] | -0.66% | +1.63% | — |
| Rational | 87.24 ± 0.32 | [87.06, 87.44] | +0.07% | -1.90% | — |
| Dynamic ReLU-B | 88.07 ± 0.62 | [87.70, 88.43] | +0.93% | +0.15% | — |
| Meta-ACON | 87.00 ± 0.47 | [86.73, 87.28] | -0.08% | +0.02% | — |
| CCA-Lite | 87.90 ± 0.71 | [87.49, 88.33] | -4.12% | -4.39% | 57.78 |
| CCA-Hinge | 88.05 ± 0.71 | [87.67, 88.49] | +0.00% | +0.00% | 57.61 |

Ganancia pareada de CCA-Hinge:

| Comparador | Ganancia pp | IC95 | d_z | p exacto | p-Holm | sig. |
|---|---:|---:|---:|---:|---:|:--:|
| SiLU igualada | +2.49 | [+2.04, +2.95] | +3.27 | 0.00195 | 0.00977 | sí |
| Rational | +0.80 | [+0.42, +1.22] | +1.19 | 0.00586 | 0.01758 | sí |
| Dynamic ReLU-B | -0.02 | [-0.46, +0.56] | -0.03 | 0.95703 | 0.95703 | no |
| Meta-ACON | +1.04 | [+0.74, +1.35] | +2.01 | 0.00195 | 0.00977 | sí |
| CCA-Lite | +0.15 | [+0.02, +0.31] | +0.59 | 0.08984 | 0.17969 | no |

## Imagen, 1 régimen (control)  (`image_cls_single`)

Familia de Holm: 5 comparaciones (piso corregido 0.0098).

| Método | Exactitud media ± sd | IC95 | ΔP | ΔF | Barajado |
|---|---:|---:|---:|---:|---:|
| SiLU (ancho original) | 98.25 ± 0.10 | [98.19, 98.31] | -22.32% | -18.37% | — |
| SiLU igualada | 98.31 ± 0.09 | [98.26, 98.36] | -0.66% | +1.63% | — |
| Rational | 98.41 ± 0.21 | [98.29, 98.53] | +0.07% | -1.90% | — |
| Dynamic ReLU-B | 98.50 ± 0.17 | [98.39, 98.59] | +0.93% | +0.15% | — |
| Meta-ACON | 98.41 ± 0.12 | [98.34, 98.48] | -0.08% | +0.02% | — |
| CCA-Lite | 98.45 ± 0.24 | [98.31, 98.60] | -4.12% | -4.39% | 98.42 |
| CCA-Hinge | 98.49 ± 0.23 | [98.35, 98.62] | +0.00% | +0.00% | 98.46 |

Ganancia pareada de CCA-Hinge:

| Comparador | Ganancia pp | IC95 | d_z | p exacto | p-Holm | sig. |
|---|---:|---:|---:|---:|---:|:--:|
| SiLU igualada | +0.18 | [+0.06, +0.28] | +0.96 | 0.01953 | 0.09766 | no |
| Rational | +0.08 | [-0.05, +0.18] | +0.38 | 0.26367 | 0.90625 | no |
| Dynamic ReLU-B | -0.01 | [-0.09, +0.07] | -0.07 | 0.86328 | 0.90625 | no |
| Meta-ACON | +0.07 | [-0.03, +0.17] | +0.41 | 0.22656 | 0.90625 | no |
| CCA-Lite | +0.04 | [-0.02, +0.09] | +0.37 | 0.27930 | 0.90625 | no |

## Audio, 6 regímenes  (`audio_digits`)

Familia de Holm: 5 comparaciones (piso corregido 0.0098).

| Método | Exactitud media ± sd | IC95 | ΔP | ΔF | Barajado |
|---|---:|---:|---:|---:|---:|
| SiLU (ancho original) | 73.28 ± 2.30 | [71.88, 74.58] | -18.89% | -18.29% | — |
| SiLU igualada | 73.92 ± 2.40 | [72.52, 75.30] | +1.26% | -0.49% | — |
| Rational | 75.13 ± 2.42 | [73.63, 76.48] | -0.50% | -1.09% | — |
| Dynamic ReLU-B | 73.07 ± 2.37 | [71.67, 74.45] | -0.21% | +0.14% | — |
| Meta-ACON | 73.82 ± 2.50 | [72.28, 75.20] | +0.19% | -0.29% | — |
| CCA-Lite | 74.27 ± 1.74 | [73.30, 75.35] | -3.49% | -4.37% | 67.25 |
| CCA-Hinge | 74.33 ± 1.55 | [73.55, 75.35] | +0.00% | +0.00% | 68.30 |

Ganancia pareada de CCA-Hinge:

| Comparador | Ganancia pp | IC95 | d_z | p exacto | p-Holm | sig. |
|---|---:|---:|---:|---:|---:|:--:|
| SiLU igualada | +0.42 | [-1.17, +1.85] | +0.16 | 0.62305 | 1.00000 | no |
| Rational | -0.80 | [-1.92, +0.32] | -0.42 | 0.21484 | 1.00000 | no |
| Dynamic ReLU-B | +1.27 | [-0.30, +2.92] | +0.46 | 0.21289 | 1.00000 | no |
| Meta-ACON | +0.52 | [-1.02, +2.03] | +0.20 | 0.54688 | 1.00000 | no |
| CCA-Lite | +0.07 | [-0.73, +0.80] | +0.05 | 0.90625 | 1.00000 | no |

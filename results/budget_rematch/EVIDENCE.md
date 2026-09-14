# Re-corrida con presupuesto igualado (filas de la Tabla 2)

Protocolo idéntico al de la Tabla 1: CCA-Hinge define el objetivo de recursos
y una búsqueda discreta ensancha la anfitriona de cada competidor hasta quedar
dentro del 105 % de sus parámetros y del 110 % de sus operaciones por muestra.
La fila *SiLU (ancho original)* se informa sin igualar, como referencia.

## Texto, 8 regímenes  (`text_cls`)

Familia de Holm: 5 comparaciones (piso corregido 0.0098).

| Método | Exactitud media ± sd | IC95 | ΔP | ΔF | Barajado |
|---|---:|---:|---:|---:|---:|
| SiLU (ancho original) | 93.27 ± 1.68 | [92.23, 94.23] | -7.47% | -13.58% | — |
| SiLU igualada | 93.42 ± 1.54 | [92.48, 94.32] | +0.73% | -5.94% | — |
| Rational | 96.28 ± 1.16 | [95.60, 96.98] | +0.81% | -1.10% | — |
| Dynamic ReLU-B | 96.67 ± 0.98 | [96.08, 97.25] | +0.59% | -4.65% | — |
| Meta-ACON | 96.34 ± 1.14 | [95.67, 97.01] | +0.17% | -4.73% | — |
| CCA-Lite | 96.37 ± 0.99 | [95.81, 96.98] | -1.32% | -2.89% | 94.80 |
| CCA-Hinge | 96.42 ± 1.03 | [95.80, 97.03] | +0.00% | +0.00% | 95.47 |

Ganancia pareada de CCA-Hinge:

| Comparador | Ganancia pp | IC95 | d_z | p exacto | p-Holm | sig. |
|---|---:|---:|---:|---:|---:|:--:|
| SiLU igualada | +2.99 | [+2.63, +3.38] | +4.72 | 0.00195 | 0.00977 | sí |
| Rational | +0.13 | [-0.05, +0.34] | +0.39 | 0.28516 | 0.85547 | no |
| Dynamic ReLU-B | -0.25 | [-0.37, -0.13] | -1.25 | 0.00781 | 0.03125 | sí |
| Meta-ACON | +0.08 | [-0.11, +0.25] | +0.26 | 0.42969 | 0.85938 | no |
| CCA-Lite | +0.05 | [-0.12, +0.22] | +0.16 | 0.64844 | 0.85938 | no |

Configuraciones elegidas:

| Método | Configuración |
|---|---|
| SiLU (ancho original) | `{'hidden': 256, 'groups': 16, 'g_hidden': 16}` |
| SiLU igualada | `{'hidden': 268, 'groups': 16, 'g_hidden': 16}` |
| Rational | `{'hidden': 268, 'groups': 16, 'g_hidden': 16}` |
| Dynamic ReLU-B | `{'hidden': 266, 'groups': 8, 'g_hidden': 8}` |
| Meta-ACON | `{'hidden': 266, 'groups': 8, 'g_hidden': 8}` |
| CCA-Lite | `{'hidden': 256, 'groups': 16, 'g_hidden': 16}` |
| CCA-Hinge | `{'hidden': 256, 'groups': 16, 'g_hidden': 16}` |

## Texto, 1 régimen (control)  (`text_cls_single`)

Familia de Holm: 5 comparaciones (piso corregido 0.0098).

| Método | Exactitud media ± sd | IC95 | ΔP | ΔF | Barajado |
|---|---:|---:|---:|---:|---:|
| SiLU (ancho original) | 99.42 ± 0.34 | [99.20, 99.60] | -7.47% | -13.58% | — |
| SiLU igualada | 99.43 ± 0.34 | [99.21, 99.61] | +0.73% | -5.94% | — |
| Rational | 99.56 ± 0.19 | [99.45, 99.67] | +0.81% | -1.10% | — |
| Dynamic ReLU-B | 99.65 ± 0.16 | [99.55, 99.74] | +0.59% | -4.65% | — |
| Meta-ACON | 99.53 ± 0.20 | [99.41, 99.64] | +0.17% | -4.73% | — |
| CCA-Lite | 99.59 ± 0.16 | [99.49, 99.69] | -1.32% | -2.89% | 99.45 |
| CCA-Hinge | 99.55 ± 0.22 | [99.41, 99.68] | +0.00% | +0.00% | 99.37 |

Ganancia pareada de CCA-Hinge:

| Comparador | Ganancia pp | IC95 | d_z | p exacto | p-Holm | sig. |
|---|---:|---:|---:|---:|---:|:--:|
| SiLU igualada | +0.12 | [+0.01, +0.25] | +0.57 | 0.11328 | 0.45312 | no |
| Rational | -0.02 | [-0.07, +0.04] | -0.17 | 0.61328 | 1.00000 | no |
| Dynamic ReLU-B | -0.10 | [-0.20, -0.03] | -0.71 | 0.03125 | 0.15625 | no |
| Meta-ACON | +0.01 | [-0.06, +0.07] | +0.10 | 0.77930 | 1.00000 | no |
| CCA-Lite | -0.05 | [-0.13, +0.03] | -0.34 | 0.31250 | 0.93750 | no |

Configuraciones elegidas:

| Método | Configuración |
|---|---|
| SiLU (ancho original) | `{'hidden': 256, 'groups': 16, 'g_hidden': 16}` |
| SiLU igualada | `{'hidden': 268, 'groups': 16, 'g_hidden': 16}` |
| Rational | `{'hidden': 268, 'groups': 16, 'g_hidden': 16}` |
| Dynamic ReLU-B | `{'hidden': 266, 'groups': 8, 'g_hidden': 8}` |
| Meta-ACON | `{'hidden': 266, 'groups': 8, 'g_hidden': 8}` |
| CCA-Lite | `{'hidden': 256, 'groups': 16, 'g_hidden': 16}` |
| CCA-Hinge | `{'hidden': 256, 'groups': 16, 'g_hidden': 16}` |

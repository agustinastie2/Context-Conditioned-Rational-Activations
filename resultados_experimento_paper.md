# Resultados Experimentales: CCA sin Filtración de Información (v3)

*Generado por `cca_leakfree.py` — semillas [1337, 1338, 1339], dispositivo `cuda`.*

## A. Metodología Experimental

Para evaluar la **Activación Condicionada por Contexto (CCA)** sin ninguna
filtración de información futura, construimos un benchmark de modelado de
lenguaje a nivel de caracteres con **32 regímenes**: el corpus
*Tiny Shakespeare* transformado por 4 transformaciones fijas (ROT13, Atbash,
inversión de mayúsculas, texto invertido) y 27 cifrados de sustitución
aleatorios, además del texto original. El Mini-Transformer causal
(3 capas, $d=128$, 4 cabezas) recibe una
ventana de atención de solo **16 caracteres** — insuficiente para
identificar el cifrado — y la capa CCA se condiciona en un resumen
(media de embeddings proyectada) de los **512 caracteres
estrictamente anteriores** a la ventana. Como todo carácter del contexto
precede a todo token predicho, **no existe canal no causal**. Comparamos:
(i) baseline SiLU; (ii) control con la historia inyectada aditivamente en los
embeddings de entrada; (iii) CCA condicionada en la historia; y (iv) la
ablación con contexto barajado entre ejemplos del batch. Entrenamiento:
3000 iteraciones de AdamW (lr=0.003, batch=64),
3 semillas. Parámetros: B0 0.61M,
B0H 0.62M, CCA 0.71M.

## B. Tabla de Resultados (LaTeX)

```latex
\begin{table}[t]
\centering
\caption{Char-level LM on a 32-regime cipher mixture of Tiny Shakespeare
(window 16, strictly-past history of 512 chars,
3000 iterations, mean $\pm$ std over 3 seeds). The
history precedes every predicted token, so no future information is used.}
\label{tab:cca_leakfree}
\begin{tabular}{lccc}
\toprule
Model & Val. Loss & PPL & Throughput (tok/s) \\
\midrule
Baseline (SiLU) & 3.0873 $\pm$ 0.017 & 21.92 $\pm$ 0.36 & 315330 \\
Baseline + history (additive) & 2.5766 $\pm$ 0.098 & 13.19 $\pm$ 1.31 & 272577 \\
CCA (history-conditioned) & 2.4518 $\pm$ 0.012 & 11.61 $\pm$ 0.13 & 161281 \\
CCA (context shuffled) & 3.6847 $\pm$ 0.013 & 39.84 $\pm$ 0.51 & 147994 \\
\bottomrule
\end{tabular}
\end{table}
```

**Resultado principal:** la CCA condicionada en historia estrictamente pasada
reduce la pérdida de validación en **20.6%** y la perplejidad en
**47.0%** frente al baseline SiLU, superando el objetivo del 15% **sin
filtración de datos**. Frente al control con inyección aditiva de la misma
información (B0H), la CCA conserva una ventaja de **4.8%** en pérdida y
**12.0%** en PPL, lo que indica que el mecanismo de enrutar la no
linealidad es una vía más eficaz de explotar el contexto que la simple
concatenación al embedding.

## C. Análisis de la Varianza de Activación

Varianza por coeficiente de los parámetros racionales generados durante
validación (promedio sobre capas, grupos y semillas):

- `a0`: 1.5839e+00
- `a1`: 1.8331e+00
- `a2`: 1.4982e+00
- `b0`: 0.0000e+00
- `b1`: 0.0000e+00

**Varianza media: 9.8303e-01.** La varianza positiva confirma que el
generador $G(c)$ produce curvas de activación distintas por secuencia: el
modelo adapta $A(x,c)$ al régimen identificado en la historia (las firmas de
frecuencia de cada cifrado), en lugar de colapsar a una no linealidad fija.

## D. Discusión de la Ablación (Shuffle Test)

Al barajar el contexto entre ejemplos del batch, la pérdida sube de
**2.4518** a **3.6847** y la perplejidad de
**11.61** a **39.84**, cayendo muy por debajo incluso
del baseline sin historia. Esto demuestra empíricamente que el modelo utiliza
activamente el contexto correcto para enrutar sus no linealidades: con la
historia de otra secuencia, las curvas seleccionadas corresponden a un cifrado
equivocado y la predicción colapsa. Dado que la historia es estrictamente
pasada, esta ganancia no puede atribuirse a filtración de información futura:
constituye evidencia causal y limpia del mecanismo de condicionamiento.

### Nota sobre el alcance del resultado

Resultados complementarios (ver `sweep_variants.py`) muestran que en dominio
único y con ventana amplia, donde la atención ya provee cómputo condicionado
al contexto y éste no aporta información irrecuperable, la ganancia asintótica
de la CCA es pequeña (~2–6%). La conclusión para el paper es que **la CCA
aporta valor cuando el contexto transporta información relevante que el
backbone no puede obtener por sí mismo** (multi-dominio, memoria más allá de
la ventana de atención), consistente con los resultados del gate sintético y
el caso nulo de CIFAR-10 de fases anteriores de esta investigación.

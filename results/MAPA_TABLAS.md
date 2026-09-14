# De dónde sale cada número del paper (CONAIISI)

Mapa de cada fila de las tablas y de la Figura 1 al archivo que la respalda.
Todos los `summary.json` tienen la misma forma: `tasks.<tarea>.methods` para las
medias y `tasks.<tarea>.cca_hinge_comparisons` para las diferencias pareadas.
El campo `family_size` dice contra cuántos comparadores corrige Holm en esa tarea.

## Tabla 1 — comparación confirmatoria (semillas 3001–3010)

| Fila | Archivo | Clave |
|---|---|---|
| Texto cifrado | `cca_hinge_confirmation/summary.json` | `tasks.text` |
| MNIST + SVHN + USPS | `cca_hinge_confirmation/summary.json` | `tasks.digits` |
| Columnas *t eager*, *t comp.*, *S* | `cca_cost_l4/costs.json` | `text`, `digits` |

## Tabla 2 — matriz completa

| Fila | Semillas | Archivo | Clave |
|---|---|---|---|
| Lenguaje por caracteres (PPL) | 4001–4010 | `paper_textchar_l4/summary.json` | `tasks.text_char` |
| Generación MNIST (bits/dim) | 4101–4110 | `paper_imagegen_short_l4/summary.json` | `tasks.image_gen_mnist` |
| Generación Fashion (bits/dim) | 4101–4110 | `paper_imagegen_short_l4/summary.json` | `tasks.image_gen_fashion` |
| Generación CIFAR-10 (bits/dim) | 4101–4110 | `paper_imagegen_short_l4/summary.json` | `tasks.image_gen_cifar` |
| Imagen, 8 regímenes ‡ | 4001–4010 | `budget_rematch_vision_audio/summary.json` | `tasks.image_cls` |
| Imagen, 1 régimen (control) ‡ | 4001–4010 | `budget_rematch_vision_audio/summary.json` | `tasks.image_cls_single` |
| Texto, 8 regímenes ‡ | 4001–4010 | `budget_rematch/summary.json` | `tasks.text_cls` |
| Texto, 1 régimen (control) ‡ | 4001–4010 | `budget_rematch/summary.json` | `tasks.text_cls_single` |
| Dígitos reales, 3 dominios † | 4001–4010 | `paper_full_comparison/summary.json` | `tasks.digits_multidomain` |
| Razonamiento, 8 claves | 4001–4010 | `paper_full_comparison/summary.json` | `tasks.algo_modular` |
| Audio, 6 regímenes ‡ | 4001–4010 | `budget_rematch_vision_audio/summary.json` | `tasks.audio_digits` |
| Sentimiento, 4 dominios reales | 4001–4010 | `paper_full_comparison/summary.json` | `tasks.text_sentiment` |

‡ Presupuesto igualado: la columna SiLU es la **SiLU igualada** (`silu_budget`).
Las mismas corridas guardan además la SiLU al ancho original bajo la clave
`silu`, que no se informa en la tabla y sirve de referencia sin igualar.

† Misma tarea que la segunda fila de la Tabla 1, con otro bloque de semillas
(4001–4010 en vez de 3001–3010). Por eso 88,96 % aquí y 88,68 % allá: son dos
corridas distintas, ambas correctas. Cuando importe una cifra puntual, la que
corresponde citar es la confirmatoria.

## Figura 1 — ventaja de CCA-Hinge sobre Dynamic ReLU-B

Se genera con `python src/make_figures_conaisi.py --skip-mecanismo`, que lee
`cca_hinge_comparisons.dyrelu` de los mismos archivos de la Tabla 2 (y de
`cca_hinge_confirmation/summary.json` para las dos entradas rotuladas
*confirmatoria*). El marcador relleno corresponde a `p_holm < 0.05`.

## Sección 3.3 — controles

| Control | Archivo | Clave |
|---|---|---|
| Inyección directa (LM carácter) | `paper_textchar_l4/summary.json` | método `b0h` |
| Inyección directa (generación) | `paper_imagegen_short_l4/summary.json` | método `b0h` |
| Inyección directa (razonamiento) | `paper_full_comparison/summary.json` | método `b0h` |
| Barajado de contexto (imagen, dígitos) | `paper_full_comparison/shuffle_summary.json`, `budget_rematch_vision_audio/summary.json` | `mean_shuffled_accuracy` |

## Verificación de integridad

Cada directorio trae `manifest.sha256`. Para comprobar que los resultados no se
tocaron:

```bash
cd results/<directorio> && sha256sum -c manifest.sha256
```

Para regenerarlos después de una corrida nueva:

```bash
python src/regen_manifests.py --check   # informa sin escribir
python src/regen_manifests.py           # reescribe
```

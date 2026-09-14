# Cambios aplicados — paper CONAIISI + repositorio

Todo lo que está en este zip va **encima de la raíz del repo**, respetando las
rutas. No borra nada: sólo pisa `paper_conaisi/main.tex`, `main.pdf`,
`figs/fig_bosque.png`, `src/make_figures_conaisi.py` y los ocho
`manifest.sha256` desactualizados; el resto son archivos nuevos.

```bash
unzip -o cca-correcciones.zip -d /tmp/corr
cp -r /tmp/corr/cca-correcciones/* /ruta/al/repo/
```

`main.tex.diff` es el diff contra la versión anterior, por si querés revisar
cada cambio del paper antes de pisarlo.

---

## 1. Datos nuevos incorporados

`results/budget_rematch_vision_audio/` — las tres tareas que faltaban
(`image_cls`, `image_cls_single`, `audio_digits`), semillas 4001–4010, familia
de Holm de 5. Crucé las 18 celdas contra la Tabla 2: **coinciden las 18**, y
también las tres comparaciones contra Dynamic ReLU-B que cita el texto
(−0,02 / p=0,957; −0,01 / p=0,906; +1,27 / p=1,000) y los valores de barajado
de la sección 3.3 (88,05 → 57,61 y 98,49 → 98,46).

Agregué `EVIDENCE.md` y `manifest.sha256` generados a partir del propio
`summary.json`, con el mismo formato que los demás directorios.

Los otros directorios de corrida traen además `budgets.json`, `checkpoint.json`,
`environment.json` y `runs.csv`. Este no. Queda pendiente si en algún momento
querés que el directorio sea verificable de punta a punta como los demás; si los
sumás, corré después `python src/regen_manifests.py
results/budget_rematch_vision_audio`.

## 2. Tabla 2

Las dos filas de texto pasan a los valores de `results/budget_rematch` y llevan ‡:

| Fila | SiLU igualada | Rat. | DyR | Meta | CCA-Lite | CCA-Hinge |
|---|---|---|---|---|---|---|
| Texto, 8 regímenes ‡ | 93,42±1,54 | 96,28±1,16 | **96,67±,98** | 96,34±1,14 | 96,37±,99 | 96,42±1,03 |
| Texto, 1 régimen (control) ‡ | 99,43±,34 | 99,56±,19 | **99,65±,16** | 99,53±,20 | 99,59±,16 | 99,55±,22 |

Las filas de imagen y audio quedan como estaban: ahora sí tienen respaldo.

**Epígrafe corregido.** Decía que en las filas ‡ «ningún método se desvía más
del 1 % de los parámetros de CCA-Hinge», y es falso: CCA-Lite queda −4,12 % en
imagen y −3,49 % en audio, y la SiLU igualada de audio queda +1,26 %. Ahora dice
que ningún competidor *excede* en más del 1,3 % y que sólo CCA-Lite se aparta
por debajo, hasta −4,12 %. Es el mismo matiz que ya trae la Tabla 1.

## 3. Figura 1

Regenerada con `python src/make_figures_conaisi.py --skip-mecanismo`. Las cinco
filas con presupuesto igualado ahora leen de los directorios de re-corrida; el
script aprende a leer el esquema de `budget_rematch*`, que nombra las claves
`mean_gain_points` / `ci95_gain_points` en vez de `mean_gain_accuracy_points` /
`ci95_gain_accuracy_points`.

Único cambio visible: «Texto, 8 regímenes» pasa de −0,18 hueco a **−0,25 relleno**.

## 4. Conclusiones corregidas

El cambio de fondo está en texto multirrégimen. Con presupuesto igualado la
diferencia es **−0,25 pp, IC [−0,37; −0,13], dz = −1,25, p_Holm = 0,0312**: ya no
es «negativa y no significativa», es significativa a favor de Dynamic ReLU-B.

- **Sección 3.2**: reescrito el párrafo. Declara el signo, relativiza la
  magnitud (un cuarto de punto sobre 96 %) y la ancla en la confirmatoria de
  texto cifrado, donde ambos quedan indistinguibles (−0,12, p=0,773). Lectura
  final: en texto CCA-Hinge no supera a Dynamic ReLU-B.
- **Sección 3.2**, controles: el rango de texto pasa de 99,42–99,64 a 99,43–99,65.
- **Sección 3.2 y limitaciones**: «las tres tareas marcadas con ‡» → «las cinco»,
  y se explicita que en las cinco la familia de Holm es de cinco comparaciones
  (el punto 2 del profesor).
- **Conclusión**: «En texto queda estadísticamente empatada con los métodos
  adaptativos» → empatada en la confirmatoria y un cuarto de punto por debajo de
  Dynamic ReLU-B en la matriz con presupuesto igualado.

Lo demás de la conclusión se sostiene sin tocar: audio detrás de la racional
estática (75,13 vs 74,33) y sentimiento sin mejorar a la SiLU fija.

## 5. Manifiestos

`src/regen_manifests.py` — regenera los `manifest.sha256` conservando el orden y
la lista de nombres original de cada directorio. Por omisión sólo toca
directorios que ya tenían manifiesto; `--check` informa sin escribir.

Regenerados los 8 que fallaban. **Los 13 directorios verifican limpio**:

```bash
for d in results/*/; do (cd "$d" && sha256sum -c manifest.sha256 2>/dev/null); done
```

## 6. Mapa de trazabilidad

`results/MAPA_TABLAS.md` — de qué archivo y qué clave sale cada fila de cada
tabla, de la Figura 1 y de los controles de la sección 3.3, más la explicación
de 88,68 % vs 88,96 % y cómo verificar los manifiestos. Es lo que necesita
alguien de CONAIISI que abra el repo a chequear un número.

## 7. Lo que falta hacer, y es lo importante

Todo esto queda **en tu copia local pero no en el repositorio público**. `git
status` muestra sin commitear los resultados que el profesor no encontró:

```bash
git add results/paper_textchar_l4/ results/paper_imagegen_short_l4/ \
        results/budget_rematch/ results/budget_rematch_vision_audio/ \
        results/cca_cost_l4/ results/paper_full_comparison/shuffle_summary.json \
        results/MAPA_TABLAS.md results/*/manifest.sha256 \
        src/regen_manifests.py src/make_figures_conaisi.py \
        src/measure_costs.py src/run_parallel.py src/export_shuffle_summary.py \
        paper_conaisi/
git commit -m "Resultados completos de generación y presupuesto igualado; manifiestos al día"
```

Antes de eso, dos cosas menores: `.inductor_cache/` y `CONAIISI2026_Template/`
aparecen como no seguidos y probablemente no quieras versionarlos (el primero es
caché de `torch.compile`); conviene sumarlos al `.gitignore`.

## 8. Notas sueltas

- **Separador decimal**: no hay problema en esta versión. `babel[spanish]`
  convierte el punto a coma en modo matemático y el `.tex` usa punto de forma
  uniforme. Lo que vio el profesor era el PDF de CACIC.
- **Dos redondeos al empate exacto**: `text_cls`/Rational da 96,285 y
  `text_cls_single`/Meta-ACON da 99,535. La tabla publica 96,28 y 99,53, que es
  lo que emite el propio `results/budget_rematch/tabla2_filas.tex`. Los dejé así
  para que coincidan con el artefacto del repo; no son errores.
- **SiLU al ancho original en audio**: la re-corrida da 73,28 % y
  `paper_full_comparison` daba 72,77 % para la misma configuración. Son ~3
  muestras sobre 600 de FSDD. No afecta ninguna cifra publicada (esa fila no se
  informa), pero si querés explicarlo, ahí está.
- El PDF compila limpio: 9 páginas, sin referencias ni citas sin resolver.
  Necesitás `texlive-lang-spanish` instalado.

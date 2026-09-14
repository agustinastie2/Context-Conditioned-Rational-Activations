"""
make_figures_conaisi.py — figuras monocromas para la version CoNaIISI del paper.
===============================================================================

`make_figures.py` dibuja las corridas originales de tres semillas de la variante
previa a CCA-Hinge y usa color. Este script es independiente: lee unicamente los
resultados vigentes (semillas 3001-3010 y 4001-4010/4101-4110) y produce tres
figuras en escala de grises, legibles en impresion monocroma como pide el
formato del congreso.

  paper_conaisi/figs/fig_mecanismo.png   curvas generadas + varianza de coefs.
  paper_conaisi/figs/fig_bosque.png      ganancia sobre Dynamic ReLU-B por tarea
  paper_conaisi/figs/fig_costo.png       costo por iteracion, eager vs compilado

Uso:  python src/make_figures_conaisi.py [--skip-mecanismo]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "paper_conaisi" / "figs"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 10,
    "font.family": "serif", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linestyle": ":", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "0.2", "text.color": "0.1", "axes.labelcolor": "0.1",
    "xtick.color": "0.2", "ytick.color": "0.2", "legend.frameon": False,
})


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
#  Figura 1 — mecanismo: la curva se deforma con el contexto
# --------------------------------------------------------------------------- #
def fig_mecanismo(steps: int = 1200, n_contexts: int = 5) -> None:
    """Entrena un modelo CCA-Hinge sobre la tarea de texto cifrado y grafica
    (a) la curva generada para varios regimenes y (b) cuanto varia cada
    coeficiente racional entre muestras."""
    os.environ["CCA_VARIANT"] = "hinge"           # leido por cca_factory
    os.environ["STEPS"] = str(steps)              # leido por text_cls al importar
    sys.path.insert(0, str(SRC))

    import torch
    import torch.nn.functional as F
    import text_cls as tc
    from cca_core import curve_at, set_cca

    sources = tc.make_sources(tc.N_CLASSES, tc.SEED)
    ciphers = tc.make_ciphers(8, tc.SEED + 1)
    Xtr, ytr = tc.gen_dataset(tc.N_TRAIN, sources, ciphers, tc.SEED)
    Xte, yte = tc.gen_dataset(tc.N_TEST, sources, ciphers, tc.SEED + 99)

    print(f"[mecanismo] entrenando CCA-Hinge ({steps} pasos) en {tc.DEVICE}", flush=True)
    model, train_seconds, n_par = tc.train("cca", (Xtr, ytr, Xte, yte))
    evaluation = tc.evaluate(model, Xte, yte, collect=True)
    coeff_var = evaluation["coeff_var"].numpy()
    print(f"[mecanismo] acc={evaluation['acc']:.4f} params={n_par} "
          f"t={train_seconds:.1f}s var={np.round(coeff_var, 5).tolist()}", flush=True)

    model.eval()
    set_cca(model, collect=False, shuffle_ctx=False)
    device = next(model.parameters()).device
    contexts = []
    for regime in range(n_contexts):
        Xr, _ = tc.gen_dataset(512, sources, [ciphers[regime]], tc.SEED + 7)
        with torch.no_grad():
            contexts.append(model.emb(Xr.to(device)).mean((0, 1)))

    grid = torch.linspace(-4, 4, 400)
    curves = curve_at(model, grid, contexts)

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(6.9, 2.15))

    styles = [(0.0, "-"), (0.3, "--"), (0.45, "-."), (0.58, (0, (3, 1, 1, 1))),
              (0.7, (0, (5, 2)))]
    left.plot(grid, F.silu(grid), color="0.55", lw=2.6,
              label="SiLU (activación fija)")
    for i, curve in enumerate(curves):
        shade, dash = styles[i % len(styles)]
        left.plot(grid, curve, color=str(shade), ls=dash, lw=1.0,
                  label=f"Régimen {i + 1}")
    left.set_xlabel("Preactivación $x$")
    left.set_ylabel("$A(x,c)$")
    left.set_title("(a) Curva aprendida", fontsize=10.5)
    left.legend(fontsize=7.2, loc="center", bbox_to_anchor=(0.52, 0.66),
                handlelength=2.0, labelspacing=0.2, borderpad=0.2,
                frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)

    stack = np.stack([curve.numpy() for curve in curves])
    deviation = stack - stack.mean(axis=0, keepdims=True)
    for i, row in enumerate(deviation):
        shade, dash = styles[i % len(styles)]
        middle.plot(grid, row, color=str(shade), ls=dash, lw=1.0,
                    label=f"Régimen {i + 1}")
    middle.axhline(0, color="0.6", lw=0.8)
    middle.set_xlabel("Preactivación $x$")
    middle.set_ylabel("$A(x,c)-\overline{A}(x)$")
    middle.set_title("(b) Diferencia por régimen", fontsize=10.5)

    names = ["$a_0$", "$a_1$", "$a_2$", "$b_0$", "$b_1$"]
    hatches = ["//", "\\\\", "xx", "..", "++"]
    positions = np.arange(len(names))
    for pos, value, hatch in zip(positions, coeff_var, hatches):
        right.bar(pos, value, 0.62, facecolor="0.85", edgecolor="0.15",
                  hatch=hatch, linewidth=0.7)
    right.set_xticks(positions, names)
    right.set_ylabel("Varianza entre muestras")
    right.set_title("(c) Coeficientes", fontsize=10.5)
    right.margins(y=0.18)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_mecanismo.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[mecanismo] escrito {OUT / 'fig_mecanismo.png'}")


# --------------------------------------------------------------------------- #
#  Figura 2 — bosque: donde ayuda y donde no, contra el rival mas fuerte
# --------------------------------------------------------------------------- #
# (etiqueta, archivo, tarea, escala) con escala: "pp" ya en puntos porcentuales,
# "frac" fraccion que se multiplica por 100, "rel" mejora relativa sobre la media
# del comparador (metricas de generacion, menores-mejor).
ACCURACY_TASKS = [
    ("Razonamiento, 8 claves", "results/paper_full_comparison/summary.json", "algo_modular", "frac"),
    ("Dígitos reales, 3 dominios", "results/paper_full_comparison/summary.json", "digits_multidomain", "frac"),
    ("Dígitos reales (confirmatoria)", "results/cca_hinge_confirmation/summary.json", "digits", "pp"),
    ("Audio, 6 regímenes", "results/budget_rematch_vision_audio/summary.json", "audio_digits", "pp_rematch"),
    ("Imagen, 8 regímenes", "results/budget_rematch_vision_audio/summary.json", "image_cls", "pp_rematch"),
    ("Imagen, 1 régimen", "results/budget_rematch_vision_audio/summary.json", "image_cls_single", "pp_rematch"),
    ("Texto, 8 regímenes", "results/budget_rematch/summary.json", "text_cls", "pp_rematch"),
    ("Texto, 1 régimen", "results/budget_rematch/summary.json", "text_cls_single", "pp_rematch"),
    ("Texto cifrado (confirmatoria)", "results/cca_hinge_confirmation/summary.json", "text", "pp"),
    ("Sentimiento, 4 dominios", "results/paper_full_comparison/summary.json", "text_sentiment", "frac"),
]
GENERATION_TASKS = [
    ("Gen. Fashion (bpd)", "results/paper_imagegen_short_l4/summary.json", "image_gen_fashion", "rel"),
    ("Gen. CIFAR-10 (bpd)", "results/paper_imagegen_short_l4/summary.json", "image_gen_cifar", "rel"),
    ("LM carácter (PPL)", "results/paper_textchar_l4/summary.json", "text_char", "rel"),
    ("Gen. MNIST (bpd)", "results/paper_imagegen_short_l4/summary.json", "image_gen_mnist", "rel"),
]


def gain_vs_dyrelu(relative: str, task: str, scale: str):
    """Devuelve (ganancia, limite inferior, limite superior, p de Holm)."""
    block = load(relative)["tasks"][task]
    comparison = block["cca_hinge_comparisons"]["dyrelu"]
    if scale == "pp_rematch":
        # results/budget_rematch nombra distinto las mismas cantidades
        gain = comparison["mean_gain_points"]
        low, high = comparison["ci95_gain_points"]
    elif scale == "pp":
        gain = comparison["mean_gain_accuracy_points"]
        low, high = comparison["ci95_gain_accuracy_points"]
    else:
        gain, (low, high) = comparison["mean_gain"], comparison["ci95_gain"]
        factor = 100.0 if scale == "frac" else 100.0 / block["methods"]["dyrelu"]["mean"]
        gain, low, high = gain * factor, low * factor, high * factor
    return gain, low, high, comparison["p_holm"]


def _forest_panel(axis, rows, title, xlabel, ticks=None):
    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows):
        _, gain, low, high, p_holm = row
        significant = p_holm < 0.05
        axis.errorbar(gain, position, xerr=[[gain - low], [high - gain]],
                      fmt="o", ms=4.2, lw=1.0, capsize=2.4, color="0.1",
                      markerfacecolor="0.1" if significant else "white",
                      markeredgecolor="0.1", markeredgewidth=0.9)
    axis.axvline(0, color="0.35", lw=0.9, ls="--")
    axis.set_yticks(positions, [row[0] for row in rows], fontsize=9)
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_title(title, fontsize=10.5)
    axis.margins(y=0.08)
    if ticks is not None:
        # Escala simetrica logaritmica: una unica ventaja de +20 pp aplastaria
        # el resto del panel en escala lineal.
        axis.set_xscale("symlog", linthresh=1.0, linscale=0.9)
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{t:g}" for t in ticks], fontsize=9)
        axis.set_xlim(min(ticks) * 1.05, max(ticks) * 1.4)


def fig_bosque() -> None:
    accuracy = [(label, *gain_vs_dyrelu(path, task, scale))
                for label, path, task, scale in ACCURACY_TASKS]
    generation = [(label, *gain_vs_dyrelu(path, task, scale))
                  for label, path, task, scale in GENERATION_TASKS]
    accuracy.sort(key=lambda row: row[1], reverse=True)
    generation.sort(key=lambda row: row[1], reverse=True)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(6.9, 2.9), gridspec_kw={"width_ratios": [1.25, 1.0]})
    _forest_panel(left, accuracy, "(a) Tareas de clasificación",
                  "Ventaja de CCA-Hinge (puntos porcentuales)",
                  ticks=[-1, 0, 1, 2, 5, 20])
    _forest_panel(right, generation, "(b) Tareas generativas",
                  "Reducción de la métrica (%)")
    fig.tight_layout(pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_bosque.png", bbox_inches="tight")
    plt.close(fig)
    for label, gain, low, high, p_holm in accuracy + generation:
        print(f"[bosque] {label:32s} {gain:+7.3f} [{low:+.3f},{high:+.3f}] p={p_holm:.4f}")
    print(f"[bosque] escrito {OUT / 'fig_bosque.png'}")


# --------------------------------------------------------------------------- #
#  Figura 3 — costo: el sobrecosto es de lanzamiento, no aritmetico
# --------------------------------------------------------------------------- #
COST_LABELS = {"silu_budget": "SiLU igualada", "rational": "Rational",
               "dyrelu": "Dynamic ReLU-B", "meta_acon": "Meta-ACON",
               "cca_lite": "CCA-Lite", "cca_hinge": "CCA-Hinge"}
COST_TASKS = [("text", "(a) Texto cifrado"), ("digits", "(b) Dígitos reales")]


def fig_costo() -> None:
    costs = load("results/cca_cost_l4/costs.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), sharey=True)
    methods = list(COST_LABELS)
    positions = np.arange(len(methods))
    for axis, (task, title) in zip(axes, COST_TASKS):
        block = costs[task]
        eager = [block[m]["eager_forward_backward_ms"] for m in methods]
        compiled = [block[m]["compiled_forward_backward_ms"] for m in methods]
        axis.bar(positions - 0.2, eager, 0.38, facecolor="white",
                 edgecolor="0.1", hatch="///", linewidth=0.8, label="eager")
        axis.bar(positions + 0.2, compiled, 0.38, facecolor="0.45",
                 edgecolor="0.1", linewidth=0.8, label="compilado")
        axis.set_xticks(positions, [COST_LABELS[m] for m in methods],
                        rotation=30, ha="right", fontsize=8.5)
        axis.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("ms por iteración\n(propagación y gradiente)", fontsize=9)
    axes[0].legend(fontsize=9, loc="upper left")
    fig.tight_layout(pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_costo.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[costo] escrito {OUT / 'fig_costo.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mecanismo", action="store_true",
                        help="No reentrena el modelo de la Figura 1.")
    parser.add_argument("--steps", type=int, default=1200)
    arguments = parser.parse_args()

    fig_bosque()
    fig_costo()
    if not arguments.skip_mecanismo:
        fig_mecanismo(steps=arguments.steps)


if __name__ == "__main__":
    main()

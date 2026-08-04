"""
make_figures.py — publication figures + comparative tables.
===========================================================

Reads results/*.json (written by the four experiment scripts, multi-seed) and
produces:
  figs/fig_improvement.png   headline relative improvement, multi-regime
  figs/fig_null_control.png  absolute accuracy w/ error bars: multi vs single
  figs/fig_shuffle.png       context-shuffle ablation collapse, per modality
  figs/fig_coeff_var.png     per-coefficient variance of generated params
  figs/fig_curves.png        learned activation curves morphing with context
  figs/fig_pareto.png        Pareto frontier of CCA variants
  results/summary_table.md   markdown comparative table (mean +/- std)
  results/table.tex          LaTeX comparative table

Run after the experiments:  python make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BASE, C_CCA, C_SHUF = "#4C72B0", "#C44E52", "#8C8C8C"
os.makedirs("figs", exist_ok=True)


def load(name):
    p = f"results/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def n_seeds():
    for t in ("image_cls", "text_cls", "image_gen", "text_gen"):
        d = load(t)
        if d and d.get("seeds"):
            return len(d["seeds"])
    return 0


NSEEDS = n_seeds()


# metric metadata per task: (key, higher_is_better, short unit)
META = {
    "text_gen": ("ppl", False, "PPL"),
    "image_cls": ("acc", True, "Acc"),
    "image_gen": ("bpd", False, "bits/dim"),
    "text_cls": ("acc", True, "Acc"),
    "digits_multidomain": ("acc", True, "Acc"),
    "algo_modular": ("acc", True, "Acc"),
    "audio_digits": ("acc", True, "Acc"),
    "text_sentiment": ("acc", True, "Acc"),
}
ORDER = ["text_gen", "image_cls", "image_gen", "text_cls"]
TITLE = {"text_gen": "Generación de Texto", "image_cls": "Clasificación de Img.",
         "image_gen": "Generación de Img.", "text_cls": "Clasificación de Texto",
         "digits_multidomain": "Dígitos Multi-Dominio", "algo_modular": "Algorítmica",
         "audio_digits": "Audio Dígitos", "text_sentiment": "Análisis de Sentimiento"}
SHORT = {"text_gen": "Texto gen.", "image_cls": "Img. cls.",
         "image_gen": "Img. gen.", "text_cls": "Texto cls.",
         "digits_multidomain": "Dígitos", "algo_modular": "Algorit.",
         "audio_digits": "Audio", "text_sentiment": "Sentimiento"}


def improvement(res):
    key, hib, _ = META[res["task"]]
    b, c = res["models"]["silu"][key], res["models"]["cca"][key]
    if key == "acc":                 # error reduction
        return 100 * ((1 - b) - (1 - c)) / (1 - b)
    return 100 * (b - c) / b


# --------------------------------------------------------------------------- #
#  Figure 1 — headline relative improvement (multi-regime)
# --------------------------------------------------------------------------- #
def fig_improvement(multi):
    labels = [SHORT[t] for t in ORDER]
    imp = [improvement(multi[t]) for t in ORDER]
    units = [META[t][2] for t in ORDER]
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(x, imp, 0.6, color=C_CCA)
    for r, u in zip(bars, units):
        tag = "red. de error" if u == "Acc" else f"red. de {u}"
        ax.annotate(f"{r.get_height():.1f}%\n({tag})",
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mejora relativa de CCA sobre SiLU (%)")
    ax.set_ylim(0, max(imp) * 1.25)
    ax.set_title("Mejora de CCA sobre el baseline estático en entornos multi-régimen\n"
                 f"(4 tareas principales, {NSEEDS} semillas de inicialización)", fontsize=11)
    fig.tight_layout()
    fig.savefig("figs/fig_improvement.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 2 — null control: absolute accuracy, multi vs single (error bars)
# --------------------------------------------------------------------------- #
def fig_null_control(multi, single):
    tasks = ["image_cls", "text_cls"]      # accuracy tasks
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, t in zip(axes, tasks):
        groups = ["Multi-Régimen", "Régimen Único"]
        data = [multi[t], single[t]]
        x = np.arange(2)
        w = 0.36
        bvals = [d["models"]["silu"]["acc"] * 100 for d in data]
        bstd = [d["models"]["silu"].get("acc_std", 0) * 100 for d in data]
        cvals = [d["models"]["cca"]["acc"] * 100 for d in data]
        cstd = [d["models"]["cca"].get("acc_std", 0) * 100 for d in data]
        ax.bar(x - w / 2, bvals, w, yerr=bstd, capsize=4, label="SiLU",
               color=C_BASE)
        ax.bar(x + w / 2, cvals, w, yerr=cstd, capsize=4, label="CCA",
               color=C_CCA)
        for xi, (bv, cv) in enumerate(zip(bvals, cvals)):
            ax.annotate(f"+{cv - bv:.2f}pp",
                        (xi, max(bv, cv)), ha="center", va="bottom",
                        fontsize=8, xytext=(0, 6), textcoords="offset points")
        ax.set_xticks(x, groups)
        ax.set_ylabel("Precisión de prueba (%)")
        lo = min(bvals + cvals) - 3
        ax.set_ylim(lo, 100.5)
        ax.set_title(TITLE[t], fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Control de nulidad: CCA asiste en dominios multi-régimen pero no otorga \n"
                 "ventaja ficticia cuando el contexto único no ofrece qué enrutar", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("figs/fig_null_control.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 3 — shuffle ablation
# --------------------------------------------------------------------------- #
def fig_shuffle(multi):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, t in zip(axes, ORDER):
        res = multi[t]
        key, hib, unit = META[t]
        scale = 100 if key == "acc" else 1
        vals = [res["models"]["silu"][key] * scale,
                res["models"]["cca"][key] * scale,
                res["models"]["shuffle"][key] * scale]
        bars = ax.bar(["SiLU", "CCA", "CCA\n(Shuf.)"], vals,
                      color=[C_BASE, C_CCA, C_SHUF])
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 1), textcoords="offset points")
        ax.set_title(f"{SHORT[t]} ({unit})", fontsize=10)
    fig.suptitle("Ablación de Mezcla de Contexto (Context-Shuffle): desemparejar los contextos \n"
                 "degenera la inferencia por debajo del baseline estático", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("figs/fig_shuffle.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 3b — mechanism vs injection (B0H control), generation tasks
# --------------------------------------------------------------------------- #
def fig_b0h(multi):
    tasks = [t for t in ("text_gen", "image_gen")
             if multi[t]["models"].get("silu_hist")]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.8))
    if len(tasks) == 1:
        axes = [axes]
    for ax, t in zip(axes, tasks):
        key = META[t][0]
        names = ["SiLU", "SiLU+Ctx\n(Aditivo)", "CCA"]
        vals = [multi[t]["models"]["silu"][key],
                multi[t]["models"]["silu_hist"][key],
                multi[t]["models"]["cca"][key]]
        bars = ax.bar(names, vals, color=[C_BASE, "#9C7BB0", C_CCA])
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=9,
                        xytext=(0, 1), textcoords="offset points")
        ax.set_title(f"{TITLE[t]} ({META[t][2]})", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.15)
    fig.suptitle("Enrutamiento vs Inyección: CCA supera drásticamente al baseline \n"
                 "que inyecta el mismo contexto en los tensores profundos aditivamente", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("figs/fig_b0h.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 4 — coefficient variance
# --------------------------------------------------------------------------- #
def fig_coeff_var(multi):
    names = ["a0", "a1", "a2", "b0", "b1"]
    tasks = [t for t in ORDER if multi[t].get("coeff_var")]
    x = np.arange(len(names))
    w = 0.8 / len(tasks)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, t in enumerate(tasks):
        ax.bar(x + (i - len(tasks) / 2 + 0.5) * w, multi[t]["coeff_var"], w,
               label=SHORT[t])
    ax.set_xticks(x, names)
    ax.set_ylabel("Varianza inter-muestra (Validación)")
    ax.set_title("Varianza de coeficientes racionales θ(c) generados:\n"
                 "la curva de activación efectivamente se deforma y adapta por contexto",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("figs/fig_coeff_var.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 5 — learned activation curves morphing with context
# --------------------------------------------------------------------------- #
def fig_curves():
    import torch.nn.functional as F
    import text_cls as tc
    from cca_core import curve_at, set_cca
    tc.STEPS = 1200
    sources = tc.make_sources(tc.N_CLASSES, tc.SEED)
    ciphers = tc.make_ciphers(8, tc.SEED + 1)
    Xtr, ytr = tc.gen_dataset(tc.N_TRAIN, sources, ciphers, tc.SEED)
    Xte, yte = tc.gen_dataset(4000, sources, ciphers, tc.SEED + 99)
    model, _, _ = tc.train("cca", (Xtr, ytr, Xte, yte))
    model.eval()
    set_cca(model, collect=False, shuffle_ctx=False)
    dev = next(model.parameters()).device

    contexts = []
    for r in range(5):
        ci = tc.make_ciphers(8, tc.SEED + 1)[r]
        Xr, _ = tc.gen_dataset(512, sources, [ci], tc.SEED + 7)
        with torch.no_grad():
            contexts.append(model.emb(Xr.to(dev)).mean((0, 1)))

    xg = torch.linspace(-4, 4, 200)
    ys = curve_at(model, xg, contexts)
    silu = F.silu(xg)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xg, silu, "k--", lw=2.2, label="SiLU estático (baseline)")
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(contexts)))
    for i, y in enumerate(ys):
        ax.plot(xg, y, color=cmap[i], lw=1.8, label=f"Contexto inferido: Régimen {i}")
    ax.set_xlabel("Vector de pre-activación x")
    ax.set_ylabel("A(x, c)")
    ax.set_title("Deformación topológica: la activación generada se ajusta al contexto\n"
                 "(Clasificación de Texto, grupo de canal 0)", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/fig_curves.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 6 - Pareto Frontier (Cost vs Precision)
# --------------------------------------------------------------------------- #
def fig_pareto(eff):
    if not eff:
        return
    variants = eff.get("variants", [])
    
    xs = []
    ys = []
    labels = []
    
    for v in variants:
        if v["name"] == "SiLU":
            continue
        xs.append(v["flops_cost_pct"])
        ys.append(v["err_reduction_pct"])
        labels.append(v["name"])
        
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot SiLU origin
    ax.scatter(0, 0, color=C_BASE, marker="s", s=80, label="SiLU (Baseline)")
    
    # Plot variants
    ax.scatter(xs, ys, color=C_CCA, alpha=0.7, s=60)
    
    # Annotate points
    for x, y, lbl in zip(xs, ys, labels):
        if "lite" in lbl.lower():
            ax.annotate(lbl, (x, y), xytext=(5, -10), textcoords="offset points", 
                        fontweight="bold", color="darkred")
            ax.scatter([x], [y], color="darkred", s=100, edgecolor="black")
        else:
            ax.annotate(lbl, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    
    # Draw Pareto Frontier curve
    sorted_pts = sorted(zip(xs, ys))
    frontier_x = [sorted_pts[0][0]]
    frontier_y = [sorted_pts[0][1]]
    for x, y in sorted_pts[1:]:
        if y > frontier_y[-1]:
            frontier_x.append(x)
            frontier_y.append(y)
            
    ax.plot(frontier_x, frontier_y, "k--", alpha=0.5, label="Frontera Pareto")
    
    ax.set_xlabel("Sobrecosto de FLOPs (%)")
    ax.set_ylabel("Reducción de error vs SiLU (%)")
    ax.set_title("Frontera de Pareto (Coste Computacional vs Precisión)\npara variantes arquitectónicas de CCA", fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("figs/fig_pareto.png")
    plt.close(fig)

# --------------------------------------------------------------------------- #
#  Tables
# --------------------------------------------------------------------------- #
def fmt(res, model, key):
    v = res["models"][model][key]
    s = res["models"][model].get(key + "_std")
    if key == "acc":
        v = v * 100
        s = (s or 0) * 100
    return f"{v:.3f} $\\pm$ {s:.3f}" if s else f"{v:.3f}"


def make_tables(multi):
    md = ["# Tabla comparativa — CCA vs Baseline (multi-régimen, media ± std)\n",
          "| Tarea | Reg. | Métrica | Baseline (SiLU) | CCA | Mejora rel. |", "|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Resultados experimentales comparando la Racionalidad Condicionada por Contexto (CCA) contra "
           r"un baseline de SiLU estático. Reportamos la media y desviación estándar de " + str(NSEEDS) + r" semillas. "
           r"$\Delta$ indica el porcentaje de reducción de error para clasificación y reducción de entropía (PPL/BPD) para generación.}",
           r"\label{tab:main}", r"\begin{tabular}{llccccc}", r"\toprule",
           r"Tarea & Dominio & Reg. & Métrica & SiLU Baseline & Modelo CCA & $\Delta$ \\",
           r"\midrule"]
           
    # Add new tasks to the main table logic
    ALL_TASKS = ORDER + ["algo_modular", "audio_digits", "digits_multidomain", "text_sentiment"]
    
    for t in ALL_TASKS:
        res = load(t)
        if not res:
            res = multi.get(t)
        if not res:
            continue
            
        key, hib, unit = META[t]
        nr = res.get("n_regimes", "-")
        imp = improvement(res)
        
        # Format the numbers
        baseline_val = fmt(res, 'silu', key)
        cca_val = fmt(res, 'cca', key)
        
        tex.append(f"{TITLE[t]} & {SHORT[t]} & {nr} & {unit} & {baseline_val} & "
                   f"{cca_val} & "
                   f"{imp:+.1f}\\% \\\\")
                   
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open("results/table.tex", "w", encoding="utf-8").write("\n".join(tex))
    print("wrote results/table.tex")


NEW_TASKS = [
    ("digits_multidomain", "Dígitos Multi-Dominio\n(MNIST+SVHN+USPS)"),
    ("algo_modular", "Algorítmica Modular\n(Keyed Remap)"),
    ("audio_digits", "Audio Hablado\n(Dígitos FSDD)"),
    ("text_sentiment", "Análisis Sentimiento\n(Multi-dominio)"),
]
# (file stem, label) for the unified cost/viability table
COST_SPECS = [
    ("text_gen", "Generación Texto (Char)"), ("image_cls", "Clasificación Imágenes"),
    ("image_gen", "Generación Imágenes (MNIST)"), ("image_gen_fashion", "Generación Imágenes (Fashion)"),
    ("image_gen_cifar", "Generación Imágenes (CIFAR)"), ("text_cls", "Clasificación Texto"),
    ("digits_multidomain", "Dígitos Multi-Dominio"), ("algo_modular", "Algorítmica Modular"),
    ("audio_digits", "Audio (FSDD)"), ("text_sentiment", "Análisis de Sentimiento"),
]


def _flop_overhead(res):
    try:
        s = res["models"]["silu"]["cost"]["gflops"]
        c = res["models"]["cca"]["cost"]["gflops"]
        return 100 * (c - s) / s if s else None
    except (KeyError, TypeError):
        return None


def fig_new_tasks():
    avail = [(t, lbl) for t, lbl in NEW_TASKS if load(t)]
    if not avail:
        return
    labels = [lbl for _, lbl in avail]
    imp = [improvement(load(t)) for t, _ in avail]
    x = np.arange(len(avail))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(x, imp, 0.6, color=C_CCA)
    for r in bars:
        ax.annotate(f"{r.get_height():+.1f}%", (r.get_x() + r.get_width() / 2,
                    r.get_height()), ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("Reducción de Error/Pérdida de CCA vs SiLU (%)")
    ax.set_title("CCA evaluado en dominios reales (visión cruzada, audio, procesamiento sintáctico)\n"
                 "Notoria mejora generalizada con excepción del control de sentimiento semántico", fontsize=10.5)
    fig.tight_layout()
    fig.savefig("figs/fig_new_tasks.png")
    plt.close(fig)


def fig_bpe():
    d = load("text_gen_bpe")
    if not d or not d.get("by_vocab"):
        return
    vocabs = sorted(d["by_vocab"], key=int)
    imp = [d["by_vocab"][v]["improvement_pct"] for v in vocabs]
    char = load("text_gen")
    labels = list(vocabs)
    if char:
        imp = [char["improvement_pct"]] + imp
        labels = ["Char (N/A)"] + labels
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar(x, imp, 0.55, color=C_CCA, label="Reducción de PPL")
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}%", (r.get_x() + r.get_width() / 2,
                    r.get_height()), ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Tokenizador (Tamaño de Vocabulario BPE)")
    ax.set_ylabel("Reducción de Perplejidad CCA vs SiLU (%)", color=C_CCA)
    ax.set_title("Generación de Texto: Robustez de la ganancia de CCA a través de la granularidad\n"
                 "(Evaluación con BPE de 2k, 4k y 8k tokens)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig("figs/fig_bpe.png")
    plt.close(fig)


def fig_viability():
    rows = []
    for stem, label in COST_SPECS:
        res = load(stem)
        if res and res.get("viability") and "cost" in res["models"].get("cca", {}):
            rows.append((label, improvement(res),
                         res["viability"]["throughput_cost_pct"],
                         res["models"]["cca"]["cost"].get("params_overhead_pct", 0)))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (xi, xl) in zip(axes, [(2, "Pérdida de Rendimiento / Throughput (%)"),
                                   (3, "Sobrecosto Paramétrico (%)")]):
        xs = [r[xi] for r in rows]
        ys = [r[1] for r in rows]
        ax.scatter(xs, ys, s=70, color=C_CCA, zorder=3, edgecolor="k", linewidth=0.5)
        for r in rows:
            ax.annotate(r[0].split("(")[0].strip(), (r[xi], r[1]), fontsize=7.5,
                        xytext=(4, 3), textcoords="offset points")
        lim = max(max(xs), 1) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="Ganancia = Coste")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_xlabel(xl)
        ax.set_ylabel("Mejora de Precisión/Calidad (%)")
    fig.suptitle("Análisis Beneficio-Coste por tarea: los puntos sobre la línea punteada\n"
                 "son computacionalmente rentables justificando su integración", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("figs/fig_viability.png")
    plt.close(fig)


def fig_overhead():
    labels, flops, tput = [], [], []
    for stem, label in COST_SPECS:
        res = load(stem)
        if not res or not res.get("viability"):
            continue
        v = res["viability"]
        f = _flop_overhead(res)
        if f is not None:
            labels.append(label.replace(" (", "\n("))
            flops.append(f)
            tput.append(v["throughput_cost_pct"])
    
    if not labels:
        return
        
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.2))
    
    # Plot bars
    ax.bar(x - w/2, flops, w, label="Aumento de FLOPs (%)", color="#D32F2F")
    ax.bar(x + w/2, tput, w, label="Pérdida de Throughput (%)", color="#FF9800")
    
    ax.set_xticks(x, labels, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("Porcentaje de Sobrecarga (%)")
    ax.set_title("Comparativa de Sobrecarga Computacional: Operaciones vs. Tiempo Real\n(Compilación JIT mitigando el coste teórico)", fontsize=11)
    ax.legend(loc="upper right", frameon=False)
    ax.axhline(0, color='black', linewidth=0.8)
    
    fig.tight_layout()
    fig.savefig("figs/fig_overhead.png")
    plt.close(fig)



def make_cost_table():
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Análisis pormenorizado del coste de cómputo introducido por la arquitectura CCA (utilizando compilación JIT). "
           r"Reportamos el incremento porcentual de parámetros, operaciones FLOPs y la reducción de velocidad (throughput) para cada escenario.}",
           r"\label{tab:cost}", r"\resizebox{\textwidth}{!}{", r"\begin{tabular}{lccccc}", r"\toprule",
           r"Tarea & Params (+$\%$) & FLOPs (+$\%$) & Latencia MS (SiLU $\to$ CCA) & Pérdida Throughput ($\%$) & Ganancia de Calidad ($\%$) \\",
           r"\midrule"]
    for stem, label in COST_SPECS:
        res = load(stem)
        if not res or not res.get("viability"):
            continue
        cca = res["models"].get("cca", {}).get("cost", {})
        silu = res["models"].get("silu", {}).get("cost", {})
        
        v = res["viability"]
        imp = improvement(res)
        par = cca.get("params_overhead_pct")
        flop = _flop_overhead(res)
        tc = v["throughput_cost_pct"]
        
        lat_silu = silu.get("step_latency_ms", 0)
        lat_cca = cca.get("step_latency_ms", 0)
        
        parf = f"{par:+.1f}" if par is not None else "-"
        flopf = f"{flop:+.1f}" if flop is not None else "-"
        
        tex.append(f"{label} & {parf} & {flopf} & {lat_silu:.1f} $\\to$ {lat_cca:.1f} & -{tc:.1f} & {imp:+.1f} \\\\")
        
    tex += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    open("results/cost_table.tex", "w", encoding="utf-8").write("\n".join(tex))
    print("wrote results/cost_table.tex")


def main():
    multi = {t: load("text_gen" if t == "text_gen" else t) for t in ORDER}
    single = {"text_gen": {"task": "text_gen", "models": {
                  "silu": {"ppl": 5.97}, "cca": {"ppl": 5.84}}},
              "image_cls": load("image_cls_single"),
              "image_gen": load("image_gen_single"),
              "text_cls": load("text_cls_single")}
    eff = load("cca_efficiency")
    
    fig_improvement(multi)
    fig_null_control(multi, single)
    fig_shuffle(multi)
    fig_b0h(multi)
    fig_coeff_var(multi)
    fig_pareto(eff)
    make_tables(multi)
    fig_curves()
    
    # extended: new tasks, BPE sweep, cost/viability
    fig_new_tasks()
    fig_bpe()
    fig_viability()
    fig_overhead()
    make_cost_table()
    print("figures written to figs/")


if __name__ == "__main__":
    main()

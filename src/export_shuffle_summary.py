"""
export_shuffle_summary.py — resume la ablacion de contexto barajado.

`paper_full_comparison.py` guarda el valor con contexto barajado por corrida en
`runs.csv` (columna `shuffle_value`), pero no lo agrega en `summary.json`. El
paper cita esas caidas, asi que este script las resume en un archivo citable:

    results/paper_full_comparison/shuffle_summary.json

Para cada tarea y metodo condicionado informa la media con contexto correcto, la
media con contexto barajado, la caida y el numero de semillas.

Uso:  python src/export_shuffle_summary.py [--runs ...] [--output ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/paper_full_comparison/runs.csv")
    parser.add_argument("--output",
                        default="results/paper_full_comparison/shuffle_summary.json")
    arguments = parser.parse_args()

    paired = defaultdict(lambda: {"value": [], "shuffle": []})
    with (ROOT / arguments.runs).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("shuffle_value"):
                continue
            entry = paired[(row["task"], row["method"])]
            entry["value"].append(float(row["value"]))
            entry["shuffle"].append(float(row["shuffle_value"]))

    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for (task, method), entry in sorted(paired.items()):
        mean_value = statistics.mean(entry["value"])
        mean_shuffle = statistics.mean(entry["shuffle"])
        summary.setdefault(task, {})[method] = {
            "n": len(entry["value"]),
            "mean_value": mean_value,
            "mean_shuffled_context": mean_shuffle,
            "drop": mean_value - mean_shuffle,
        }

    destination = ROOT / arguments.output
    destination.write_text(json.dumps({
        "source": arguments.runs,
        "note": ("Contexto barajado entre muestras del mismo lote. Las metricas "
                 "de generacion son menores-mejor, de modo que su 'drop' "
                 "negativo tambien indica degradacion."),
        "tasks": summary,
    }, indent=2) + "\n", encoding="utf-8")

    for task, methods in summary.items():
        for method, block in methods.items():
            print(f"{task:22s} {method:10s} n={block['n']:2d} "
                  f"{block['mean_value']:.5f} -> {block['mean_shuffled_context']:.5f}")
    print(f"escrito {destination}")


if __name__ == "__main__":
    main()

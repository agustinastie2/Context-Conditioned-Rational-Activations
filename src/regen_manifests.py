"""
regen_manifests.py — regenera los manifiestos SHA-256 de results/.
===============================================================================

Los manifiestos sirven para confirmar que los resultados publicados no se
tocaron despues de generarse. Varios quedaron desactualizados respecto de los
archivos que acompanan, de modo que este script los vuelve a calcular.

Comportamiento:
  * Si el directorio ya tiene manifest.sha256, conserva el orden y la lista de
    nombres original y solo actualiza los digests. Un archivo listado que ya no
    exista se informa y se omite.
  * Si no lo tiene, crea uno con los archivos del nivel superior del directorio
    (sin recorrer subcarpetas como workers/ ni logs/), en orden alfabetico.

Formato: "<sha256>  <nombre>\\n", identico al que emiten los scripts de corrida.

Uso:
  python src/regen_manifests.py            # regenera todo results/
  python src/regen_manifests.py --check    # solo informa, no escribe
  python src/regen_manifests.py results/budget_rematch_vision_audio
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "manifest.sha256"
SKIP = {MANIFEST}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listed_names(manifest: Path) -> list[str]:
    """Nombres en el orden en que los dejo la corrida original."""
    names = []
    for line in manifest.read_text(encoding="ascii").splitlines():
        if line.strip():
            names.append(line.split("  ", 1)[1])
    return names


def target_names(directory: Path) -> list[str]:
    manifest = directory / MANIFEST
    if manifest.exists():
        return listed_names(manifest)
    return sorted(p.name for p in directory.iterdir()
                  if p.is_file() and p.name not in SKIP)


def regenerate(directory: Path, check_only: bool) -> tuple[int, int, int]:
    """Devuelve (coincidian, actualizados, faltantes)."""
    manifest = directory / MANIFEST
    previous = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="ascii").splitlines():
            if line.strip():
                value, name = line.split("  ", 1)
                previous[name] = value

    lines, same, changed, missing = [], 0, 0, 0
    for name in target_names(directory):
        path = directory / name
        if not path.exists():
            print(f"  FALTA    {name}  (estaba en el manifiesto, ya no en disco)")
            missing += 1
            continue
        value = digest(path)
        if previous.get(name) == value:
            same += 1
        else:
            changed += 1
            state = "nuevo" if name not in previous else "cambiado"
            print(f"  ACTUALIZA {name}  ({state})")
        lines.append(f"{value}  {name}")

    if not check_only and lines:
        manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    return same, changed, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=[],
                        help="Directorios a regenerar (por omision, todo results/).")
    parser.add_argument("--check", action="store_true",
                        help="Informa diferencias sin reescribir nada.")
    parser.add_argument("--create-missing", action="store_true",
                        help="Tambien crea manifiestos en directorios que nunca tuvieron uno.")
    arguments = parser.parse_args()

    if arguments.paths:
        # Un directorio pedido a mano se procesa aunque no tenga manifiesto.
        directories = [Path(p) if Path(p).is_absolute() else ROOT / p
                       for p in arguments.paths]
    else:
        results = ROOT / "results"
        directories = sorted(d for d in results.iterdir()
                             if d.is_dir() and any(d.iterdir())
                             and (arguments.create_missing or (d / MANIFEST).exists()))

    totals = [0, 0, 0]
    for directory in directories:
        if not directory.is_dir():
            print(f"[--] {directory}: no existe", file=sys.stderr)
            continue
        print(f"[{directory.relative_to(ROOT)}]")
        same, changed, missing = regenerate(directory, arguments.check)
        print(f"  {same} sin cambios, {changed} regenerados, {missing} faltantes")
        totals = [a + b for a, b in zip(totals, (same, changed, missing))]

    verb = "se regenerarian" if arguments.check else "regenerados"
    print(f"\nTotal: {totals[0]} sin cambios, {totals[1]} {verb}, {totals[2]} faltantes")


if __name__ == "__main__":
    main()

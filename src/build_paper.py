"""
build_paper.py — convert PAPER.md to LaTeX (.tex) and PDF.
=========================================================

* LaTeX: pandoc (bundled via pypandoc-binary), standalone article.
* PDF:   PAPER.md -> HTML (pandoc) with figures embedded as base64, then
         printed to PDF by Microsoft Edge in headless mode (already installed
         on Windows; no LaTeX engine required).

Outputs land in paper/.   Run:  python build_paper.py   (from repo root)
"""

import base64
import glob
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pypandoc


def repo_root():
    """Locate the directory that contains PAPER.md (script may live in src/)."""
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent):
        if (cand / "PAPER.md").exists():
            return cand
    raise SystemExit("PAPER.md not found next to build_paper.py or its parent")


ROOT = repo_root()
MD = ROOT / "PAPER.md"
OUT = ROOT / "paper"
OUT.mkdir(exist_ok=True)

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; max-width: 100%; }
h1 { font-size: 19pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 14pt; margin: 16pt 0 4pt; border-bottom: 1px solid #ccc;
     padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 11pt 0 3pt; }
p, li { text-align: justify; }
code, pre { font-family: 'Consolas', monospace; font-size: 9pt; }
pre { background: #f5f5f5; padding: 7pt; border-radius: 4px; overflow-x: auto;
      white-space: pre-wrap; page-break-inside: avoid; }
code { background: #f0f0f0; padding: 0 2px; border-radius: 3px; }
pre code { background: none; padding: 0; }
img { max-width: 96%; display: block; margin: 8pt auto; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; text-align: left; }
th { background: #efefef; }
h1, h2, h3 { page-break-after: avoid; }
em { color: #222; }
"""


def embed_images(html: str) -> str:
    """Replace <img src="figs/..."> with base64 data URIs resolved from ROOT."""
    def repl(m):
        src = m.group(1)
        path = (ROOT / src).resolve()
        if not path.exists():
            return m.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        ext = path.suffix.lstrip(".").lower()
        return f'src="data:image/{ext};base64,{data}"'
    return re.sub(r'src="([^"]+)"', repl, html)


def build_latex(md_text: str):
    tex = OUT / "PAPER.tex"
    pypandoc.convert_text(
        md_text, "latex", format="markdown",
        outputfile=str(tex),
        extra_args=["--standalone", "--toc", "-V", "geometry:margin=2.5cm",
                    "-V", "documentclass=article", "-V", "fontsize=11pt",
                    # figs/ live one level up from paper/PAPER.tex
                    "-V", r"header-includes=\graphicspath{{../}{./}}",
                    "--resource-path", str(ROOT)])
    print(f"  wrote {tex.relative_to(ROOT)}  (compile from paper/ or repo root)")


def find_edge():
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def build_pdf(md_text: str):
    body = pypandoc.convert_text(md_text, "html", format="markdown",
                                 extra_args=["--mathml"])
    body = embed_images(body)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    tmp = OUT / "_paper_tmp.html"
    tmp.write_text(html, encoding="utf-8")

    edge = find_edge()
    pdf = OUT / "PAPER.pdf"
    if not edge:
        print("  [PDF] no Edge/Chrome found; wrote HTML only:",
              tmp.relative_to(ROOT))
        return
    url = tmp.resolve().as_uri()
    cmd = [edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf}", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if pdf.exists() and pdf.stat().st_size > 0:
        print(f"  wrote {pdf.relative_to(ROOT)} ({pdf.stat().st_size//1024} KB)")
        tmp.unlink(missing_ok=True)
    else:
        print("  [PDF] Edge did not produce a PDF; HTML kept at",
              tmp.relative_to(ROOT), "\n", r.stderr[-400:])


def main():
    md_text = MD.read_text(encoding="utf-8")
    print(f"Building paper from {MD.relative_to(ROOT)} -> {OUT.relative_to(ROOT)}/")
    build_latex(md_text)
    build_pdf(md_text)


if __name__ == "__main__":
    main()

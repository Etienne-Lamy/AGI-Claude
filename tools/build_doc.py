#!/usr/bin/env python3
"""Construit les versions .tex et .pdf d'un document .md (le .md reste la source).

Convention SCL : les documents de fond sont versionnés en .md (canonique) et dérivés
en .tex (source LaTeX, Overleaf-compilable) + .pdf. Ce script fait les deux dérivations
de façon reproductible :

    md → tex   : pandoc (via pypandoc_binary, aucun LaTeX système requis)
    tex → pdf  : tectonic (moteur LaTeX autonome, XeTeX)

Astuce de titre : le H1 (`# …`) du markdown devient le TITRE du document
(`--shift-heading-level-by=-1`), et les `##` deviennent les sections.

    python3 tools/build_doc.py SCL_bilan_modules_orchestrateur.md
    TECTONIC=/chemin/tectonic python3 tools/build_doc.py doc.md   # si tectonic hors PATH

Dépendances : `pip install pypandoc_binary` ; binaire `tectonic` sur le PATH ou via
la variable d'environnement TECTONIC.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pypandoc


def _tectonic():
    return os.environ.get("TECTONIC") or shutil.which("tectonic")


def construire(md_path: str, auteur="Etienne Lamy", date="2026-07-21"):
    md = Path(md_path).resolve()
    if not md.exists():
        sys.exit(f"introuvable : {md}")
    tex, pdf = md.with_suffix(".tex"), md.with_suffix(".pdf")

    args = [
        "--standalone",
        "--shift-heading-level-by=-1",      # le H1 devient le titre du document
        "--toc", "--toc-depth=2",
        "-V", "lang=fr",
        "-V", "geometry:margin=2.3cm",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue", "-V", "urlcolor=blue", "-V", "toccolor=black",
        "-V", "fontsize=11pt",
        "-M", f"author={auteur}",
        "-M", f"date={date}",
        "--metadata=lang:fr",
    ]
    # 1) .tex versionné
    pypandoc.convert_file(str(md), "latex", outputfile=str(tex), extra_args=args)
    print(f"écrit : {tex.name}")

    # 2) .pdf via tectonic
    tec = _tectonic()
    if not tec:
        print("tectonic introuvable (PATH ou $TECTONIC) — .pdf non généré.")
        return tex, None
    subprocess.run([tec, "--outdir", str(md.parent), str(tex)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"écrit : {pdf.name}")
    return tex, pdf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage : python3 tools/build_doc.py <document.md>")
    for p in sys.argv[1:]:
        construire(p)

"""Assemble a self-contained manuscript folder + zip (Overleaf / arXiv source) after the static checks pass.

Usage (from article-01-seattle/):
    python src/build_manuscript_bundle.py
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
OUT = MS / "build"


def main():
    shutil.copy(ROOT / "literature" / "references.bib", MS / "references.bib")
    check = subprocess.run([sys.executable, str(ROOT / "src" / "check_manuscript.py")], capture_output=True, text=True)
    print(check.stdout.strip())
    if check.returncode != 0:
        raise SystemExit("checks failed: bundle not built")
    OUT.mkdir(exist_ok=True)
    files = [MS / "main.tex", MS / "references.bib"] + sorted((MS / "tables").glob("*.tex")) + \
        sorted((MS / "figures").glob("*.pdf"))
    zpath = OUT / "article01_seattle_manuscript_source.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(MS).as_posix())
    print(f"{len(files)} files -> {zpath}")
    if "--pdf" in sys.argv:
        compile_pdf(files)
    if "--pdf-fr" in sys.argv:
        # French reading copy: same tables, figures and references; not part of the submission zip.
        fr = subprocess.run([sys.executable, str(ROOT / "src" / "check_manuscript.py"), "main_fr.tex"],
                            capture_output=True, text=True)
        print("FR:", fr.stdout.strip())
        if fr.returncode != 0:
            raise SystemExit("French copy checks failed")
        compile_pdf(files + [MS / "main_fr.tex"], tex_name="main_fr",
                    out_name="article01_seattle_preprint_FR_lecture.pdf", keep_bbl=False)


def find_tex_bin():
    """Locate a pdflatex installation (PATH first, then a per-user MiKTeX install)."""
    hit = shutil.which("pdflatex")
    if hit:
        return Path(hit).parent
    cand = Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"
    if (cand / "pdflatex.exe").exists():
        return cand
    raise SystemExit("pdflatex not found: install a TeX distribution or compile the zip on Overleaf")


def compile_pdf(files, tex_name="main", out_name="article01_seattle_preprint.pdf", keep_bbl=True):
    """Compile in a scratch copy (pdflatex, bibtex, pdflatex x2) exactly as arXiv would from the zip root."""
    tex = find_tex_bin()
    work = OUT / ("latex" if tex_name == "main" else f"latex_{tex_name}")
    if work.exists():
        shutil.rmtree(work)
    for f in files:
        dst = work / f.relative_to(MS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{tex_name}.tex"]
    steps = [latex, ["bibtex", tex_name], latex, latex]
    for cmd in steps:
        r = subprocess.run([str(tex / cmd[0])] + cmd[1:], cwd=work, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit(f"{cmd[0]} failed")
    log = (work / f"{tex_name}.log").read_text(encoding="utf-8", errors="replace")
    warnings = [l for l in log.splitlines() if "Warning" in l and ("undefined" in l or "Overfull" in l or "multiply" in l)]
    overfull = [l for l in log.splitlines() if l.startswith("Overfull")]
    print(f"undefined/multiply-defined warnings: {len(warnings)}; overfull boxes: {len(overfull)}")
    for l in (warnings + overfull)[:15]:
        print("  ", l)
    shutil.copy(work / f"{tex_name}.pdf", OUT / out_name)
    if keep_bbl:
        shutil.copy(work / "main.bbl", MS / "main.bbl")
    print("pdf ->", OUT / out_name)


if __name__ == "__main__":
    main()

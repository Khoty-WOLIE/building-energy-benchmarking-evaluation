"""Run directories with full provenance (protocol v1.0, section 10)."""
import datetime as dt
import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # article-01-seattle/


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def start_run(exp_id, data_paths):
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    d = ROOT / "results" / "experiments" / exp_id / run_id
    d.mkdir(parents=True, exist_ok=False)
    shutil.copy(ROOT / "experiments" / "config" / "protocol_v1.yaml", d / "config.yaml")
    (d / "data_sha256.txt").write_text("\n".join(f"{sha256(p)}  {Path(p).name}" for p in data_paths) + "\n")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit, dirty = "unknown", ""
    (d / "commit.txt").write_text(f"{commit}{' (working tree modified)' if dirty else ''}\n")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (d / "env.txt").write_text(f"python {sys.version}\nplatform {platform.platform()}\n"
                               f"processor {platform.processor()}\n\n{freeze}")
    return d

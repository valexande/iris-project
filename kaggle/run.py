"""Run a repo notebook on Kaggle and pull its outputs back into the repo.

    python kaggle/run.py sync-src            # upload src/ + notebooks as the private dataset <user>/arxiv-clf-src
    python kaggle/run.py push 01 [--gpu]     # sync, then run notebooks/01_*.py on Kaggle (private)
    python kaggle/run.py pull 01             # wait for it, then download outputs + executed notebook

Requires the Kaggle CLI and a token in ~/.kaggle/access_token.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(sys.executable).parent
KAGGLE = str(BIN / "kaggle")
JUPYTEXT = str(BIN / "jupytext")
USER = "alexandrosvass"
SRC_DATASET = f"{USER}/arxiv-clf-src"
ARXIV_DATASET = "Cornell-University/arxiv"

# Which earlier notebooks' outputs each notebook reads (attached as kernel_sources).
DEPENDS = {"01": [], "02": ["01"], "03": ["01", "02"], "04": ["01", "02"]}


def sh(*args, check=True):
    print("$", " ".join(args), flush=True)
    return subprocess.run(args, check=check, text=True, capture_output=True)


def notebook(nb_id: str) -> Path:
    return next((ROOT / "notebooks").glob(f"{nb_id}_*.py"))


def slug(nb_id: str) -> str:
    return "arxiv-" + notebook(nb_id).stem.replace("_", "-")


def to_ipynb(nb: Path, dest: Path) -> Path:
    sh(JUPYTEXT, "--to", "ipynb", "-o", str(dest), str(nb))
    doc = json.loads(dest.read_text())
    doc["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    doc["metadata"]["language_info"] = {"name": "python"}
    dest.write_text(json.dumps(doc))
    return dest


def sync_src():
    """Upload src/*.py and the notebooks (as .ipynb) as a new version of the private code dataset."""
    with tempfile.TemporaryDirectory() as tmp:
        for f in (ROOT / "src").glob("*.py"):
            shutil.copy(f, tmp)
        for nb in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.py")):
            to_ipynb(nb, Path(tmp) / f"{nb.stem}.ipynb")
        meta = {"title": "arxiv-clf-src", "id": SRC_DATASET, "licenses": [{"name": "CC0-1.0"}]}
        (Path(tmp) / "dataset-metadata.json").write_text(json.dumps(meta))
        exists = sh(KAGGLE, "datasets", "status", SRC_DATASET, check=False).returncode == 0
        cmd = ([KAGGLE, "datasets", "version", "-p", tmp, "-m", "update src"] if exists
               else [KAGGLE, "datasets", "create", "-p", tmp])
        print(sh(*cmd).stdout)
    # Wait until the new version is processed, otherwise a pushed kernel may mount the old one.
    time.sleep(20)
    for _ in range(40):
        out = sh(KAGGLE, "datasets", "status", SRC_DATASET, check=False).stdout
        if "ready" in out.lower():
            return
        time.sleep(10)


# Kaggle's own notebook runner does not hand back the executed notebook, so each notebook is run
# by a small script with papermill, which writes the executed .ipynb into /kaggle/working.
RUNNER = """import glob, os, papermill as pm
nb = glob.glob("/kaggle/input/**/{stem}.ipynb", recursive=True)[0]
os.chdir("/kaggle/working")
pm.execute_notebook(nb, "/kaggle/working/{stem}.ipynb", kernel_name="python3", log_output=True, cwd="/kaggle/working")
"""


def push(nb_id: str, gpu: bool):
    sync_src()
    nb = notebook(nb_id)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "run.py").write_text(RUNNER.format(stem=nb.stem))
        meta = {
            "id": f"{USER}/{slug(nb_id)}",
            "title": slug(nb_id),
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": gpu,
            "enable_internet": gpu,  # GPU notebooks download models from the Hugging Face Hub
            "dataset_sources": [ARXIV_DATASET, SRC_DATASET],
            "kernel_sources": [f"{USER}/{slug(d)}" for d in DEPENDS[nb_id]],
            "competition_sources": [],
        }
        if gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(sh(KAGGLE, "kernels", "push", "-p", tmp).stdout)


def pull(nb_id: str, poll: int = 30):
    ref = f"{USER}/{slug(nb_id)}"
    while True:
        status = sh(KAGGLE, "kernels", "status", ref, check=False).stdout
        print(time.strftime("%H:%M:%S"), status.strip(), flush=True)
        if any(s in status.lower() for s in ("complete", "error", "cancel")):
            break
        time.sleep(poll)
    out = ROOT / "kaggle" / "output" / nb_id
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    sh(KAGGLE, "kernels", "output", ref, "-p", str(out), "-o")
    # Copy small artefacts into the repo; large data/model files stay in kaggle/output (gitignored).
    for sub in ("reports/figures", "results"):
        if (out / sub).exists():
            shutil.copytree(out / sub, ROOT / sub, dirs_exist_ok=True)
    executed = out / f"{notebook(nb_id).stem}.ipynb"
    if executed.exists():
        shutil.copy(executed, ROOT / "notebooks" / executed.name)
    print("outputs in", out)
    if "error" in status.lower():
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["sync-src", "push", "pull"])
    p.add_argument("nb", nargs="?")
    p.add_argument("--gpu", action="store_true")
    a = p.parse_args()
    if a.action == "sync-src":
        sync_src()
    elif a.action == "push":
        push(a.nb, a.gpu)
    else:
        pull(a.nb)

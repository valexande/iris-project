"""Shared helpers: seeding, locating inputs on Kaggle or locally, JSON I/O."""
import glob
import json
import os
import random
from pathlib import Path

import numpy as np

SEED = 42
ON_KAGGLE = os.path.exists("/kaggle/input")
REPO_ROOT = Path(__file__).resolve().parent.parent

# On Kaggle everything written under /kaggle/working becomes a downloadable output.
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else REPO_ROOT
FIG_DIR = OUT_DIR / "reports" / "figures"
RESULTS_DIR = OUT_DIR / "results"
DATA_DIR = OUT_DIR / "data"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_input(filename: str) -> Path:
    """Find a file among attached Kaggle inputs (datasets or other notebooks' outputs) or the local data dir.

    Kaggle's mount layout has changed over time (/kaggle/input/<slug>/ vs
    /kaggle/input/datasets/<owner>/<slug>/), so search recursively instead of hard-coding it.
    """
    roots = ["/kaggle/input", str(DATA_DIR), str(REPO_ROOT / "data")]
    for root in roots:
        hits = sorted(glob.glob(os.path.join(root, "**", filename), recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"{filename} not found under {roots}")


def ensure_dirs() -> None:
    for d in (FIG_DIR, RESULTS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_str_keys(obj), f, indent=2, default=_to_builtin)


def _str_keys(o):
    """json only accepts str/int keys; pandas often hands us numpy ints (e.g. years)."""
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(_to_builtin(k) if isinstance(k, np.generic) else k)): _str_keys(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_str_keys(v) for v in o]
    return o


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _to_builtin(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))

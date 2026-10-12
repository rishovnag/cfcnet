"""Config loading, run bookkeeping and environment capture."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO / "configs" / "paper.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def data_path(cfg: dict, index: str, data_dir: str | Path | None = None) -> Path:
    data_dir = Path(data_dir) if data_dir else REPO / "data" / "raw"
    icfg = cfg["indices"][index]
    p = data_dir / icfg["file"]
    if not p.exists() and icfg.get("fallback") and (data_dir / icfg["fallback"]).exists():
        return data_dir / icfg["fallback"]
    return p


def load_frame(cfg: dict, index: str, data_dir=None):
    """Load an index with the sample start configured in configs/paper.yaml."""
    from .data import load_index
    return load_index(data_path(cfg, index, data_dir), cfg["indices"][index].get("start"))


def runs_dir(out: str | Path | None = None) -> Path:
    d = Path(out) if out else REPO / "runs"
    (d / "predictions").mkdir(parents=True, exist_ok=True)
    (d / "histories").mkdir(parents=True, exist_ok=True)
    return d


def completed(csv: Path, keys: list[str]) -> set:
    """Resume support: the set of key tuples already present in a results CSV."""
    if not csv.exists():
        return set()
    df = pd.read_csv(csv)
    return set(map(tuple, df[keys].astype(str).values.tolist()))


def append_row(csv: Path, row: dict) -> None:
    """Append one run to a CSV log, keeping a single consistent header: rows of different targets
    carry different per-column metrics, and diverged runs carry no metrics at all."""
    new = pd.DataFrame([row])
    if not csv.exists():
        new.to_csv(csv, index=False)
        return
    header = list(pd.read_csv(csv, nrows=0).columns)
    if set(new.columns) <= set(header):
        new.reindex(columns=header).to_csv(csv, mode="a", header=False, index=False)
    else:                                    # new columns: rewrite the log with the union of columns
        old = pd.read_csv(csv)
        pd.concat([old, new], ignore_index=True).to_csv(csv, index=False)


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "statsmodels", "tensorflow", "keras", "ncps"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod] = None
    try:
        import tensorflow as tf
        env["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        env["gpus"] = []
    return env


def write_environment(out: Path) -> None:
    """Write out/environment.json. A previous record from a different environment is kept as
    environment_<its date>.json, so runs made before a package change keep their provenance."""
    env, f = environment(), out / "environment.json"
    if f.exists():
        try:
            same = json.loads(f.read_text()) == env
        except ValueError:
            same = False
        if not same:
            f.rename(out / f"environment_{time.strftime('%Y%m%d-%H%M%S', time.localtime(f.stat().st_mtime))}.json")
    with open(f, "w") as fh:
        json.dump(env, fh, indent=2)


def quiet_tensorflow() -> None:
    """Must be called before TensorFlow is imported."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

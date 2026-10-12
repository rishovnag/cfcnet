#!/usr/bin/env python
"""Merge the per-run row files written by the workers into the CSV logs used by the analysis
scripts (06_dm_tests, 07_make_tables, 08_make_figures):

    runs/fixed_origin.csv, runs/walk_forward.csv, runs/history_ablation.csv, runs/environment.json

Safe to run while workers are still running (it only reads finished row files).
"""
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

FIRST = {
    "fixed": ["experiment", "index", "target", "repr", "arch", "units", "test_frac", "seed", "run_id", "motor"],
    "wf": ["index", "target", "repr", "arch", "units", "fold", "seed", "regime", "run_id"],
    "hist": ["index", "target", "repr", "arch", "units", "hist_len", "hist_years", "seed", "run_id"],
}
FILES = {"fixed": "fixed_origin.csv", "wf": "walk_forward.csv", "hist": "history_ablation.csv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    args = ap.parse_args()
    out = Path(args.out)
    rows = [json.loads(f.read_text()) for f in sorted((out / "rows").glob("*.json"))]
    if not rows:
        print("no finished runs yet")
        return
    df = pd.DataFrame(rows)
    folds = out / "walk_forward_folds.csv"
    for kind, g in df.groupby("kind"):
        g = g.dropna(axis=1, how="all").copy()
        for c in ("fold", "hist_len", "seed", "units", "motor", "n_params", "n_train", "n_test", "pack", "epochs"):
            if c in g.columns and g[c].notna().all():
                g[c] = g[c].astype(int)
        if kind == "wf" and folds.exists():
            reg = pd.read_csv(folds)[["fold", "regime"]]
            g = g.drop(columns=["regime"], errors="ignore").merge(reg, on="fold", how="left")
        first = [c for c in FIRST[kind] if c in g.columns]
        g = g[first + [c for c in g.columns if c not in first and c != "kind"]]
        g = g.sort_values([c for c in first if c not in ("run_id",)]).reset_index(drop=True)
        g.to_csv(out / FILES[kind], index=False)
        print(f"{FILES[kind]}: {len(g)} runs ({(g.get('status') == 'ok').sum()} ok)")
    env = {"hosts": sorted(df.get("host", pd.Series(dtype=str)).dropna().unique().tolist()),
           "gpus": sorted(df.get("gpu_name", pd.Series(dtype=str)).dropna().unique().tolist())}
    try:
        from cfcnet.io import environment
        env.update(environment())
    except Exception:
        pass
    (out / "environment.json").write_text(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the job queue for the GPU runner (the full paper grid by default).

    python gpu/plan.py                         # all experiments, targets, representations
    python gpu/plan.py --pack 3                # seeds per job (default from configs/gpu.yaml)
    python gpu/plan.py --only fixed --experiment headline --targets ohlc
    python gpu/plan.py --import <runs dir>     # reuse runs already finished elsewhere (e.g. on a CPU machine)

Writes runs/jobs.jsonl, runs/walk_forward_folds.csv and runs/adf_per_fold.csv, and prints how many
runs are already complete. Re-running the plan never repeats a finished run.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from cfcnet import jobs as J  # noqa: E402
from cfcnet.io import REPO, load_config  # noqa: E402

KEYS = {"fixed_origin.csv": "fixed", "walk_forward.csv": "wf", "history_ablation.csv": "hist"}


def import_runs(src: Path, out: Path) -> int:
    """Convert finished runs of the reference path (CSV logs + npz + json) into row files."""
    n = 0
    for fname, kind in KEYS.items():
        f = src / fname
        if not f.exists():
            continue
        for r in pd.read_csv(f).to_dict("records"):
            rid = r["run_id"]
            if (out / "rows" / f"{rid}.json").exists():
                continue
            for sub, ext in (("predictions", "npz"), ("histories", "json")):
                s = src / sub / f"{rid}.{ext}"
                if s.exists() and s.resolve() != (out / sub / f"{rid}.{ext}").resolve():
                    shutil.copy(s, out / sub / f"{rid}.{ext}")
            r = {k: v for k, v in r.items() if not (isinstance(v, float) and pd.isna(v))}
            r.update({"kind": kind, "imported_from": str(src)})
            (out / "rows" / f"{rid}.json").write_text(json.dumps(r, default=float))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=str(REPO / "runs"))
    ap.add_argument("--pack", type=int, default=None)
    ap.add_argument("--only", nargs="*", choices=["fixed", "wf", "hist"], default=None)
    ap.add_argument("--experiment", nargs="*", default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--reprs", nargs="*", default=None)
    ap.add_argument("--arch", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=None, help="override the seed count (e.g. 2 for a smoke test)")
    ap.add_argument("--import", dest="imp", default=None, help="runs folder with finished runs to reuse (e.g. from a CPU machine)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rt = yaml.safe_load(open(args.runtime or REPO / "configs" / "gpu.yaml"))
    out = Path(args.out)
    for d in ("rows", "locks", "predictions", "histories", "logs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if args.imp:
        print(f"imported {import_runs(Path(args.imp), out)} finished runs from {args.imp}")
    jobs = J.plan(cfg, pack=args.pack or rt.get("pack", 1), data_dir=args.data_dir, experiments=args.experiment,
                  include=tuple(args.only or ("fixed", "wf", "hist")), targets=args.targets, reprs=args.reprs,
                  archs=args.arch, seeds_override=list(range(args.seeds)) if args.seeds else None)
    J.save(jobs, out / "jobs.jsonl")
    folds, adf = J.walk_forward_tables(cfg, args.data_dir)
    folds.to_csv(out / "walk_forward_folds.csv", index=False)
    adf.to_csv(out / "adf_per_fold.csv", index=False)
    runs = [r for j in jobs for r in j.run_ids]
    done = sum(J.run_done(out / "rows", r) for r in runs)
    left = [j for j in jobs if not J.job_done(out / "rows", j)]
    by = pd.DataFrame([{"kind": j.kind, "arch": j.arch, "runs": len(j.run_ids)} for j in jobs])
    print(by.groupby(["kind", "arch"]).runs.sum().unstack(fill_value=0).to_string())
    print(f"\n{len(jobs)} jobs / {len(runs)} runs; {done} runs already complete; {len(left)} jobs to run "
          f"({len(folds)} walk-forward folds)")
    print(f"plan -> {out / 'jobs.jsonl'}")


if __name__ == "__main__":
    main()

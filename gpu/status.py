#!/usr/bin/env python
"""Progress of the job queue: finished runs per kind / target / representation / architecture,
running jobs (lock files), failures, and the latest line of every worker log.

    python gpu/status.py [--out runs]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

from cfcnet import jobs as J  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs"))
    args = ap.parse_args()
    out = Path(args.out)
    plan = J.load(out / "jobs.jsonl")
    recs = []
    for j in plan:
        for r in j.run_ids:
            recs.append({"kind": j.kind, "target": j.target, "repr": j.repr, "arch": j.arch,
                         "done": J.run_done(out / "rows", r)})
    df = pd.DataFrame(recs)
    t = df.groupby(["kind", "target", "repr", "arch"]).done.agg(["sum", "count"])
    t["pct"] = (100 * t["sum"] / t["count"]).round(0)
    print(t.to_string())
    print(f"\nTOTAL {int(df.done.sum())}/{len(df)} runs ({100 * df.done.mean():.1f}%)")
    locks = list((out / "locks").glob("*.lock"))
    print(f"running jobs: {len(locks)}")
    for f in locks[:20]:
        d = json.loads(f.read_text() or "{}")
        print(f"  {f.stem}  [{d.get('worker', '?')}]")
    fails = list((out / "failed").glob("*.txt"))
    if fails:
        print(f"jobs with failures: {len(fails)} (see {out / 'failed'})")
    for log in sorted((out / "logs").glob("worker_*.log")):
        lines = log.read_text(errors="ignore").strip().splitlines()
        if lines:
            print(f"{log.stem}: {lines[-1][:140]}")


if __name__ == "__main__":
    main()

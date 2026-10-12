#!/usr/bin/env python
"""Find the fastest workers-per-GPU x seeds-per-job setting on THIS machine before the full run.

    python gpu/autotune.py                                  # defaults below, GPU 0
    python gpu/autotune.py --workers 8 16 24 32 --pack 1 3 --epochs 25 --arch cfc ltc gru

For every combination it launches the real runner on a short probe (NIFTY 100 headline window,
joint OHLC, levels; `--epochs` epochs; one job per worker per architecture) and measures the
throughput in full-run equivalents per hour (runs x epochs / 400 / hour). Compilation time is
included, so the estimate is slightly pessimistic for 400-epoch runs. Results go to
runs/autotune.csv; --write stores the best setting in configs/gpu.yaml.
"""
import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from cfcnet import jobs as J  # noqa: E402
from cfcnet.io import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", nargs="*", type=int, default=[4, 8, 16, 24, 32])
    ap.add_argument("--pack", nargs="*", type=int, default=[1, 3])
    ap.add_argument("--arch", nargs="*", default=["cfc", "ltc", "gru"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--gpus", nargs="*", type=int, default=[0])
    ap.add_argument("--runtime", default=str(ROOT / "configs" / "gpu.yaml"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--write", action="store_true", help="store the best setting in the runtime config")
    args = ap.parse_args()
    cfg = load_config()
    full_epochs = int(cfg["protocol"]["epochs"])
    results = []
    for P in args.pack:
        base = J.plan(cfg, pack=P, data_dir=args.data_dir, experiments=["headline"], include=("fixed",),
                      targets=["ohlc"], reprs=["levels"], archs=args.arch, seeds_override=list(range(P)))
        for W in args.workers:
            tmp = ROOT / "runs" / "autotune" / f"w{W}_p{P}"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            probe = []
            for i in range(W * len(args.gpus)):
                for j in base:
                    probe.append(replace(j, job_id=f"{j.job_id}_probe{i}", run_ids=[f"{r}_probe{i}" for r in j.run_ids]))
            J.save(probe, tmp / "jobs.jsonl")
            cmd = [sys.executable, str(ROOT / "gpu" / "run_parallel.py"), "--out", str(tmp),
                   "--workers-per-gpu", str(W), "--epochs", str(args.epochs), "--runtime", args.runtime, "--no-merge"]
            cmd += ["--cpu-only"] if args.cpu_only else ["--gpus", *map(str, args.gpus)]
            if args.data_dir:
                cmd += ["--data-dir", args.data_dir]
            t0 = time.time()
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
            wall = time.time() - t0
            n_runs = len(list((tmp / "rows").glob("*.json")))
            per_h = n_runs * args.epochs / full_epochs / wall * 3600
            results.append({"workers_per_gpu": W, "pack": P, "gpus": len(args.gpus), "runs": n_runs,
                            "wall_s": round(wall, 1), "full_run_equiv_per_hour": round(per_h, 1)})
            print(results[-1], flush=True)
    df = pd.DataFrame(results).sort_values("full_run_equiv_per_hour", ascending=False)
    (ROOT / "runs").mkdir(exist_ok=True)
    df.to_csv(ROOT / "runs" / "autotune.csv", index=False)
    print("\n" + df.to_string(index=False))
    best = df.iloc[0]
    print(f"\nbest: workers_per_gpu={int(best.workers_per_gpu)} pack={int(best.pack)}")
    if args.write:
        rt = yaml.safe_load(open(args.runtime))
        rt["workers_per_gpu"], rt["pack"] = int(best.workers_per_gpu), int(best.pack)
        txt = open(args.runtime).read()
        import re
        txt = re.sub(r"^workers_per_gpu:\s*\d+", f"workers_per_gpu: {rt['workers_per_gpu']}", txt, flags=re.M)
        txt = re.sub(r"^pack:\s*\d+", f"pack: {rt['pack']}", txt, flags=re.M)
        open(args.runtime, "w").write(txt)
        print(f"updated {args.runtime}")


if __name__ == "__main__":
    main()

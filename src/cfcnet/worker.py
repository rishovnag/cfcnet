"""One worker process: claims jobs from the plan, trains them on its GPU, writes one row file per run.

    python -m cfcnet.worker --plan runs/jobs.jsonl --out runs --worker-id 0 [--epochs N]

Started by gpu/run_parallel.py with the environment from gpuenv.worker_env. Workers never
write to a shared file: each run produces runs/rows/<run_id>.json (+ predictions / history), and
job ownership is a lock file created with O_EXCL, so any number of workers on any number of GPUs
can share one run folder.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim(lock_dir: Path, job_id: str, worker: str, fail_dir: Path | None = None) -> bool:
    """Take ownership of a job with an O_EXCL lock file. A lock left by a worker that died without
    cleaning up (OOM kill, SIGKILL, node reboot of the same host) is detected through its pid,
    recorded as a failure of that job, and taken over."""
    path = lock_dir / f"{job_id}.lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                info = json.loads(path.read_text() or "{}")
            except (OSError, ValueError):
                return False                       # being written right now by another worker
            if info.get("host") != socket.gethostname() or _pid_alive(int(info.get("pid", -1))):
                return False
            if fail_dir is not None:
                with open(fail_dir / f"{job_id}.txt", "a") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {info.get('worker')} died while running "
                             f"(pid {info.get('pid')} gone; out of memory?)\n")
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"worker": worker, "host": socket.gethostname(), "pid": os.getpid(), "t": time.time()}))
        return True
    return False


def release(lock_dir: Path, job_id: str) -> None:
    try:
        os.remove(lock_dir / f"{job_id}.lock")
    except FileNotFoundError:
        pass


def write_row(rows_dir: Path, run_id: str, row: dict) -> None:
    tmp = rows_dir / f".{run_id}.json.tmp"
    tmp.write_text(json.dumps(row, default=float))
    os.replace(tmp, rows_dir / f"{run_id}.json")          # atomic: a row file is either complete or absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-jobs", type=int, default=0, help="stop after this many jobs (0 = until the queue is empty)")
    args = ap.parse_args()

    import yaml
    from . import gpuenv
    from .io import REPO, load_config
    rt = yaml.safe_load(open(args.runtime or REPO / "configs" / "gpu.yaml"))

    import tensorflow as tf  # noqa: F401  (after the environment is set by the launcher)
    dev = gpuenv.configure_tf(rt)

    from . import jobs as J
    from .data import make_pairs, prepare_window
    from .fast import train_pack
    from .io import load_frame

    cfg = load_config(args.config)
    pr = cfg["protocol"]
    epochs = args.epochs or int(pr["epochs"])
    out = Path(args.out)
    rows_dir, lock_dir, fail_dir = out / "rows", out / "locks", out / "failed"
    for d in (rows_dir, lock_dir, fail_dir, out / "predictions", out / "histories"):
        d.mkdir(parents=True, exist_ok=True)
    plan = J.load(Path(args.plan))
    me = f"{socket.gethostname()}:{args.worker_id}"
    print(f"[{me}] {dev} jobs={len(plan)} epochs={epochs}", flush=True)

    frames, pairs_cache = {}, {}
    n_done = 0
    for job in plan:
        fails = fail_dir / f"{job.job_id}.txt"
        if fails.exists() and len(fails.read_text().splitlines()) > int(rt.get("max_retries", 2)):
            continue                                   # failed repeatedly: left for inspection, see runs/failed/
        if J.job_done(rows_dir, job) or not claim(lock_dir, job.job_id, me, fail_dir):
            continue
        try:
            todo = [(s, r) for s, r in zip(job.seeds, job.run_ids) if not J.run_done(rows_dir, r)]
            if not todo:
                continue
            if job.index not in frames:
                frames[job.index] = load_frame(cfg, job.index, args.data_dir)
            key = (job.index, job.target, job.repr)
            if key not in pairs_cache:
                pairs_cache[key] = make_pairs(frames[job.index], job.target, job.repr, pr["feature_cols"],
                                              pr["add_weekday"])
            pairs = pairs_cache[key]
            prep = prepare_window(pairs, *job.window, pr["warmup"])
            t0 = time.time()
            xla = rt.get("xla_ncps", True) if job.arch in ("cfc", "ltc") else False
            rows = train_pack(prep, job.arch, job.units, [s for s, _ in todo], epochs=epochs,
                              learning_rate=pr["learning_rate"], xla=xla, whole_loop=rt.get("whole_loop", True),
                              target_names=pairs.target_cols, run_ids=[r for _, r in todo], out_dir=out,
                              extra={"worker": me, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                                     "gpu_name": dev.get("gpu_name", ""), "epochs": epochs})
            for (seed, run_id), row in zip(todo, rows):
                write_row(rows_dir, run_id, {"kind": job.kind, "job_id": job.job_id, "run_id": run_id, "seed": seed,
                                             "index": job.index, "target": job.target, "repr": job.repr,
                                             "arch": job.arch, "units": job.units,
                                             "motor": 4 if job.target == "ohlc" else 1, **job.meta, **row})
            n_done += 1
            r2s = ", ".join("%.3f" % r.get("r2_joint", float("nan")) for r in rows)
            print(f"[{me}] {job.job_id}: {len(rows)} runs in {time.time() - t0:.0f}s (R2 {r2s})", flush=True)
        except Exception as exc:
            traceback.print_exc()
            sys.stdout.flush()
            with open(fails, "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {me} {type(exc).__name__}: {exc}\n")
            release(lock_dir, job.job_id)
            raise                              # the launcher restarts the worker (bounded retries)
        release(lock_dir, job.job_id)
        if args.max_jobs and n_done >= args.max_jobs:
            break
    print(f"[{me}] queue empty, {n_done} jobs done", flush=True)


if __name__ == "__main__":
    main()

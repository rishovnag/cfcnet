#!/usr/bin/env python
"""Run the job queue on one or more GPUs with many concurrent workers.

    python gpu/run_parallel.py                                  # settings from configs/gpu.yaml
    python gpu/run_parallel.py --gpus 0 1 --workers-per-gpu 12  # two GPUs, 24 workers
    python gpu/run_parallel.py --epochs 2 --workers-per-gpu 4   # smoke test

Each worker is a separate process (GPU environment set before TensorFlow loads), pinned to one GPU
and, when the host has enough cores, to its own CPU cores. With --mps the NVIDIA Multi-Process
Service lets the workers' kernels run concurrently on the GPU's SMs; a batch-1 recurrent model
occupies only a small fraction of an H100, so many workers per GPU are needed to fill it.
Workers that crash are restarted (bounded by max_retries); progress and an ETA are printed
periodically. When the queue is empty, gpu/merge.py is run to build the CSV logs.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml  # noqa: E402

from cfcnet import gpuenv, jobs as J  # noqa: E402


def start_mps(out: Path) -> bool:
    if shutil.which("nvidia-cuda-mps-control") is None:
        print("MPS: nvidia-cuda-mps-control not found; continuing without MPS")
        return False
    r = subprocess.run(["bash", str(ROOT / "gpu" / "mps.sh"), "start"], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--runtime", default=str(ROOT / "configs" / "gpu.yaml"))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--gpus", nargs="*", type=int, default=None)
    ap.add_argument("--workers-per-gpu", type=int, default=None)
    ap.add_argument("--cpu-only", action="store_true", help="run workers without a GPU (testing)")
    ap.add_argument("--mps", dest="mps", action="store_true", default=None)
    ap.add_argument("--no-mps", dest="mps", action="store_false")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--keep-locks", action="store_true", help="do not clear lock files left by a previous launch")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    rt = yaml.safe_load(open(args.runtime))
    gpus = [None] if args.cpu_only else (args.gpus if args.gpus is not None else rt.get("gpus", [0]))
    wpg = args.workers_per_gpu or int(rt.get("workers_per_gpu", 8))
    use_mps = (rt.get("mps", True) if args.mps is None else args.mps) and not args.cpu_only
    out = Path(args.out)
    plan_path = Path(args.plan) if args.plan else out / "jobs.jsonl"
    plan = J.load(plan_path)
    total = sum(len(j.run_ids) for j in plan)
    for d in ("rows", "locks", "logs", "failed"):
        (out / d).mkdir(parents=True, exist_ok=True)
    if not args.keep_locks:
        for f in (out / "locks").glob("*.lock"):
            f.unlink()
    if use_mps and not args.cpu_only:
        start_mps(out)

    slots = [(g, i) for g in gpus for i in range(wpg)]
    cpus = gpuenv.cpu_slices(len(slots), int(rt.get("cpu_threads_per_worker", 2))) if rt.get("pin_cpus", True) \
        else [None] * len(slots)
    base_env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if args.cpu_only:
        base_env["CUDA_VISIBLE_DEVICES"] = ""
    if use_mps and not args.cpu_only:                     # MPS clients find the server through these
        base_env.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
        base_env.setdefault("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-log")

    def spawn(k):
        g, i = slots[k]
        wid = f"g{'cpu' if g is None else g}w{i}"
        env = gpuenv.worker_env(base_env, None if args.cpu_only else g, rt, out / "xla_cache", cpus[k])
        cmd = [sys.executable, "-m", "cfcnet.worker", "--plan", str(plan_path), "--out", str(out),
               "--worker-id", wid, "--runtime", args.runtime]
        for flag, val in (("--config", args.config), ("--data-dir", args.data_dir), ("--epochs", args.epochs)):
            if val is not None:
                cmd += [flag, str(val)]
        log = open(out / "logs" / f"worker_{wid}.log", "a")
        return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

    procs = {}
    for k in range(len(slots)):
        procs[k] = spawn(k)
        time.sleep(0.5)                                   # stagger TF / CUDA initialisation
    retries = {k: 0 for k in procs}
    print(f"{len(procs)} workers on GPUs {gpus} ({wpg} per GPU, MPS={'on' if use_mps else 'off'}); "
          f"{total} runs in {len(plan)} jobs; logs in {out / 'logs'}", flush=True)

    def stop(*_):
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        sys.exit(130)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    t_start = time.time()
    done0 = sum(J.run_done(out / "rows", r) for j in plan for r in j.run_ids)
    last = 0.0
    while True:
        alive = 0
        for k, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                alive += 1
            elif rc != 0 and retries[k] < int(rt.get("max_retries", 2)) + 2:
                retries[k] += 1
                procs[k] = spawn(k)
                alive += 1
        if time.time() - last >= rt.get("progress_every_s", 60) or alive == 0:
            last = time.time()
            done = sum(1 for _ in (out / "rows").glob("*.json"))
            done = min(done, total)
            el = time.time() - t_start
            rate = (done - done0) / el * 3600 if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else float("inf")
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{total} runs | {rate:.0f} runs/h | "
                  f"ETA {eta:.1f} h | workers alive {alive}", flush=True)
        if alive == 0:
            break
        time.sleep(5)
    missing = [r for j in plan for r in j.run_ids if not J.run_done(out / "rows", r)]
    failed = list((out / "failed").glob("*.txt"))
    if failed:
        print(f"{len(failed)} jobs recorded failures; see {out / 'failed'} and the worker logs")
    if missing:
        print(f"{len(missing)} runs incomplete (e.g. {missing[0]}); re-launch the same command to resume them, "
              f"or lower workers_per_gpu / memory if workers were killed")
    else:
        print(f"all {total} runs complete")
    if not args.no_merge:
        subprocess.run([sys.executable, str(ROOT / "gpu" / "merge.py"), "--out", str(out)], check=False)


if __name__ == "__main__":
    main()

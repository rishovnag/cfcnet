"""Process-level parallelism for the experiment grid, on the CPU cores and the GPU.

A single run trains on one sequence of a few thousand sessions with batch size 1, i.e. a long
chain of small sequential ops, so extra intra-op threads do not make it faster (measured on an
M3 Ultra, 32 cores: CfC(80,4) 0.80 s/epoch at 1, 4 or 32 threads; LSTM-80 0.34 s/epoch at 1
thread vs 0.55 s at 32). Throughput comes from running many independent runs at once: one
worker process per core, one TensorFlow / BLAS thread per worker (32 concurrent CfC(80,4) runs
complete ~22x more epochs per second than one).

GPU. On the Apple-silicon GPU (tensorflow-metal) the Keras LSTM/GRU layers train ~1.7x faster
per run than on one core (LSTM-80 0.25 vs 0.39 s/epoch on NIFTY 100), and several runs share the
GPU well; the ncps CfC/LTC cells are ~1000x SLOWER there and must stay on the CPU. Each GPU worker
also occupies a CPU core to drive the GPU. Measured mixes (CfC on CPU + LSTM on GPU, total work
in CPU-core equivalents): 32+0 cores = 32.0; 28+4 = 34.3; 24+8 = 35.5; 24+16 = 35.4; 16+16 = 31.8.
Hence the default of 8 GPU workers next to (cores - 8) CPU workers. Once no GPU task is left,
the CPU pool grows back to every core. Which architectures go to the GPU is decided per
architecture, never per run, so every table cell is trained on one device.

Only LSTM goes to the GPU by default. Over full 400-epoch runs (NIFTY 100, seeds 0-3, both
targets) LSTM on the Metal GPU reproduces the CPU R2 and MASE to 4 decimals, whereas GRU is
systematically worse on the GPU (mean R2 0.637 vs 0.651 for the adjusted close, lower on every
seed), which would bias the GRU-vs-CfC comparison. --gpu-archs lstm gru overrides this.

Workers are spawned processes, so every run keeps its own seeding and Keras session exactly as
in a serial run. Only the parent process writes the CSV logs; workers write the per-run
prediction and history files, whose names are unique per run.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
MAX_TASKS_PER_CHILD = 25    # recycle workers periodically to bound TF/Keras memory growth
DEFAULT_GPU_WORKERS = 8     # see the measured mixes above
GPU_ARCHS = ("lstm",)  # see the module docstring: GRU is faster on the GPU but not equivalent


def default_workers() -> int:
    """All logical cores (performance + efficiency cores on Apple silicon)."""
    return os.cpu_count() or 1


def add_args(ap, gpu: bool = False):
    ap.add_argument("--workers", type=int, default=None,
                    help="CPU worker processes (default: all cores minus the GPU workers; "
                         "1 with --gpu-workers 0 = run in this process)")
    ap.add_argument("--threads", type=int, default=1, help="TensorFlow / BLAS threads per worker (default 1)")
    if gpu:
        ap.add_argument("--gpu-workers", type=int, default=None,
                        help=f"processes training on the GPU (default: {DEFAULT_GPU_WORKERS} if TensorFlow "
                             "sees a GPU, else 0; 0 = CPU only)")
        ap.add_argument("--gpu-archs", nargs="*", default=list(GPU_ARCHS),
                        help="architectures trained on the GPU (default: lstm)")
    return ap


def gpu_available() -> bool:
    import tensorflow as tf
    return bool(tf.config.list_physical_devices("GPU"))


def gpu_workers(args) -> int:
    n = getattr(args, "gpu_workers", None)
    if n is None:
        n = DEFAULT_GPU_WORKERS if gpu_available() else 0
    return max(0, n)


def device_for(arch: str, args, n_gpu: int) -> str:
    return "gpu" if n_gpu and arch in (getattr(args, "gpu_archs", None) or ()) else "cpu"


def _init_worker(threads: int, tensorflow: bool, gpu: bool) -> None:
    if tensorflow:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)
        if not gpu:
            _hide_gpus(tf)


def _hide_gpus(tf) -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _call(fn, kwargs):
    try:
        return fn(**kwargs)
    except Exception as e:      # return, not raise, so the parent can log it with the worker's traceback
        return TaskError(f"{type(e).__name__}: {e}", traceback.format_exc())


class TaskError:
    def __init__(self, message: str, tb: str):
        self.message, self.traceback = message, tb

    def __repr__(self):
        return f"TaskError({self.message})"


def run(fn, tasks, *, workers: int | None = None, threads: int = 1, tensorflow: bool = False,
        gpu_workers: int = 0, on_gpu=None):
    """Run fn(**kwargs) for every (key, kwargs) in `tasks`; yield (key, result) as runs finish.

    Tasks for which on_gpu(key) is true run in `gpu_workers` processes that see the GPU; all
    others run in CPU-only processes. Within each device, tasks start in the given order, so pass
    the most expensive first. A failed task yields a TaskError instead of a result. With
    workers=1 and no GPU workers the tasks run in this process, in order, as a plain loop would."""
    tasks = list(tasks)
    if not tasks:
        return
    is_gpu = (lambda key: bool(on_gpu(key))) if (gpu_workers and on_gpu) else (lambda key: False)
    gpu_q = deque(t for t in tasks if is_gpu(t[0]))
    cpu_q = deque(t for t in tasks if not is_gpu(t[0]))
    n_gpu = min(gpu_workers, len(gpu_q))
    n_cpu = min(workers or max(1, default_workers() - n_gpu), len(cpu_q)) if cpu_q else 0
    total = n_cpu + n_gpu
    t0, n = time.time(), len(tasks)

    if n_cpu <= 1 and n_gpu == 0:
        if tensorflow:      # CfC/LTC must not land on a Metal GPU (a CUDA GPU stays visible, e.g. for --jit)
            import tensorflow as tf
            if any(tf.config.experimental.get_device_details(g).get("device_name") == "METAL"
                   for g in tf.config.list_physical_devices("GPU")):
                _hide_gpus(tf)
        for i, (key, kwargs) in enumerate(tasks, 1):
            yield key, _call(fn, kwargs)
            _progress(i, n, t0)
        return

    # Spawned children inherit this environment; set thread limits before they import numpy / TF.
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    os.environ.update({v: str(threads) for v in _THREAD_VARS})
    kw = {"max_tasks_per_child": MAX_TASKS_PER_CHILD} if sys.version_info >= (3, 11) else {}
    make = lambda size, gpu: ProcessPoolExecutor(size, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                                 initargs=(threads, tensorflow, gpu), **kw)
    # the CPU pool may grow to every core once the GPU queue is empty
    pools = {"cpu": make(max(total, 1), False) if cpu_q else None, "gpu": make(n_gpu, True) if n_gpu else None}
    print(f"[parallel] {n} tasks: {len(cpu_q)} on {n_cpu} CPU workers"
          + (f", {len(gpu_q)} on {n_gpu} GPU workers" if n_gpu else "") + f" x {threads} thread(s)", flush=True)
    running = {}

    def fill():
        busy_gpu = sum(dev == "gpu" for _, dev in running.values())
        while gpu_q and busy_gpu < n_gpu:
            key, kwargs = gpu_q.popleft()
            running[pools["gpu"].submit(_call, fn, kwargs)] = (key, "gpu")
            busy_gpu += 1
        if pools["gpu"] is not None and not gpu_q and busy_gpu == 0:
            pools["gpu"].shutdown(wait=False)       # free the GPU workers' cores and memory
            pools["gpu"] = None
        cap = total - (n_gpu if gpu_q else busy_gpu)
        busy_cpu = len(running) - busy_gpu
        while cpu_q and busy_cpu < cap:
            key, kwargs = cpu_q.popleft()
            running[pools["cpu"].submit(_call, fn, kwargs)] = (key, "cpu")
            busy_cpu += 1

    try:
        fill()
        i = 0
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                key, _dev = running.pop(f)
                try:
                    res = f.result()
                except Exception as e:      # worker died (e.g. killed); the run can be resumed later
                    res = TaskError(f"{type(e).__name__}: {e}", traceback.format_exc())
                i += 1
                yield key, res
                _progress(i, n, t0)
            fill()
        for ex in pools.values():
            if ex is not None:
                ex.shutdown(wait=True)
    except BaseException:       # Ctrl-C or an error in the caller: stop the workers now, not after their runs
        for ex in pools.values():
            if ex is None:
                continue
            procs = list((getattr(ex, "_processes", None) or {}).values())
            ex.shutdown(wait=False, cancel_futures=True)
            for pr in procs:
                pr.terminate()
        raise
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def _progress(i: int, n: int, t0: float) -> None:
    el = time.time() - t0
    eta = el / i * (n - i)
    print(f"[{i}/{n}] elapsed {el / 60:.1f} min, ETA ~{eta / 60:.1f} min", flush=True)


def report_failures(failures: list) -> None:
    """Print the failed tasks and exit non-zero (finished runs are already logged; re-run to resume)."""
    if not failures:
        return
    print(f"\n{len(failures)} task(s) failed:", file=sys.stderr)
    for key, err in failures:
        print(f"--- {key}\n{err.traceback}", file=sys.stderr)
    sys.exit(1)

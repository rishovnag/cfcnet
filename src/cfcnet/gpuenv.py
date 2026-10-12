"""Process-level GPU set-up. Environment variables must be set BEFORE TensorFlow is imported,
which is why every worker is a separate process started by gpu/run_parallel.py."""
from __future__ import annotations

import os
from pathlib import Path


def worker_env(base: dict, gpu: int | None, rt: dict, cache_dir: Path | None, cpus: list[int] | None = None) -> dict:
    """Environment for one worker process."""
    env = dict(base)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"          # stream-ordered allocator: less fragmentation across many models
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    threads = str(rt.get("cpu_threads_per_worker", 2))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        env[k] = threads
    env["TF_NUM_INTEROP_THREADS"] = "1"
    xla = rt.get("xla_flags", "") or ""
    if xla:
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " " + xla).strip()
    if rt.get("xla_cache", True) and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["TF_XLA_FLAGS"] = (env.get("TF_XLA_FLAGS", "") + f" --tf_xla_persistent_cache_directory={cache_dir}").strip()
    if cpus:
        env["CFCNET_CPUS"] = ",".join(map(str, cpus))
    return env


def configure_tf(rt: dict) -> dict:
    """Call right after importing TensorFlow inside a worker. Returns a short device summary."""
    import tensorflow as tf
    if os.environ.get("CFCNET_CPUS") and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {int(c) for c in os.environ["CFCNET_CPUS"].split(",")})
        except OSError:
            pass
    tf.config.threading.set_intra_op_parallelism_threads(int(rt.get("cpu_threads_per_worker", 2)))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    gpus = tf.config.list_physical_devices("GPU")
    mem = int(rt.get("memory_limit_mb", 0) or 0)
    for g in gpus:
        try:
            if mem > 0:
                tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])
            else:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:                              # already initialised
            pass
    tf.config.experimental.enable_tensor_float_32_execution(bool(rt.get("tf32", False)))
    if rt.get("deterministic", False):
        tf.config.experimental.enable_op_determinism()
    name = ""
    if gpus:
        try:
            name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "")
        except Exception:
            pass
    return {"gpus": len(gpus), "gpu_name": name}


def cpu_slices(n_workers: int, threads: int) -> list[list[int] | None]:
    """Disjoint CPU sets per worker when the host has enough cores, else no pinning."""
    try:
        cores = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return [None] * n_workers
    if len(cores) < n_workers * threads:
        return [None] * n_workers
    return [cores[i * threads:(i + 1) * threads] for i in range(n_workers)]

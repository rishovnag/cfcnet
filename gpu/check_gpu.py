#!/usr/bin/env python
"""Check the GPU software stack before a long run.

    python gpu/check_gpu.py

Reports TensorFlow / Keras / ncps versions, the CUDA and cuDNN builds, the visible GPUs, whether
MPS is running, then trains a tiny CfC (XLA), LTC (XLA) and GRU (cuDNN) for a few epochs on the
GPU through the same code path as the workers, and checks the result against the reference
`model.fit` path.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")


def main():
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import keras
    import ncps
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | ncps {getattr(ncps, '__version__', '?')}")
    info = tf.sysconfig.get_build_info()
    print(f"CUDA build {info.get('cuda_version')} | cuDNN build {info.get('cudnn_version')}")
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        d = tf.config.experimental.get_device_details(g)
        print(f"GPU {g.name}: {d.get('device_name')} (compute capability {d.get('compute_capability')})")
    if not gpus:
        print("WARNING: no GPU visible to TensorFlow (install tensorflow[and-cuda] matching the driver)")
    try:
        r = subprocess.run(["bash", str(ROOT / "gpu" / "mps.sh"), "status"], capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "MPS: unknown")
    except Exception:
        print("MPS: unknown")

    from cfcnet.data import fixed_origin_window, make_pairs, prepare_window
    from cfcnet.fast import train_pack
    from cfcnet.train import run_one
    rng = np.random.default_rng(0)
    n = 600
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Date": pd.bdate_range("2010-01-01", periods=n), "Open": close * 1.001, "High": close * 1.004,
                       "Low": close * 0.996, "Close": close, "AdjClose": close, "Volume": rng.integers(1, 1e6, n) * 1.0})
    pairs = make_pairs(df, "ohlc", "levels")
    prep = prepare_window(pairs, *fixed_origin_window(pairs.n, 0.1))
    for arch in ("cfc", "ltc", "gru"):
        t0 = time.time()
        fast = train_pack(prep, arch, 32, [0, 1], epochs=5)
        tf_ = time.time() - t0
        ref = run_one(prep, arch, 32, 0, epochs=5)
        d = abs(fast[0]["last_loss"] - ref["last_loss"]) / max(abs(ref["last_loss"]), 1e-12)
        print(f"{arch}: fast path {tf_:.1f}s for 2 seeds x 5 epochs (incl. compile), xla={fast[0]['xla']}; "
              f"relative difference to model.fit {d:.1e} {'OK' if d < 1e-3 else 'CHECK'}")
    print("check complete")


if __name__ == "__main__":
    main()

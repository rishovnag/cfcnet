#!/usr/bin/env python
"""Serial inference-latency benchmark (the publishable numbers for the parameters/latency table).

Timings recorded by the parallel runner are contended (many workers share the GPU), so the
paper's latency figures come from this script, run ALONE on an otherwise idle machine:

    python scripts/12_latency.py --device cpu    # every architecture x width x target, on the CPU
    python scripts/12_latency.py               # on GPU 0, if one is available

For each configuration it builds the model (weights do not affect latency), compiles the forward
pass, warms it up, then times `--repeats` full passes over the NIFTY 100 fixed-origin test window
(warm-up + test sessions) and reports the median and the 10th/90th percentiles.
Writes runs/latency_cpu.csv (--device cpu; the numbers reported in the paper) or runs/latency_gpu.csv.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    out = Path(args.out)
    locks = list((out / "locks").glob("*.lock")) if (out / "locks").exists() else []
    if locks:
        print(f"WARNING: {len(locks)} jobs appear to be running; latency must be measured on an idle GPU")

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from cfcnet import cli
    from cfcnet.data import fixed_origin_window, make_pairs, prepare_window
    from cfcnet.io import load_config, load_frame
    from cfcnet.models import ARCHS, build_model, label
    from cfcnet.train import set_seeds

    cfg = load_config(args.config)
    pr = cfg["protocol"]
    df = load_frame(cfg, "NIFTY100", args.data_dir)
    import platform
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    dev = tf.config.list_physical_devices("GPU")
    name = tf.config.experimental.get_device_details(dev[0]).get("device_name", "GPU") if dev else "CPU"
    rows = []
    for target in pr["targets"]:
        pairs = make_pairs(df, target, "levels", pr["feature_cols"], pr["add_weekday"])
        prep = prepare_window(pairs, *fixed_origin_window(pairs.n, pr["test_frac"]), pr["warmup"])
        x = tf.constant(prep.test_x)
        for width in ("default", "ablation"):
            units = cli.units_for(cfg, target, width)
            for arch in ARCHS:
                set_seeds(0)
                tf.keras.backend.clear_session()
                m = build_model(arch, x.shape[2], units=units, motor=cli.motor_for(target))
                f = tf.function(lambda inp, m=m: m(inp, training=False))
                for _ in range(args.warmup):
                    f(x).numpy()
                ts = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    f(x).numpy()
                    ts.append(1000 * (time.perf_counter() - t0))
                ts = np.array(ts)
                n_test = len(prep.test_level)
                rows.append({"target": target, "arch": arch, "units": units, "motor": cli.motor_for(target),
                             "model": label(arch, units, cli.motor_for(target)), "n_params": m.count_params(),
                             "sessions_per_pass": int(x.shape[1]), "n_test": n_test,
                             "median_ms": np.median(ts), "p10_ms": np.percentile(ts, 10),
                             "p90_ms": np.percentile(ts, 90), "ms_per_session": np.median(ts) / n_test,
                             "device": name, "repeats": args.repeats, "platform": cpu if name == "CPU" else name,
                             "tf_version": tf.__version__})
                print(f"{target:8s} {rows[-1]['model']:11s} {rows[-1]['median_ms']:9.2f} ms/pass "
                      f"({rows[-1]['ms_per_session']:.3f} ms/session)", flush=True)
    fn = out / ("latency_gpu.csv" if args.device == "gpu" else "latency_cpu.csv")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(fn, index=False)
    print("wrote", fn)


if __name__ == "__main__":
    main()

"""Train one model on one window and evaluate it (Algorithms 1 and 2 of the paper).

`run_one` is the reference path (Keras `model.fit`, one model per call). The optional GPU path
(`cfcnet.fast`, used by gpu/) trains the same models with a compiled loop and reuses
`finalize_run`, so both paths produce identical log rows, predictions and loss histories.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .data import Prepared
from .metrics import evaluate
from .models import build_model


def set_seeds(seed: int, deterministic: bool = False) -> None:
    import tensorflow as tf
    from tensorflow import keras
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


# Relative seconds per epoch per training session (CPU, one thread; CfC(80,4) 0.80 s, LTC 2.0 s,
# LSTM/GRU ~0.35 s on NIFTY 100). Only used to submit the longest runs first.
ARCH_COST = {"ltc": 2.5, "cfc": 1.0, "lstm": 0.45, "gru": 0.45}


def expected_cost(arch: str, prep: Prepared) -> float:
    return ARCH_COST.get(arch, 1.0) * prep.train_x.shape[1]


def run_one(prep: Prepared, arch: str, units: int, seed: int, *, epochs: int = 400,
            learning_rate: float = 1e-3, batch_size: int = 1, jit_compile: bool = False,
            target_names=None, run_id: str | None = None, out_dir: Path | None = None,
            verbose: int = 0) -> dict:
    """Train on prep.train_* and evaluate on prep.test_* (warm-up rows dropped).

    Returns a flat dict of metrics and timings. If out_dir is given, the scaled and
    unscaled test predictions are saved to out_dir/predictions/<run_id>.npz and the
    per-epoch loss history to out_dir/histories/<run_id>.json."""
    import tensorflow as tf
    from tensorflow import keras

    K = prep.train_y.shape[-1]          # 1 (adjusted close) or 4 (joint OHLC)
    set_seeds(seed)
    keras.backend.clear_session()
    model = build_model(arch, prep.train_x.shape[2], units=units, motor=K,
                        learning_rate=learning_rate, jit_compile=jit_compile)

    t0 = time.time()
    hist = model.fit(prep.train_x, prep.train_y, batch_size=batch_size, epochs=epochs,
                     verbose=verbose, shuffle=False)
    train_time = time.time() - t0
    losses = [float(v) for v in hist.history["loss"]]

    t0 = time.time()
    pred = model(prep.test_x, training=False).numpy()[0]
    test_time = time.time() - t0
    return finalize_run(prep, pred, losses, n_params=model.count_params(), train_time=train_time,
                        test_time=test_time, tf_version=tf.__version__, target_names=target_names,
                        run_id=run_id, out_dir=out_dir)


def finalize_run(prep: Prepared, pred: np.ndarray, losses: list, *, n_params: int, train_time: float,
                 test_time: float, tf_version: str | None = None, target_names=None,
                 run_id: str | None = None, out_dir: Path | None = None) -> dict:
    """Evaluate the test pass `pred` (model space, warm-up rows included), write the predictions and
    the loss history, and return the log row."""
    if tf_version is None:
        import tensorflow as tf
        tf_version = tf.__version__
    losses = [float(v) for v in losses]
    P = prep.warmup
    p_model = pred[P:, :]
    y = prep.test_level
    n_test = len(y)
    row = {"n_params": int(n_params), "n_train": int(prep.train_x.shape[1]),
           "n_test": int(n_test), "train_start": prep.train_dates[0], "train_end": prep.train_dates[1],
           "test_start": prep.test_dates[0], "test_end": prep.test_dates[1],
           "last_loss": losses[-1], "train_time_s": train_time, "test_time_s": test_time,
           "latency_ms_per_day": 1000.0 * test_time / n_test, "tf_version": tf_version, "status": "ok"}
    with np.errstate(all="ignore"):
        p = prep.pred_to_eval(p_model) if np.isfinite(p_model).all() else p_model
    if np.isfinite(p).all() and np.isfinite(losses[-1]):
        with np.errstate(all="ignore"):
            r_pred = prep.returns_of(p)
        r_true = prep.returns_of(y)
        row.update(evaluate(y, p, prep.test_rw, prep.close_col, names=target_names,
                            ret_true=r_true, ret_pred=r_pred if np.isfinite(r_pred).all() else None))
    else:
        row["status"] = "diverged"   # non-finite loss or predictions; kept in the log, excluded from means

    if out_dir is not None and run_id is not None:
        out_dir = Path(out_dir)
        _, b, c = prep.window
        ok = np.isfinite(p).all()
        np.savez_compressed(
            out_dir / "predictions" / f"{run_id}.npz",
            dates=np.asarray(prep.pairs.target_dates[b:c]).astype("datetime64[D]").astype(str),
            pred_scaled=p, actual_scaled=y, rw_scaled=prep.test_rw,          # scaled levels (evaluation space)
            pred=prep.l_scaler.inverse_transform(p) if ok else p,             # index points
            actual=prep.pairs.Y_level[b:c], warmup_pred_model=pred[:P, :])
        with open(out_dir / "histories" / f"{run_id}.json", "w") as fh:
            json.dump({"loss": losses}, fh)
    return row

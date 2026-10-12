"""GPU training path (developed on an NVIDIA H100): the same models as `train.run_one`, trained with a compiled loop.

What is optimised, and why it does not change the experiment
------------------------------------------------------------
* **Same models.** Every model is built by `models.build_model` (ncps `AutoNCP` wiring + `CfC`/`LTC`,
  or Keras `LSTM`/`GRU` + `Dense`) after the same `set_seeds(seed)` / `clear_session()` sequence as
  `run_one`, so the initial weights are identical.
* **Same optimisation.** One Adam step per epoch on the MSE of the full training sequence
  (batch dimension 1), exactly what `model.fit(batch_size=1)` does with one sample.
* **Compiled training loop.** All epochs run inside one `tf.function`; for CfC and LTC the whole
  loop is XLA-compiled (`jit_compile=True`), which removes per-step Python and kernel-launch
  overhead from the 400 x T recurrent steps. LSTM and GRU stay on the cuDNN fused-RNN kernel
  (which XLA cannot compile), still inside one graph-mode loop.
* **Seed packing (optional, `pack > 1`).** Several seeds of the same cell are trained in one
  graph. Their parameters are disjoint and the loss is the SUM of the per-model MSEs, so the
  gradient for each model is exactly its own gradient; Adam is element-wise, so one optimizer over
  the union of variables performs exactly the per-model updates. Packing therefore reproduces the
  separate runs (up to floating-point reduction order) while amortising launch and compilation
  overhead. `tests/test_fast.py` checks this against `run_one`.

Timing columns from this path are recorded under contention (several runs share a GPU); the
publishable inference latency comes from the serial benchmark `scripts/12_latency.py`.
"""
from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np

from .data import Prepared
from .models import build_model
from .train import finalize_run, set_seeds

NCPS_ARCHS = ("cfc", "ltc")


def build_pack(arch: str, n_features: int, units: int, motor: int, seeds, learning_rate: float = 1e-3):
    """Build one model per seed with exactly the initialisation sequence of `run_one`."""
    from tensorflow import keras
    models = []
    for s in seeds:
        set_seeds(int(s))
        keras.backend.clear_session()
        m = build_model(arch, n_features, units=units, motor=motor, learning_rate=learning_rate)
        models.append(m)
    return models


def _make_train_fn(models, x, y, epochs: int, learning_rate: float, xla: bool, whole_loop: bool):
    import tensorflow as tf
    from tensorflow import keras

    variables = [v for m in models for v in m.trainable_variables]
    opt = keras.optimizers.Adam(learning_rate)
    opt.build(variables)

    def one_step():
        with tf.GradientTape() as tape:
            losses = [tf.reduce_mean(tf.square(m(x, training=True) - y)) for m in models]
            total = tf.add_n(losses)          # disjoint parameters: d total / d theta_k = d loss_k / d theta_k
        grads = tape.gradient(total, variables)
        opt.apply(grads, variables)
        return tf.stack(losses)

    if whole_loop:
        @tf.function(jit_compile=xla, reduce_retracing=True)
        def train_all():
            ta = tf.TensorArray(tf.float32, size=epochs)
            for e in tf.range(epochs):
                ta = ta.write(e, one_step())
            return ta.stack()
        return lambda: train_all().numpy()

    step = tf.function(one_step, jit_compile=xla, reduce_retracing=True)

    def loop():
        out = np.empty((epochs, len(models)), np.float32)
        for e in range(epochs):
            out[e] = step().numpy()
        return out
    return loop


def train_pack(prep: Prepared, arch: str, units: int, seeds, *, epochs: int = 400, learning_rate: float = 1e-3,
               xla: bool | None = None, whole_loop: bool = True, target_names=None, run_ids=None,
               out_dir: Path | None = None, extra: dict | None = None) -> list[dict]:
    """Train `len(seeds)` independent models of one cell together and evaluate each.

    xla=None -> XLA for CfC/LTC, graph mode (cuDNN) for LSTM/GRU.
    Returns one log row per seed, in the order of `seeds`."""
    import tensorflow as tf

    xla = (arch in NCPS_ARCHS) if xla is None else xla
    K = prep.train_y.shape[-1]
    models = build_pack(arch, prep.train_x.shape[2], units, K, seeds, learning_rate)
    x = tf.constant(prep.train_x)
    y = tf.constant(prep.train_y)
    train = _make_train_fn(models, x, y, epochs, learning_rate, xla, whole_loop)

    t0 = time.time()
    losses = train()                                      # (epochs, n_models), loss before each update
    train_time = (time.time() - t0) / len(models)

    xt = tf.constant(prep.test_x)
    rows = []
    for k, (m, seed) in enumerate(zip(models, seeds)):
        predict = tf.function(lambda inp, m=m: m(inp, training=False))
        t0 = time.time()
        pred = predict(xt).numpy()[0]
        test_time = time.time() - t0
        row = finalize_run(prep, pred, [float(v) for v in losses[:, k]], n_params=m.count_params(),
                           train_time=train_time, test_time=test_time, target_names=target_names,
                           run_id=None if run_ids is None else run_ids[k], out_dir=out_dir)
        row.update({"pack": len(models), "xla": bool(xla), "timing_contended": True,
                    "host": socket.gethostname(), **(extra or {})})
        rows.append(row)
    return rows

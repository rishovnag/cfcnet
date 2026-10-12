"""Model factory for the four architectures compared in the paper.

CfC and LTC follow the construction used in the original notebooks exactly:

    wiring = wirings.AutoNCP(units, motor)          # e.g. AutoNCP(80, 4)
    model  = keras.Sequential([InputLayer((None, n_features)),
                               CfC(wiring, return_sequences=True)])   # or LTC(...)

The `motor` neurons of the AutoNCP wiring ARE the outputs (4 for joint OHLC, 1 for the
single-output adjusted-close task); there is no extra readout layer, and all motor neurons
share the wiring's inter and command neurons (one shared latent state). ncps >= 1.0 runs LTC
with its built-in semi-implicit fused solver (the only solver exposed by the public API).

LSTM and GRU use one recurrent layer of `units` cells followed by Dense(motor), as in the
notebooks. Parameter counts at 7 input channels reproduce Table 7 of the paper:
CfC(80,4) 19,736; LTC(80,4) 28,102; LSTM-80 28,484; GRU-80 21,684;
CfC(28,4) 2,504; LTC(28,4) 4,026.
"""
from __future__ import annotations

ARCHS = ("cfc", "ltc", "lstm", "gru")


def _ncps_layers():
    """Keras-3 API (ncps.keras) with a fallback to the older ncps.tf module."""
    from ncps import wirings
    try:
        from ncps.keras import CfC, LTC
    except ImportError:  # ncps < 1.0 / Keras 2
        from ncps.tf import CfC, LTC
    return wirings, CfC, LTC


def build_model(arch: str, n_features: int, units: int = 80, motor: int = 4,
                learning_rate: float = 1e-3, jit_compile: bool = False):
    from tensorflow import keras

    arch = arch.lower()
    if arch in ("cfc", "ltc"):
        wirings, CfC, LTC = _ncps_layers()
        wiring = wirings.AutoNCP(units, motor)
        cell = CfC if arch == "cfc" else LTC
        layers = [cell(wiring, return_sequences=True)]
    elif arch == "lstm":
        layers = [keras.layers.LSTM(units, return_sequences=True), keras.layers.Dense(motor)]
    elif arch == "gru":
        layers = [keras.layers.GRU(units, return_sequences=True), keras.layers.Dense(motor)]
    else:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHS}")

    model = keras.models.Sequential([keras.layers.InputLayer(shape=(None, n_features))] + layers)
    compile_kw = {}
    if jit_compile and arch in ("cfc", "ltc"):
        compile_kw["jit_compile"] = True   # XLA helps CfC/LTC; LSTM/GRU stay on cuDNN kernels
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mean_squared_error", **compile_kw)
    return model


def label(arch: str, units: int, motor: int) -> str:
    """Display name used in the paper's tables."""
    return f"{arch.upper()} ({units},{motor})" if arch in ("cfc", "ltc") else f"{arch.upper()}-{units}"

"""Forecast-accuracy measures.

All measures are computed on the scaled targets of the test window (warm-up rows removed).

* joint R^2 / RMSE / MAE pool all K target columns (4 * n_test values for OHLC);
* per-column R^2 / RMSE / MAE are logged for every run;
* MASE = MAE / MAE_RW, where RW is the random walk y_hat_{t+1} = y_t on the same window,
  so MASE < 1 means the model beats the random walk (Hyndman & Koehler, 2006);
* r2_ret = R^2 of the implied next-session log returns (relative to C_t, or AdjC_t), a
  scale-free measure that is not inflated by the persistence of price levels;
* DA = share of sessions on which the forecast close change (C_hat_{t+1} - C_t) has the same
  sign as the realised change (C_{t+1} - C_t). Sessions with no realised change are excluded.

For one-step forecasts the MAE of the implied change (y_hat_{t+1} - y_t) equals the level
MAE because y_t is known at the forecast origin, so MASE is its scale-free version.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression(y: np.ndarray, p: np.ndarray) -> dict:
    a, b = y.reshape(-1), p.reshape(-1)
    return {"r2": float(r2_score(a, b)), "rmse": float(np.sqrt(mean_squared_error(a, b))),
            "mae": float(mean_absolute_error(a, b))}


def directional_accuracy(y: np.ndarray, p: np.ndarray, prev: np.ndarray, col: int) -> float:
    real = np.sign(y[:, col] - prev[:, col])
    pred = np.sign(p[:, col] - prev[:, col])
    keep = real != 0
    return float(np.mean(real[keep] == pred[keep])) if keep.any() else float("nan")


def evaluate(y: np.ndarray, p: np.ndarray, rw: np.ndarray, close_col: int, names=None,
             ret_true: np.ndarray | None = None, ret_pred: np.ndarray | None = None) -> dict:
    """y, p, rw: (n_test, K) arrays of scaled LEVELS (the common evaluation space).
    ret_true / ret_pred: optional log returns relative to the forecast origin, for the
    scale-free returns R^2 (r2_ret), which does not depend on the level of the series."""
    out = {f"{k}_joint": v for k, v in regression(y, p).items()}
    if ret_true is not None and ret_pred is not None:
        out["r2_ret"] = regression(ret_true, ret_pred)["r2"]
    rw_mae = float(np.mean(np.abs(y - rw)))
    out["mase"] = out["mae_joint"] / rw_mae if rw_mae > 0 else float("nan")
    out["da"] = directional_accuracy(y, p, rw, close_col)
    names = names or [f"c{i}" for i in range(y.shape[1])]
    for i, nm in enumerate(names):
        for k, v in regression(y[:, i], p[:, i]).items():
            out[f"{k}_{nm.lower()}"] = v
    return out

"""Benchmark forecasts evaluated on exactly the same test targets and scaling as the networks.

* Random walk: y_hat_{t+1} = y_t (the current session's OHLC carried forward).
* Close-anchored naive: every component of y_{t+1} forecast by the current close C_t (AdjC_t for
  the adjusted-close target, where it coincides with the random walk). For the joint OHLC target
  this is the stronger naive benchmark: the next open is far closer to C_t than to O_t.
* ARIMA: one model per target column, fitted on the training targets with (p, d, q) chosen by
  AIC over a small grid (d = 1 with drift by default). Test forecasts are one-step-ahead with the
  parameters held fixed and the information set updated with every new session, i.e. the
  forecast of y_{t+1} uses y_1..y_t only.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np

from .data import Pairs


def random_walk(p: Pairs, b: int, c: int) -> np.ndarray:
    """Raw-scale RW forecasts for test pairs [b, c)."""
    return p.Y_prev_level[b:c].copy()


def naive_close(p: Pairs, b: int, c: int) -> np.ndarray:
    """Raw-scale close-anchored naive forecasts for test pairs [b, c): C_t for every column."""
    return np.repeat(p.base[b:c], p.Y_level.shape[1], axis=1)


def _fit_best_arima(y_train: np.ndarray, ps, qs, d, trend):
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    for pp, qq in itertools.product(ps, qs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(y_train, order=(pp, d, qq), trend=trend).fit()
            if best is None or res.aic < best[0].aic:
                best = (res, (pp, d, qq))
        except Exception:  # non-invertible / failed fits are skipped
            continue
    if best is None:
        raise RuntimeError("no ARIMA order could be fitted")
    return best


def arima_column(series: np.ndarray, n_train: int, ps=(0, 1, 2), qs=(0, 1, 2), d=1, trend="t"):
    """One target column: fit on series[:n_train] (order by AIC), then one-step-ahead forecasts of
    series[n_train:] with the parameters held fixed. Returns (forecasts, order)."""
    res, order = _fit_best_arima(series[:n_train], ps, qs, d, trend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = res.apply(series)                       # same parameters, extended data
        pred = full.get_prediction(start=n_train, end=len(series) - 1, dynamic=False).predicted_mean
    return np.asarray(pred), order


def arima(p: Pairs, a: int, b: int, c: int, ps=(0, 1, 2), qs=(0, 1, 2), d=1, trend="t"):
    """Raw-scale one-step-ahead ARIMA forecasts for test pairs [b, c), plus the chosen orders.

    The ARIMA is fitted on the target series itself: the target column sequence
    y_{a+2}, ..., y_{c+1} (pair targets) is one contiguous series, so the forecast for pair j
    conditions on targets of pairs < j, i.e. on y_1..y_t for target y_{t+1}."""
    cols = [arima_column(p.Y_level[a:c, k], b - a, ps, qs, d, trend) for k in range(p.Y_level.shape[1])]
    return np.stack([pred for pred, _ in cols], axis=1), [order for _, order in cols]

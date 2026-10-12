"""Statistical tests: Diebold-Mariano with the Harvey-Leybourne-Newbold correction,
unit-root / stationarity diagnostics, and Engle-Granger cointegration."""
from __future__ import annotations

import numpy as np
from scipy import stats as scistats


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1):
    """Diebold & Mariano (1995) test of equal predictive accuracy with the
    Harvey, Leybourne & Newbold (1997) small-sample correction.

    loss_a, loss_b: per-period losses (here: daily joint squared error averaged over the
    OHLC columns). A negative statistic means model A has the lower loss.
    Returns (statistic, two-sided p-value from Student t with T-1 d.o.f.)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T, dbar = len(d), d.mean()
    var = ((d - dbar) ** 2).mean()
    for k in range(1, h):
        var += 2 * ((d[k:] - dbar) * (d[:-k] - dbar)).mean()
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / T)
    dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(dm), float(2 * scistats.t.sf(abs(dm), df=T - 1))


def joint_sq_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Daily joint squared error: mean over the target columns."""
    return ((y - p) ** 2).mean(axis=1)


def unit_root_table(series: np.ndarray) -> dict:
    import warnings
    from statsmodels.tsa.stattools import adfuller, kpss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # API FutureWarnings; KPSS p-values are truncated at 0.01 / 0.10
        lv, fd = adfuller(series, autolag="AIC"), adfuller(np.diff(series), autolag="AIC")
        kp = kpss(series, regression="c", nlags="auto")
    return {"adf_level_stat": lv[0], "adf_level_p": lv[1], "adf_diff_stat": fd[0], "adf_diff_p": fd[1],
            "kpss_level_stat": kp[0], "kpss_level_p": kp[1]}


def engle_granger(y: np.ndarray, x: np.ndarray) -> float:
    """p-value of the Engle-Granger cointegration test of y on x."""
    from statsmodels.tsa.stattools import coint
    return float(coint(y, x)[1])

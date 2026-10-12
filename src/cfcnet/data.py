"""Data loading, one-step-ahead pairing, chronological splits and train-only scaling.

Information timing (Section 2 of the paper)
-------------------------------------------
At the close of session t we observe u_t = (O_t, H_t, L_t, V_t, AdjC_t, C_t, d_t).
The target is a quantity of the NEXT session, t+1. Two targets are studied:

* ``adjclose``  y_{t+1} = AdjC_{t+1}                              (single output)
* ``ohlc``      y_{t+1} = (O_{t+1}, H_{t+1}, L_{t+1}, C_{t+1})     (joint, four outputs)

and two representations of inputs and targets:

* ``levels``   inputs are the raw session values; targets are next-session price levels;
* ``returns``  inputs are log changes relative to the previous session
               (log O_t/C_{t-1}, log H_t/C_{t-1}, log L_t/C_{t-1}, log V-change,
               log AdjC_t/AdjC_{t-1}, log C_t/C_{t-1}); targets are next-session log returns
               relative to the forecast origin (log P_{t+1}/C_t for OHLC, log AdjC_{t+1}/AdjC_t).
               Level forecasts are recovered as P_hat_{t+1} = base_t * exp(r_hat).

Pairs are (u_t, y_{t+1}). Session 1 is dropped in BOTH representations, so that
the two representations share identical pairs, windows and test dates.
No component of y_{t+1} is ever an input at or before step t
(``tests/test_pipeline.py`` checks this).

Robustness protocols only (``timing="same_day"``, levels only): pairs (u_t, y_t), i.e. the target
is a quantity of the SAME session whose prices are inputs. It is not a forecast and is used solely
by scripts/10_primary_window.py for the robustness analysis of Section 6.4.
``prepare_window(..., scaler_fit="full")`` likewise fits the scalers on the whole series (a
look-ahead) for that analysis; every forecasting experiment uses the defaults.

All metrics are computed on the LEVEL scale, after Min-Max scaling with a scaler fitted on
the training-window levels, so that models trained on levels and on returns and the
benchmarks are evaluated on exactly the same numbers.

Index arithmetic is on PAIR indices j = 0..n-1; windows are half-open [a, b) for training
pairs and [b, c) for test pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DEFAULT_FEATURES = ["Open", "High", "Low", "Volume", "AdjClose", "Close"]
OHLC = ["Open", "High", "Low", "Close"]
TARGETS = {"ohlc": OHLC, "adjclose": ["AdjClose"]}
REPRESENTATIONS = ("levels", "returns")


# --------------------------------------------------------------------------- loading
def _parse_volume(v):
    """Some vendor files write volumes as strings such as '2.86B' or '512.3M'."""
    if isinstance(v, (int, float, np.number)):
        return float(v)
    v = str(v).strip().replace(",", "")
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}
    if v and v[-1].upper() in mult:
        return float(v[:-1]) * mult[v[-1].upper()]
    try:
        return float(v)
    except ValueError:
        return np.nan


def flat_ohlc(df: pd.DataFrame) -> pd.Series:
    """Sessions on which the vendor reports a single price (O = H = L = C)."""
    return (df.Open == df.High) & (df.High == df.Low) & (df.Low == df.Close)


def genuine_ohlc_start(df: pd.DataFrame, window: int = 252, max_share: float = 0.05):
    """First session after the last trailing `window`-session block in which more than
    `max_share` of sessions are flat (O = H = L = C). Isolated flat sessions later in the
    sample do not move the start. Returns a Timestamp (the first date if no such block)."""
    flat = flat_ohlc(df).to_numpy()
    share = pd.Series(flat.astype(float)).rolling(window, min_periods=1).mean().to_numpy()
    bad = np.flatnonzero(share > max_share)
    if len(bad) == 0:
        return df.Date.iloc[0]
    last_flat = np.flatnonzero(flat[: bad[-1] + 1])[-1]      # last flat session of the dense block
    return df.Date.iloc[min(last_flat + 1, len(df) - 1)]


def load_index(path: str | Path, start: str | None = None) -> pd.DataFrame:
    """Load one index file into a frame with columns Date, Open, High, Low, Close, AdjClose, Volume.

    * Yahoo Finance CSVs (Date, Open, High, Low, Close, Adj Close, Volume), as written by
      data/download_data.py, are used as is.
    * An SSE workbook (columns date, close, open, high, low, volume) is also accepted;
      it has no adjusted series, so AdjClose := Close.
    * Sessions with a missing field or a non-positive price are dropped.
    * `start`: None (keep everything), an ISO date, or "auto_ohlc" (genuine_ohlc_start).
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
        assert raw.shape[1] == 6, f"unexpected SSE layout: {list(raw.columns)}"
        df = pd.DataFrame({"Date": pd.to_datetime(raw.iloc[:, 0]), "Close": raw.iloc[:, 1].astype(float),
                           "Open": raw.iloc[:, 2].astype(float), "High": raw.iloc[:, 3].astype(float),
                           "Low": raw.iloc[:, 4].astype(float), "Volume": raw.iloc[:, 5].map(_parse_volume)})
        df["AdjClose"] = df["Close"]
    else:
        df = pd.read_csv(path).rename(columns={"Adj Close": "AdjClose"})
        df["Date"] = pd.to_datetime(df["Date"])
        if "AdjClose" not in df:
            df["AdjClose"] = df["Close"]
    df = df[["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]].dropna()
    df = df[(df[["Open", "High", "Low", "Close", "AdjClose"]] > 0).all(axis=1)]
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    if start == "auto_ohlc":
        start = genuine_ohlc_start(df)
    if start is not None:
        df = df[df["Date"] >= pd.Timestamp(start)].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- pairing
@dataclass
class Pairs:
    """One-step-ahead supervised pairs (u_t, y_{t+1}) in a given representation."""
    X_raw: np.ndarray          # (n, 6)  model-space input channels (scaled later)
    weekday: np.ndarray        # (n, 1)  d_t, never scaled
    Y_raw: np.ndarray          # (n, K)  model-space target (levels or log returns)
    Y_level: np.ndarray        # (n, K)  next-session levels y_{t+1}
    Y_prev_level: np.ndarray   # (n, K)  y_t (the random-walk forecast of y_{t+1})
    base: np.ndarray           # (n, 1)  reference level for returns (C_t, or AdjC_t)
    input_dates: np.ndarray
    target_dates: np.ndarray
    close_col: int
    target: str = "ohlc"
    representation: str = "levels"
    target_cols: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.Y_raw)

    def to_level(self, y_model_raw: np.ndarray, rows: slice) -> np.ndarray:
        """Convert model-space target values (unscaled) for pair rows `rows` to levels."""
        if self.representation == "levels":
            return y_model_raw
        return self.base[rows] * np.exp(y_model_raw)


def make_pairs(df: pd.DataFrame, target: str = "ohlc", representation: str = "levels",
               feature_cols=DEFAULT_FEATURES, add_weekday: bool = True, timing: str = "next_day") -> Pairs:
    assert target in TARGETS and representation in REPRESENTATIONS
    assert timing in ("next_day", "same_day")
    if timing == "same_day":
        return _same_day_pairs(df, target, feature_cols, add_weekday)
    tcols = TARGETS[target]
    P = {c: df[c].to_numpy(dtype=np.float64) for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]}
    N = len(df)
    t = np.arange(1, N - 1)          # forecast origins (0-based sessions); session 0 dropped
    nxt = t + 1
    lv = np.stack([P[c] for c in tcols], axis=1)
    base = (P["Close"] if target == "ohlc" else P["AdjClose"])[t].reshape(-1, 1)

    if representation == "levels":
        X = np.stack([P[c][t] for c in feature_cols], axis=1)
        Y = lv[nxt]
    else:
        prevC, prevA = P["Close"][t - 1], P["AdjClose"][t - 1]
        feats = {"Open": np.log(P["Open"][t] / prevC), "High": np.log(P["High"][t] / prevC),
                 "Low": np.log(P["Low"][t] / prevC), "Close": np.log(P["Close"][t] / prevC),
                 "AdjClose": np.log(P["AdjClose"][t] / prevA),
                 "Volume": np.log1p(P["Volume"][t]) - np.log1p(P["Volume"][t - 1])}
        X = np.stack([feats[c] for c in feature_cols], axis=1)
        Y = np.log(lv[nxt] / base)
    wd = df["Date"].dt.weekday.to_numpy(dtype=np.float64)[t].reshape(-1, 1)
    if not add_weekday:
        wd = np.zeros((len(t), 0))
    return Pairs(X_raw=np.array(X, copy=True), weekday=wd, Y_raw=np.array(Y, copy=True),
                 Y_level=np.array(lv[nxt], copy=True), Y_prev_level=np.array(lv[t], copy=True), base=base,
                 input_dates=df["Date"].to_numpy()[t], target_dates=df["Date"].to_numpy()[nxt],
                 close_col=len(tcols) - 1, target=target, representation=representation, target_cols=list(tcols))


def _same_day_pairs(df: pd.DataFrame, target: str, feature_cols, add_weekday: bool) -> Pairs:
    """Same-session task (levels; robustness protocols only): input u_t, target y_t of the SAME
    session. Uses the same sessions t = 1..N-2 as the next-day pairs so that windows and test dates
    line up with the forecasting experiments up to a one-session shift. The reference 'random walk'
    is y_{t-1}."""
    tcols = TARGETS[target]
    P = {c: df[c].to_numpy(dtype=np.float64) for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"]}
    N = len(df)
    t = np.arange(1, N - 1)
    lv = np.stack([P[c] for c in tcols], axis=1)
    X = np.stack([P[c][t] for c in feature_cols], axis=1)
    wd = df["Date"].dt.weekday.to_numpy(dtype=np.float64)[t].reshape(-1, 1)
    if not add_weekday:
        wd = np.zeros((len(t), 0))
    base = (P["Close"] if target == "ohlc" else P["AdjClose"])[t - 1].reshape(-1, 1)
    return Pairs(X_raw=X, weekday=wd, Y_raw=np.array(lv[t], copy=True), Y_level=np.array(lv[t], copy=True),
                 Y_prev_level=np.array(lv[t - 1], copy=True), base=base,
                 input_dates=df["Date"].to_numpy()[t], target_dates=df["Date"].to_numpy()[t],
                 close_col=len(tcols) - 1, target=target, representation="levels", target_cols=list(tcols))


# --------------------------------------------------------------------------- windows
def fixed_origin_window(n: int, test_frac: float) -> tuple[int, int, int]:
    """Chronological split identical to sklearn's train_test_split(shuffle=False)."""
    n_test = math.ceil(test_frac * n)
    return 0, n - n_test, n


def walk_forward_windows(n: int, train_w: int, test_w: int, step: int) -> list[tuple[int, int, int]]:
    """Rolling folds anchored to the END of the series (the most recent data is always tested)."""
    folds, test_end = [], n
    while test_end - test_w - train_w >= 0:
        test_start = test_end - test_w
        folds.append((test_start - train_w, test_start, test_end))
        test_end -= step
    return folds[::-1]


def history_windows(n: int, test_frac: float, lengths) -> list[tuple[int, int, int]]:
    """Fixed test window (the 95-5 partition) with training histories of varying length."""
    _, b, c = fixed_origin_window(n, test_frac)
    out = []
    for L in lengths:
        L = b if L is None else int(L)
        if L <= b:
            out.append((b - L, b, c))
    return sorted(set(out), key=lambda w: w[1] - w[0])


# --------------------------------------------------------------------------- scaling
@dataclass
class Prepared:
    train_x: np.ndarray        # (1, m, 7)  model space
    train_y: np.ndarray        # (1, m, K)  model space
    test_x: np.ndarray         # (1, P + n_test, 7)
    test_level: np.ndarray     # (n_test, K) scaled actual levels (evaluation space)
    test_rw: np.ndarray        # (n_test, K) scaled random-walk forecasts y_t
    f_scaler: MinMaxScaler
    t_scaler: MinMaxScaler     # model-space target scaler
    l_scaler: MinMaxScaler     # level scaler (evaluation space)
    pairs: Pairs
    window: tuple
    warmup: int
    train_dates: tuple
    test_dates: tuple
    close_col: int

    def pred_to_eval(self, pred_model_scaled: np.ndarray) -> np.ndarray:
        """Model output on the test rows (warm-up removed) -> scaled levels."""
        _, b, c = self.window
        raw = self.t_scaler.inverse_transform(pred_model_scaled)
        return self.l_scaler.transform(self.pairs.to_level(raw, slice(b, c)))

    def returns_of(self, level_scaled: np.ndarray) -> np.ndarray:
        """Scaled levels -> log returns relative to the forecast-origin base."""
        _, b, c = self.window
        return np.log(self.l_scaler.inverse_transform(level_scaled) / self.pairs.base[b:c])


def prepare_window(p: Pairs, a: int, b: int, c: int, warmup: int = 10, scaler_fit: str = "train") -> Prepared:
    """Fit all scalers on training pairs [a, b) ONLY and apply them to the test pairs [b, c).
    The test pass is preceded by `warmup` pairs from the end of the training window; their
    outputs are excluded from every metric.

    scaler_fit="full" fits the scalers on [a, c) instead (a look-ahead); it exists only for the
    robustness analysis of scripts/10_primary_window.py and is never used by a forecasting experiment."""
    assert 0 <= a < b < c <= p.n and b - a > warmup
    assert scaler_fit in ("train", "full")
    e = b if scaler_fit == "train" else c
    f_scaler = MinMaxScaler().fit(p.X_raw[a:e])
    t_scaler = MinMaxScaler().fit(p.Y_raw[a:e])
    l_scaler = MinMaxScaler().fit(p.Y_level[a:e])
    X = np.concatenate([f_scaler.transform(p.X_raw[a:c]), p.weekday[a:c]], axis=1)
    m = b - a
    as3 = lambda z: np.expand_dims(z, 0).astype(np.float32)
    d = lambda j: str(pd.Timestamp(p.target_dates[j]).date())
    return Prepared(
        train_x=as3(X[:m]), train_y=as3(t_scaler.transform(p.Y_raw[a:b])),
        test_x=as3(X[m - warmup:]),
        test_level=l_scaler.transform(p.Y_level[b:c]), test_rw=l_scaler.transform(p.Y_prev_level[b:c]),
        f_scaler=f_scaler, t_scaler=t_scaler, l_scaler=l_scaler, pairs=p, window=(a, b, c), warmup=warmup,
        train_dates=(d(a), d(b - 1)), test_dates=(d(b), d(c - 1)), close_col=p.close_col)


def regime_label(close: np.ndarray, bull: float = 0.10, bear: float = -0.10):
    """Label a test window by its annualised return (as in the walk-forward section)."""
    yrs = len(close) / 252.0
    ann_ret = (close[-1] / close[0]) ** (1.0 / yrs) - 1.0
    ann_vol = np.diff(np.log(close)).std(ddof=1) * np.sqrt(252)
    lab = "bull" if ann_ret >= bull else ("bear" if ann_ret <= bear else "sideways")
    return lab, float(ann_ret), float(ann_vol)

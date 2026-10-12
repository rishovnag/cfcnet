"""Leakage, representation, scaling and split tests. Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfcnet.cli import parse_seeds  # noqa: E402
from cfcnet.data import (fixed_origin_window, genuine_ohlc_start, history_windows, make_pairs,  # noqa: E402
                         prepare_window, walk_forward_windows)
from cfcnet.metrics import evaluate  # noqa: E402
from cfcnet.stats import dm_test  # noqa: E402

GRID = [(t, r) for t in ("adjclose", "ohlc") for r in ("levels", "returns")]


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"Date": pd.bdate_range("2010-01-01", periods=n), "Open": open_, "High": high,
                         "Low": low, "Close": close, "AdjClose": close, "Volume": rng.integers(1, 1e6, n).astype(float)})


@pytest.mark.parametrize("target,rep", GRID)
def test_targets_are_next_session(target, rep):
    df = synthetic()
    p = make_pairs(df, target, rep)
    assert p.n == len(df) - 2                           # session 0 dropped in both representations
    col = "Close" if target == "ohlc" else "AdjClose"
    np.testing.assert_allclose(p.Y_level[:, -1], df[col].values[2:])
    np.testing.assert_allclose(p.Y_prev_level[:, -1], df[col].values[1:-1])
    np.testing.assert_allclose(p.to_level(p.Y_raw, slice(0, p.n)), p.Y_level, rtol=1e-12)
    assert (p.target_dates > p.input_dates).all()


@pytest.mark.parametrize("target,rep", GRID)
def test_inputs_never_see_the_future(target, rep):
    """Changing session s must not change any input row whose forecast origin is before s,
    and must change the target of the pair whose target session is s."""
    df = synthetic()
    base = make_pairs(df, target, rep)
    for s in [5, 200, len(df) - 1]:
        d2 = df.copy()
        d2.loc[s, ["Open", "High", "Low", "Close", "AdjClose", "Volume"]] *= 1.5
        p2 = make_pairs(d2, target, rep)
        j_target = s - 2                                 # pair whose target session is s
        np.testing.assert_allclose(p2.X_raw[: j_target + 1], base.X_raw[: j_target + 1])
        assert not np.allclose(p2.Y_raw[j_target], base.Y_raw[j_target])


def test_same_windows_across_representations():
    df = synthetic()
    a = make_pairs(df, "ohlc", "levels")
    b = make_pairs(df, "ohlc", "returns")
    assert a.n == b.n and (a.target_dates == b.target_dates).all()
    np.testing.assert_allclose(a.Y_level, b.Y_level)


def test_chronological_split_matches_train_test_split():
    from sklearn.model_selection import train_test_split
    n = 4579
    tr, te = train_test_split(np.arange(n), test_size=0.05, shuffle=False)
    assert fixed_origin_window(n, 0.05) == (0, len(tr), n) and te[0] == len(tr)
    p = make_pairs(synthetic())
    prep = prepare_window(p, *fixed_origin_window(p.n, 0.05))
    assert pd.Timestamp(prep.train_dates[1]) < pd.Timestamp(prep.test_dates[0])


@pytest.mark.parametrize("target,rep", GRID)
def test_scalers_ignore_test_data(target, rep):
    p = make_pairs(synthetic(), target, rep)
    a, b, c = fixed_origin_window(p.n, 0.1)
    s1 = prepare_window(p, a, b, c)
    p.X_raw[b:] *= 50.0
    p.Y_raw[b:] *= 50.0
    p.Y_level[b:] *= 50.0
    s2 = prepare_window(p, a, b, c)
    for name in ["f_scaler", "t_scaler", "l_scaler"]:
        np.testing.assert_allclose(getattr(s1, name).data_max_, getattr(s2, name).data_max_)
    np.testing.assert_allclose(s1.train_x, s2.train_x)


def test_warmup_shapes_and_evaluation_space():
    p = make_pairs(synthetic(), "ohlc", "returns")
    a, b, c = fixed_origin_window(p.n, 0.05)
    prep = prepare_window(p, a, b, c, warmup=10)
    assert prep.train_x.shape == (1, b - a, 7) and prep.train_y.shape == (1, b - a, 4)
    assert prep.test_x.shape == (1, 10 + c - b, 7)
    np.testing.assert_allclose(prep.test_x[0, :10], prep.train_x[0, -10:])
    assert set(np.unique(prep.train_x[0, :, 6])) <= {0, 1, 2, 3, 4}
    # a perfect model-space forecast maps exactly onto the scaled test levels
    perfect = prep.t_scaler.transform(p.Y_raw[b:c])
    np.testing.assert_allclose(prep.pred_to_eval(perfect), prep.test_level, atol=1e-9)
    np.testing.assert_allclose(prep.test_rw[1:], prep.test_level[:-1], atol=1e-9)


def test_genuine_ohlc_start():
    df = synthetic(800)
    for c in ["Open", "High", "Low"]:
        df.loc[:299, c] = df.loc[:299, "Close"]
        df.loc[600, c] = df.loc[600, "Close"]                       # isolated flat day is ignored
    assert genuine_ohlc_start(df) == df.Date.iloc[300]


def test_windows_and_seeds():
    folds = walk_forward_windows(4579, 1260, 252, 252)
    assert len(folds) == 13 and folds[-1][2] == 4579
    assert all(folds[i][2] == folds[i + 1][1] for i in range(len(folds) - 1))
    hw = history_windows(4579, 0.05, [1260, None])
    assert hw[0][1] == hw[-1][1] and hw[-1][0] == 0
    assert parse_seeds("3", 30) == [0, 1, 2] and parse_seeds("0-0", 30) == [0] and parse_seeds(None, 2) == [0, 1]
    with pytest.raises(ValueError):
        parse_seeds("0", 30)


def test_metrics_and_dm():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(200, 4)).cumsum(axis=0)
    rw = np.vstack([y[:1] - 0.1, y[:-1]])
    assert abs(evaluate(y, rw, rw, close_col=3)["mase"] - 1.0) < 1e-12
    stat, pv = dm_test(np.ones(50) + rng.normal(0, .1, 50), np.ones(50) * 2)
    assert stat < 0 and pv < 0.05


def test_param_counts_match_paper():
    pytest.importorskip("ncps")
    pytest.importorskip("tensorflow")
    from cfcnet.models import build_model
    expected = {("cfc", 80, 4): 19736, ("ltc", 80, 4): 28102, ("lstm", 80, 4): 28484, ("gru", 80, 4): 21684,
                ("cfc", 28, 4): 2504, ("ltc", 28, 4): 4026, ("lstm", 28, 4): 4148, ("gru", 28, 4): 3224}
    for (arch, units, motor), n in expected.items():
        assert build_model(arch, 7, units, motor).count_params() == n
    for arch in ["cfc", "ltc", "lstm", "gru"]:                      # adjusted-close widths build
        for units in (20, 7):
            assert build_model(arch, 7, units, 1).count_params() > 0


def test_append_row_keeps_one_header(tmp_path):
    from cfcnet.io import append_row, completed
    f = tmp_path / "log.csv"
    append_row(f, {"seed": 0, "target": "adjclose", "r2_adjclose": 0.5})
    append_row(f, {"seed": 1, "target": "ohlc", "r2_open": 0.4, "r2_close": 0.3})
    append_row(f, {"seed": 2, "target": "ohlc", "status": "diverged"})
    df = pd.read_csv(f)
    assert len(df) == 3 and {"r2_adjclose", "r2_open", "status"} <= set(df.columns)
    assert completed(f, ["seed", "target"]) == {("0", "adjclose"), ("1", "ohlc"), ("2", "ohlc")}


def test_same_day_pairs_contain_the_target_and_next_day_do_not():
    """The same-session robustness task supervises on prices that are inputs of the same step;
    the forecasting task (next_day) never does."""
    df = synthetic(120)
    sd = make_pairs(df, "ohlc", "levels", timing="same_day")
    nd = make_pairs(df, "ohlc", "levels")
    cols = [0, 1, 2, 5]                                   # Open, High, Low, Close among the inputs
    np.testing.assert_allclose(sd.X_raw[:, cols], sd.Y_raw)
    assert not np.allclose(nd.X_raw[:, cols], nd.Y_raw)
    np.testing.assert_allclose(nd.X_raw[1:, cols], nd.Y_raw[:-1])   # next-day target = next row's input
    assert (sd.target_dates == sd.input_dates).all() and (nd.target_dates > nd.input_dates).all()


def test_full_series_scaler_differs_only_when_requested():
    from cfcnet.data import fixed_origin_window, prepare_window
    p = make_pairs(synthetic(200), "ohlc", "levels")
    w = fixed_origin_window(p.n, 0.2)
    tr, fu = prepare_window(p, *w), prepare_window(p, *w, scaler_fit="full")
    np.testing.assert_allclose(tr.l_scaler.data_max_, p.Y_level[w[0]:w[1]].max(0))
    np.testing.assert_allclose(fu.l_scaler.data_max_, p.Y_level[w[0]:w[2]].max(0))

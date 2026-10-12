"""The GPU path must reproduce the reference path. Run with:  pytest -q"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
pytest.importorskip("tensorflow")
pytest.importorskip("ncps")

from test_pipeline import synthetic  # noqa: E402

from cfcnet.data import fixed_origin_window, make_pairs, prepare_window  # noqa: E402


@pytest.fixture(scope="module")
def prep():
    p = make_pairs(synthetic(220), "ohlc", "returns")
    return prepare_window(p, *fixed_origin_window(p.n, 0.1))


@pytest.mark.parametrize("arch", ["cfc", "ltc", "lstm", "gru"])
def test_packed_compiled_training_matches_model_fit(prep, arch):
    """Two seeds trained together (summed loss, one Adam, compiled loop; XLA for CfC/LTC) give the
    same loss history and test metrics as two separate `model.fit` runs."""
    from cfcnet.fast import train_pack
    from cfcnet.train import run_one
    ref = [run_one(prep, arch, 12, s, epochs=3) for s in (0, 1)]
    fast = train_pack(prep, arch, 12, [0, 1], epochs=3)
    for r, f in zip(ref, fast):
        assert f["status"] == r["status"] == "ok"
        assert f["n_params"] == r["n_params"]
        np.testing.assert_allclose(f["last_loss"], r["last_loss"], rtol=1e-4)
        np.testing.assert_allclose(f["r2_joint"], r["r2_joint"], rtol=1e-3, atol=1e-5)


def test_plan_covers_the_grid():
    from cfcnet import jobs as J
    from cfcnet.io import load_config
    cfg = load_config()
    jobs = J.plan(cfg, pack=3)
    runs = [r for j in jobs for r in j.run_ids]
    assert len(runs) == len(set(runs))
    head = [r for r in runs if r.startswith("headline_")]
    assert len(head) == 2 * 2 * 4 * cfg["protocol"]["seeds"]          # targets x reprs x archs x seeds
    assert jobs[0].cost >= jobs[-1].cost                               # longest jobs first
    assert all(len(j.seeds) <= 3 for j in jobs)

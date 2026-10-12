#!/usr/bin/env python
"""Where does any skill over the random walk come from? Per-column skill of every run.

    python scripts/11_skill_decomposition.py            # writes runs/skill.csv

For every finished network run (fixed-origin, walk-forward, history ablation) and every
fixed-origin benchmark, on the scaled test levels:

* mase_<col>       MAE of the column / MAE of the random walk y_t on that column;
* rel_naive_<col>  MAE of the column / MAE of the close-anchored naive C_t on that column;
* rel_naive        joint MAE / joint MAE of the close-anchored naive;
* dm_naive_stat, dm_naive_p   Diebold-Mariano (HLN) test of the joint squared error against the
                   close-anchored naive (negative statistic: the run is more accurate).

For the joint OHLC target the random walk carries O_t, H_t, L_t forward, whereas the next open is
anchored at the current close; the close-anchored naive separates skill on the open gap and the
intraday range from skill on the close. For the adjusted-close target both benchmarks coincide.
The column scaling is recovered exactly from the saved raw and scaled actual values.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cfcnet import cli  # noqa: E402
from cfcnet.io import runs_dir  # noqa: E402
from cfcnet.stats import dm_test, joint_sq_loss  # noqa: E402

COLS = {4: ["open", "high", "low", "close"], 1: ["adjclose"]}


def close_naive_scaled(z) -> np.ndarray:
    """C_t expressed in each column's scaled units (exact inverse of the per-column Min-Max map)."""
    A, S, R = z["actual"].astype(float), z["actual_scaled"].astype(float), z["rw_scaled"].astype(float)
    K = A.shape[1]
    if K == 1:
        return R.copy()
    i, j = int(np.argmin(S[:, 0])), int(np.argmax(S[:, 0]))
    k = (A[j] - A[i]) / (S[j] - S[i])           # raw = k * scaled + c, per column
    c = A[i] - k * S[i]
    close_t = k[3] * R[:, 3] + c[3]
    return (close_t[:, None] - c[None, :]) / k[None, :]


def skill(z, pred) -> dict:
    y, rw = z["actual_scaled"], z["rw_scaled"]
    nv = close_naive_scaled(z)
    ae = lambda p: np.abs(p - y).mean(axis=0)
    m, r, n = ae(pred), ae(rw), ae(nv)
    out = {}
    for q, col in enumerate(COLS[y.shape[1]]):
        out[f"mase_{col}"] = m[q] / r[q]
        out[f"rel_naive_{col}"] = m[q] / n[q]
    out["mase_joint"] = m.mean() / r.mean()
    out["rel_naive"] = m.mean() / n.mean()
    stat, p = dm_test(joint_sq_loss(y, pred), joint_sq_loss(y, nv))
    out.update(dm_naive_stat=stat, dm_naive_p=p, naive_vs_rw=n.mean() / r.mean())
    return out


def main():
    args = cli.base_parser(__doc__).parse_args()
    out = runs_dir(args.out)
    rows = []
    for name in ["fixed_origin", "walk_forward", "history_ablation"]:
        f = out / f"{name}.csv"
        if not f.exists():
            continue
        for r in pd.read_csv(f).to_dict("records"):
            if r.get("status") != "ok":
                continue
            z = np.load(out / "predictions" / f"{r['run_id']}.npz")
            rows.append({"source": name, "run_id": r["run_id"], "target": r["target"], "repr": r["repr"],
                         "arch": r["arch"], **skill(z, z["pred_scaled"])})
    b = out / "baselines.csv"
    if b.exists():
        for r in pd.read_csv(b).to_dict("records"):
            tag = f"baseline_{r['index']}_{r['target']}_tf{round(r['test_frac'] * 100)}_{r['model']}"
            zb = out / "predictions" / f"{tag}.npz"

            # the benchmark files hold no raw levels: borrow them from any network run on the same window
            src = sorted((out / "predictions").glob(f"*_{r['index']}_{r['target']}_levels_*_tf{round(r['test_frac'] * 100)}_s0.npz"))
            if not zb.exists() or not src:
                continue
            zn = dict(np.load(src[0]))
            zbb = np.load(zb)
            if zn["actual_scaled"].shape != zbb["actual_scaled"].shape or not np.allclose(zn["actual_scaled"], zbb["actual_scaled"]):
                continue
            rows.append({"source": "baseline", "run_id": tag, "target": r["target"], "repr": "---",
                         "arch": r["model"], **skill(zn, zbb["pred_scaled"])})
    pd.DataFrame(rows).to_csv(out / "skill.csv", index=False)
    print(f"skill.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()

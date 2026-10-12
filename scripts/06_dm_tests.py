#!/usr/bin/env python
"""Diebold-Mariano tests (Harvey-Leybourne-Newbold correction, h = 1) on the saved predictions.

* runs/dm_vs_rw.csv      every network run (fixed-origin, walk-forward, history, primary window) vs its
                         random walk (for the same-session robustness task the reference is y_{t-1});
* runs/dm_pairwise.csv   within each fixed-origin cell (experiment, index, target, width, split):
                         all pairs among {arch x representation, random walk, ARIMA}, using the
                         daily joint squared error averaged over seeds for each network.
Loss = daily joint squared error (mean over the target columns) on the scaled test levels.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cfcnet import cli  # noqa: E402
from cfcnet.io import runs_dir  # noqa: E402
from cfcnet.stats import dm_test, joint_sq_loss  # noqa: E402


def main():
    args = cli.base_parser(__doc__).parse_args()
    out = runs_dir(args.out)
    rows = []
    for name in ["fixed_origin", "walk_forward", "history_ablation", "primary_window"]:
        f = out / f"{name}.csv"
        if not f.exists():
            continue
        for r in pd.read_csv(f).to_dict("records"):
            if r.get("status") != "ok":
                continue
            z = np.load(out / "predictions" / f"{r['run_id']}.npz")
            stat, p = dm_test(joint_sq_loss(z["actual_scaled"], z["pred_scaled"]),
                              joint_sq_loss(z["actual_scaled"], z["rw_scaled"]))
            rows.append({"source": name, "run_id": r["run_id"], "target": r["target"], "repr": r["repr"],
                         "arch": r["arch"], "dm_stat": stat, "p_value": p, "model_better": stat < 0,
                         "significant_5pct": bool(stat < 0 and p < 0.05),
                         "rw_significantly_better": bool(stat > 0 and p < 0.05),
                         "variant": f"{r['timing']}_{r['scaler_fit']}" if name == "primary_window" else ""})
    pd.DataFrame(rows).to_csv(out / "dm_vs_rw.csv", index=False)
    print(f"dm_vs_rw.csv: {len(rows)} runs")

    f = out / "fixed_origin.csv"
    if not f.exists():
        return
    fo = pd.read_csv(f)
    fo = fo[fo.status == "ok"]
    pair_rows = []
    for (exp, index, target, units, tf_), g in fo.groupby(["experiment", "index", "target", "units", "test_frac"]):
        losses, z = {}, None
        for (arch, rep), ga in g.groupby(["arch", "repr"]):
            L = []
            for rid in ga.run_id:
                z = np.load(out / "predictions" / f"{rid}.npz")
                L.append(joint_sq_loss(z["actual_scaled"], z["pred_scaled"]))
            losses[f"{arch}_{rep}"] = np.mean(L, axis=0)
        losses["rw"] = joint_sq_loss(z["actual_scaled"], z["rw_scaled"])
        bf = out / "predictions" / f"baseline_{index}_{target}_tf{round(tf_ * 100)}_arima.npz"
        if bf.exists():
            zb = np.load(bf)
            losses["arima"] = joint_sq_loss(zb["actual_scaled"], zb["pred_scaled"])
        for m1, m2 in itertools.combinations(sorted(losses), 2):
            stat, p = dm_test(losses[m1], losses[m2])
            pair_rows.append({"experiment": exp, "index": index, "target": target, "units": units, "test_frac": tf_,
                              "model_a": m1, "model_b": m2, "dm_stat": stat, "p_value": p})
    pd.DataFrame(pair_rows).to_csv(out / "dm_pairwise.csv", index=False)
    print(f"dm_pairwise.csv: {len(pair_rows)} comparisons")


if __name__ == "__main__":
    main()

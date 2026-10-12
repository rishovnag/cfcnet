#!/usr/bin/env python
"""Random-walk, close-anchored naive and ARIMA benchmarks on every fixed-origin window, for both targets.

    python scripts/03_baselines.py                       # all experiments
    python scripts/03_baselines.py --experiment headline

Benchmarks are evaluated in the same scaled-level space as the networks, so they serve both
representations. ARIMA fits run in parallel, one target column per worker (--workers N).
Writes runs/baselines.csv and runs/predictions/baseline_<...>.npz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from cfcnet import baselines, cli, parallel  # noqa: E402
from cfcnet.data import fixed_origin_window, make_pairs, prepare_window  # noqa: E402
from cfcnet.io import append_row, completed, load_frame, runs_dir  # noqa: E402
from cfcnet.metrics import evaluate  # noqa: E402

KEYS = ["index", "target", "test_frac", "model"]


def log(csv, out, index, target, tf_, model, prep, pairs, raw, orders):
    y = prep.test_level
    p = prep.l_scaler.transform(raw)
    row = {"index": index, "target": target, "test_frac": tf_, "model": model,
           "test_start": prep.test_dates[0], "test_end": prep.test_dates[1], "n_test": len(y),
           "orders": str(orders) if orders else "",
           **evaluate(y, p, prep.test_rw, prep.close_col, pairs.target_cols,
                      ret_true=prep.returns_of(y), ret_pred=prep.returns_of(p))}
    append_row(csv, row)
    np.savez_compressed(out / "predictions" / f"baseline_{index}_{target}_tf{round(tf_ * 100)}_{model}.npz",
                        pred_scaled=p, actual_scaled=y, rw_scaled=prep.test_rw)
    print(f"{index} {target} {tf_} {model}: R2={row['r2_joint']:.4f} MASE={row['mase']:.3f} "
          f"DA={row['da']:.3f}", flush=True)


def main():
    ap = parallel.add_args(cli.add_grid_args(cli.base_parser(__doc__)))
    ap.add_argument("--experiment", nargs="*", default=None)
    ap.add_argument("--no-arima", action="store_true")
    args = ap.parse_args()
    cfg = cli.load(args)
    pr, bl = cfg["protocol"], cfg["baselines"]
    out = runs_dir(args.out)
    csv = out / "baselines.csv"
    done = completed(csv, KEYS)
    targets = args.targets or pr["targets"]

    jobs = set()
    for name in args.experiment or cfg["experiments"]:
        ex = cfg["experiments"][name]
        for index in ex["indices"]:
            for tf_ in ex["test_fracs"]:
                for target in targets:
                    jobs.add((index, target, tf_))
    frames, cells, tasks = {}, {}, []
    for job in sorted(jobs):
        index, target, tf_ = job
        df = frames.setdefault(index, load_frame(cfg, index, args.data_dir))
        pairs = make_pairs(df, target, "levels", pr["feature_cols"], pr["add_weekday"])
        a, b, c = fixed_origin_window(pairs.n, tf_)
        prep = prepare_window(pairs, a, b, c, pr["warmup"])
        cells[job] = (pairs, prep)
        if (index, target, str(tf_), "rw") not in done:
            log(csv, out, index, target, tf_, "rw", prep, pairs, pairs.Y_prev_level[b:c].copy(), None)
        if target == "ohlc" and (index, target, str(tf_), "naive_close") not in done:
            log(csv, out, index, target, tf_, "naive_close", prep, pairs, baselines.naive_close(pairs, b, c), None)
        if args.no_arima or (index, target, str(tf_), "arima") in done:
            continue
        for k in range(pairs.Y_level.shape[1]):      # one task per target column
            tasks.append(((job, k), dict(series=pairs.Y_level[a:c, k], n_train=b - a, ps=bl["arima_p"],
                                         qs=bl["arima_q"], d=bl["arima_d"], trend=bl["arima_trend"])))
    tasks.sort(key=lambda t: -t[1]["n_train"])

    cols, failures = {}, []
    for (job, k), res in parallel.run(baselines.arima_column, tasks, workers=args.workers, threads=args.threads):
        if isinstance(res, parallel.TaskError):
            failures.append(((job, k), res))
            continue
        cols.setdefault(job, {})[k] = res
        pairs, prep = cells[job]
        if len(cols[job]) == pairs.Y_level.shape[1]:   # all columns of this cell are in
            fits = [cols[job][j] for j in range(len(cols[job]))]
            raw = np.stack([pred for pred, _ in fits], axis=1)
            log(csv, out, *job, "arima", prep, pairs, raw, [order for _, order in fits])
    parallel.report_failures(failures)

if __name__ == "__main__":
    main()

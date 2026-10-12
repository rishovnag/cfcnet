#!/usr/bin/env python
"""Rolling-origin walk-forward cross-validation on NIFTY 100 (Section 6.5), for both targets
and both representations.

    python scripts/04_walk_forward.py            # targets x reprs x 4 archs x folds x 3 seeds
    python scripts/04_walk_forward.py --targets ohlc --reprs levels --arch cfc gru --epochs 2 --max-folds 2

The number of folds follows from the sample length (5-year train / 1-year test, stepped 1 year,
anchored to the end of the series). Writes runs/walk_forward.csv, runs/walk_forward_folds.csv and
runs/adf_per_fold.csv; fold dates and regime labels are logged before any training. Runs execute
in parallel, one per core (--workers N; --workers 1 for a serial loop).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfcnet.io import quiet_tensorflow  # noqa: E402

quiet_tensorflow()
import pandas as pd  # noqa: E402

from cfcnet import cli, parallel  # noqa: E402
from cfcnet.data import make_pairs, prepare_window, regime_label, walk_forward_windows  # noqa: E402
from cfcnet.io import append_row, completed, load_frame, runs_dir, write_environment  # noqa: E402
from cfcnet.stats import unit_root_table  # noqa: E402
from cfcnet.train import expected_cost, run_one  # noqa: E402

KEYS = ["index", "target", "repr", "arch", "fold", "seed"]


def fold_table(pairs, folds, wf):
    rows, adf = [], []
    close = pairs.Y_level[:, pairs.close_col]
    d = lambda j: str(pd.Timestamp(pairs.target_dates[j]).date())
    for k, (a, b, c) in enumerate(folds):
        lab, ret, vol = regime_label(close[b:c], wf["bull_threshold"], wf["bear_threshold"])
        rows.append({"fold": k, "train_start": d(a), "train_end": d(b - 1), "test_start": d(b),
                     "test_end": d(c - 1), "regime": lab, "ann_return": ret, "ann_vol": vol})
        adf.append({"fold": k, **unit_root_table(close[a:b])})
    return pd.DataFrame(rows), pd.DataFrame(adf)


def main():
    ap = parallel.add_args(cli.add_grid_args(cli.base_parser(__doc__)), gpu=True)
    ap.add_argument("--arch", nargs="*")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--max-folds", type=int, default=None, help="keep only the most recent folds")
    args = ap.parse_args()
    cfg = cli.load(args)
    pr, wf = cfg["protocol"], cfg["walk_forward"]
    out = runs_dir(args.out)
    write_environment(out)
    n_gpu = parallel.gpu_workers(args)
    index = wf["index"]
    df = load_frame(cfg, index, args.data_dir)

    ref = make_pairs(df, "ohlc", "levels", pr["feature_cols"], pr["add_weekday"])
    folds = walk_forward_windows(ref.n, wf["train_window"], wf["test_window"], wf["step"])
    fold_df, adf_df = fold_table(ref, folds, wf)
    fold_df.to_csv(out / "walk_forward_folds.csv", index=False)
    adf_df.to_csv(out / "adf_per_fold.csv", index=False)
    print(fold_df.to_string(index=False), flush=True)
    run_folds = list(enumerate(folds))[-args.max_folds:] if args.max_folds else list(enumerate(folds))

    csv = out / "walk_forward.csv"
    done = completed(csv, KEYS)
    seeds = cli.parse_seeds(args.seeds, 0) if args.seeds else wf["seeds"]
    tasks = []
    for target, rep in cli.grid(cfg, args):
        units = cli.units_for(cfg, target, "default")
        pairs = make_pairs(df, target, rep, pr["feature_cols"], pr["add_weekday"])
        for k, (a, b, c) in run_folds:
            prep = prepare_window(pairs, a, b, c, pr["warmup"])
            for arch in args.arch or wf["archs"]:
                for seed in seeds:
                    if (index, target, rep, arch, str(k), str(seed)) in done:
                        continue
                    run_id = f"wf_{index}_{target}_{rep}_{arch}{units}_fold{k}_s{seed}"
                    meta = {"index": index, "target": target, "repr": rep, "arch": arch, "units": units, "fold": k,
                            "seed": seed, "regime": fold_df.loc[k, "regime"], "run_id": run_id,
                            "device": parallel.device_for(arch, args, n_gpu)}
                    tasks.append((meta, dict(prep=prep, arch=arch, units=units, seed=seed, epochs=pr["epochs"],
                                             learning_rate=pr["learning_rate"], batch_size=pr["batch_size"],
                                             jit_compile=pr["jit_compile_ncps"], target_names=pairs.target_cols,
                                             run_id=run_id, out_dir=out)))
    tasks.sort(key=lambda t: -expected_cost(t[0]["arch"], t[1]["prep"]))

    failures = []
    for meta, row in parallel.run(run_one, tasks, workers=args.workers, threads=args.threads,
                                  tensorflow=True, gpu_workers=n_gpu, on_gpu=lambda m: m["device"] == "gpu"):
        if isinstance(row, parallel.TaskError):
            failures.append((meta["run_id"], row))
            print(f"{meta['run_id']}: FAILED {row.message}", flush=True)
            continue
        row = {**meta, **row}
        append_row(csv, row)
        print(f"{meta['run_id']} ({row['regime']}): R2={row.get('r2_joint', float('nan')):.4f} "
              f"MASE={row.get('mase', float('nan')):.3f} [{row['status']}]", flush=True)
    parallel.report_failures(failures)

if __name__ == "__main__":
    main()

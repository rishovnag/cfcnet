#!/usr/bin/env python
"""Primary window: the headline comparison on the NIFTY 100 primary sample (to 16 July 2024) and
its robustness to the protocol (Sections 6.3-6.4 of the paper; Tables 4 and 8, Figures 6 and 10).

    python scripts/10_primary_window.py                  # all variants, 10 seeds, in parallel
    python scripts/10_primary_window.py --seeds 3 --epochs 2 --out runs_smoke   # smoke test

The primary sample is the NIFTY 100 file data/primary/NIFTY100_2024-07-16.csv (Yahoo ^CNX100,
2005-11-30 to 2024-07-16; see data/primary/README.md). The headline models (95-5 split, levels,
default widths, the optimiser and epochs of configs/paper.yaml) are trained under the variants of
configs/paper.yaml:primary_window, each a triple (data, timing, scaler_fit):

    (primary, next_day, train)  the one-step-ahead protocol of the paper -> the headline (Table 4)
    (primary, same_day, train)  robustness protocol (2): target of the input session, train-only scaler
    (primary, same_day, full)   robustness protocol (1): as (2), with the scalers fitted on the full series

Protocols (1) and (2) are robustness checks, not forecasts: under same_day the target prices are
among the inputs. The log records every column's R2, so single-column metrics can be recomputed.
Results are appended to runs/primary_window.csv (resume-safe).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfcnet.io import quiet_tensorflow  # noqa: E402

quiet_tensorflow()

from cfcnet import cli, parallel  # noqa: E402
from cfcnet.data import fixed_origin_window, make_pairs, prepare_window  # noqa: E402
from cfcnet.data import load_index  # noqa: E402
from cfcnet.io import REPO, append_row, completed, load_frame, runs_dir, write_environment  # noqa: E402
from cfcnet.train import expected_cost, run_one  # noqa: E402

KEYS = ["index", "data", "target", "timing", "scaler_fit", "arch", "units", "seed"]


def main():
    ap = parallel.add_args(cli.base_parser(__doc__), gpu=True)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--arch", nargs="*", default=None)
    ap.add_argument("--seeds", default=None, help="'10' (seeds 0..9), '0,1,2' or '5-9'")
    args = ap.parse_args()
    cfg = cli.load(args)
    pr, ex = cfg["protocol"], cfg["primary_window"]
    out = runs_dir(args.out)
    write_environment(out)
    n_gpu = parallel.gpu_workers(args)
    csv = out / "primary_window.csv"
    done = completed(csv, KEYS)
    seeds = cli.parse_seeds(args.seeds, ex["seeds"])

    tasks = []
    index = ex["index"]
    frames = {"primary": load_index(REPO / ex["primary_file"]), "extended": load_frame(cfg, index, args.data_dir)}
    for target in args.targets or pr["targets"]:
        units = cli.units_for(cfg, target, ex["width"])
        for data, timing, scaler_fit in ex["variants"]:
            pairs = make_pairs(frames[data], target, "levels", pr["feature_cols"], pr["add_weekday"], timing=timing)
            prep = prepare_window(pairs, *fixed_origin_window(pairs.n, ex["test_frac"]), pr["warmup"],
                                  scaler_fit=scaler_fit)
            for arch in args.arch or ex["archs"]:
                for seed in seeds:
                    key = (index, data, target, timing, scaler_fit, arch, str(units), str(seed))
                    if key in done:
                        continue
                    run_id = f"pw_{index}_{data}_{target}_{timing}_{scaler_fit}_{arch}{units}_s{seed}"
                    meta = {**dict(zip(KEYS, key)), "units": units, "seed": seed, "repr": "levels",
                            "test_frac": ex["test_frac"], "run_id": run_id, "motor": cli.motor_for(target),
                            "device": parallel.device_for(arch, args, n_gpu)}
                    tasks.append((meta, dict(prep=prep, arch=arch, units=units, seed=seed, epochs=pr["epochs"],
                                             learning_rate=pr["learning_rate"], batch_size=pr["batch_size"],
                                             jit_compile=pr["jit_compile_ncps"], target_names=pairs.target_cols,
                                             run_id=run_id, out_dir=out)))
    tasks.sort(key=lambda t: -expected_cost(t[0]["arch"], t[1]["prep"]))
    print(f"{len(tasks)} runs to do ({len(done)} already in {csv})", flush=True)

    failures = []
    for meta, row in parallel.run(run_one, tasks, workers=args.workers, threads=args.threads,
                                  tensorflow=True, gpu_workers=n_gpu, on_gpu=lambda m: m["device"] == "gpu"):
        if isinstance(row, parallel.TaskError):
            failures.append((meta["run_id"], row))
            print(f"{meta['run_id']}: FAILED {row.message}", flush=True)
            continue
        row = {**meta, **row}
        append_row(csv, row)
        print(f"{meta['run_id']}: R2={row.get('r2_joint', float('nan')):.4f} "
              f"R2_open={row.get('r2_open', float('nan')):.4f} [{row['status']}] ({row['train_time_s']:.0f}s)",
              flush=True)
    parallel.report_failures(failures)


if __name__ == "__main__":
    main()

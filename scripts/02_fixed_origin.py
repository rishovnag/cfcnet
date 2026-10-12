#!/usr/bin/env python
"""Fixed-origin experiments, run for both targets (adjusted close, joint OHLC) and both
representations (levels, returns).

    python scripts/02_fixed_origin.py --experiment headline          # NIFTY 100, 4 archs, 30 seeds
    python scripts/02_fixed_origin.py --experiment capacity          # ablation widths (7 / 28 units)
    python scripts/02_fixed_origin.py --experiment split_sweep       # 80-20 ... 95-5, NIFTY 100 / 500
    python scripts/02_fixed_origin.py --experiment cross_market      # NIFTY 50, S&P 500, Nikkei 225, SSE
    python scripts/02_fixed_origin.py --experiment headline capacity split_sweep cross_market   # one shared pool
    python scripts/02_fixed_origin.py --experiment headline --targets ohlc --reprs levels \
        --arch cfc --seeds 3 --epochs 2                                # smoke test

Runs execute in parallel, one per core (--workers N to change, --workers 1 for a serial loop);
the longest runs are started first.
Widths come from configs/paper.yaml (adjclose: 20 / 7 units; ohlc: 80 / 28 units).
Results are appended to runs/fixed_origin.csv (resume-safe: finished runs are skipped).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfcnet.io import quiet_tensorflow  # noqa: E402

quiet_tensorflow()

from cfcnet import cli, parallel  # noqa: E402
from cfcnet.data import fixed_origin_window, make_pairs, prepare_window  # noqa: E402
from cfcnet.io import append_row, completed, load_frame, runs_dir, write_environment  # noqa: E402
from cfcnet.train import expected_cost, run_one  # noqa: E402

KEYS = ["experiment", "index", "target", "repr", "arch", "units", "test_frac", "seed"]


def main():
    ap = parallel.add_args(cli.add_grid_args(cli.base_parser(__doc__)), gpu=True)
    ap.add_argument("--experiment", required=True, nargs="+",
                    choices=["headline", "capacity", "split_sweep", "cross_market"])
    ap.add_argument("--index", nargs="*", help="override the experiments' index lists")
    ap.add_argument("--arch", nargs="*", help="override the experiments' architecture lists")
    ap.add_argument("--seeds", default=None, help="'30' (seeds 0..29), '0,1,2' or '5-9'")
    args = ap.parse_args()
    cfg = cli.load(args)
    pr = cfg["protocol"]
    out = runs_dir(args.out)
    write_environment(out)
    n_gpu = parallel.gpu_workers(args)
    csv = out / "fixed_origin.csv"
    done = completed(csv, KEYS)
    seeds = cli.parse_seeds(args.seeds, pr["seeds"])

    tasks, frames, preps = [], {}, {}
    for experiment in args.experiment:
        ex = cfg["experiments"][experiment]
        for index in args.index or ex["indices"]:
            df = frames.setdefault(index, load_frame(cfg, index, args.data_dir))
            for target, rep in cli.grid(cfg, args):
                units = cli.units_for(cfg, target, ex["width"])
                for tf_ in ex["test_fracs"]:
                    if (index, target, rep, tf_) not in preps:
                        pairs = make_pairs(df, target, rep, pr["feature_cols"], pr["add_weekday"])
                        preps[index, target, rep, tf_] = prepare_window(pairs, *fixed_origin_window(pairs.n, tf_),
                                                                        pr["warmup"])
                    prep = preps[index, target, rep, tf_]
                    for arch in args.arch or ex["archs"]:
                        for seed in seeds:
                            key = (experiment, index, target, rep, arch, str(units), str(tf_), str(seed))
                            if key in done:
                                continue
                            run_id = f"{experiment}_{index}_{target}_{rep}_{arch}{units}_tf{round(tf_ * 100)}_s{seed}"
                            meta = {**dict(zip(KEYS, [experiment, index, target, rep, arch, units, tf_, seed])),
                                    "run_id": run_id, "motor": cli.motor_for(target),
                                    "device": parallel.device_for(arch, args, n_gpu)}
                            tasks.append((meta, dict(prep=prep, arch=arch, units=units, seed=seed, epochs=pr["epochs"],
                                                     learning_rate=pr["learning_rate"], batch_size=pr["batch_size"],
                                                     jit_compile=pr["jit_compile_ncps"],
                                                     target_names=prep.pairs.target_cols, run_id=run_id, out_dir=out)))
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
        print(f"{meta['run_id']}: R2={row.get('r2_joint', float('nan')):.4f} "
              f"MASE={row.get('mase', float('nan')):.3f} [{row['status']}] "
              f"({row['train_time_s']:.0f}s)", flush=True)
    parallel.report_failures(failures)


if __name__ == "__main__":
    main()

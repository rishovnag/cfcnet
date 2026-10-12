#!/usr/bin/env python
"""Controlled history-length ablation on the fixed 95-5 test window (Section 6.5), for both
targets and both representations.

    python scripts/05_history_ablation.py        # targets x reprs x {CFC, GRU} x 5 histories x 3 seeds

Runs execute in parallel, one per core (--workers N). Writes runs/history_ablation.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfcnet.io import quiet_tensorflow  # noqa: E402

quiet_tensorflow()

from cfcnet import cli, parallel  # noqa: E402
from cfcnet.data import history_windows, make_pairs, prepare_window  # noqa: E402
from cfcnet.io import append_row, completed, load_frame, runs_dir, write_environment  # noqa: E402
from cfcnet.train import expected_cost, run_one  # noqa: E402

KEYS = ["index", "target", "repr", "arch", "hist_len", "seed"]


def main():
    ap = parallel.add_args(cli.add_grid_args(cli.base_parser(__doc__)), gpu=True)
    ap.add_argument("--arch", nargs="*")
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    cfg = cli.load(args)
    pr, ha = cfg["protocol"], cfg["history_ablation"]
    out = runs_dir(args.out)
    write_environment(out)
    n_gpu = parallel.gpu_workers(args)
    index = ha["index"]
    df = load_frame(cfg, index, args.data_dir)
    csv = out / "history_ablation.csv"
    done = completed(csv, KEYS)
    seeds = cli.parse_seeds(args.seeds, 0) if args.seeds else ha["seeds"]
    tasks = []
    for target, rep in cli.grid(cfg, args):
        units = cli.units_for(cfg, target, "default")
        pairs = make_pairs(df, target, rep, pr["feature_cols"], pr["add_weekday"])
        for a, b, c in history_windows(pairs.n, ha["test_frac"], ha["history_lengths"]):
            L = b - a
            prep = prepare_window(pairs, a, b, c, pr["warmup"])
            for arch in args.arch or ha["archs"]:
                for seed in seeds:
                    if (index, target, rep, arch, str(L), str(seed)) in done:
                        continue
                    run_id = f"hist_{index}_{target}_{rep}_{arch}{units}_L{L}_s{seed}"
                    meta = {"index": index, "target": target, "repr": rep, "arch": arch, "units": units,
                            "hist_len": L, "hist_years": round(L / 252, 1), "seed": seed, "run_id": run_id,
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
        print(f"{meta['run_id']}: R2={row.get('r2_joint', float('nan')):.4f} [{row['status']}]", flush=True)
    parallel.report_failures(failures)

if __name__ == "__main__":
    main()

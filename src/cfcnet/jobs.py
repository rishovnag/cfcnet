"""The complete experiment grid as a list of independent jobs.

A *job* trains the seeds of one cell together (seed packing) on one data window:

    kind = fixed   experiment x index x target x representation x arch x width x split  (30 seeds)
    kind = wf      target x representation x arch x fold                              (3 seeds)
    kind = hist    target x representation x arch x history length                    (3 seeds)

Each job lists the run ids it produces. A run is complete when runs/rows/<run_id>.json exists,
so the plan can be re-generated and re-launched at any time without repeating finished runs.
Jobs carry a cost estimate (training steps x architecture factor) so the scheduler can start the
longest jobs first (longest-processing-time-first keeps the tail of the run short).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import cli
from .data import fixed_origin_window, history_windows, make_pairs, walk_forward_windows
from .io import load_frame

# relative cost per training step (from walk-forward timings: LTC 110 s, CfC 77 s,
# LSTM/GRU 34 s for 1,260 steps x 400 epochs); used only to order the queue
ARCH_COST = {"ltc": 1.45, "cfc": 1.0, "lstm": 0.45, "gru": 0.45}


@dataclass
class Job:
    job_id: str
    kind: str
    index: str
    target: str
    repr: str
    arch: str
    units: int
    window: tuple
    seeds: list
    run_ids: list
    meta: dict = field(default_factory=dict)
    cost: float = 0.0


def _chunks(seq, n):
    n = max(1, int(n))
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def plan(cfg: dict, pack: int = 1, data_dir=None, experiments=None, include=("fixed", "wf", "hist"),
         targets=None, reprs=None, archs=None, seeds_override=None) -> list[Job]:
    pr = cfg["protocol"]
    targets = targets or pr["targets"]
    reprs = reprs or pr["representations"]
    E = int(pr["epochs"])
    jobs: list[Job] = []
    frames, npairs = {}, {}

    def n_of(index):
        if index not in npairs:
            frames[index] = load_frame(cfg, index, data_dir)
            npairs[index] = make_pairs(frames[index], "ohlc", "levels", pr["feature_cols"], pr["add_weekday"]).n
        return npairs[index]

    def add(kind, prefix, index, target, rep, arch, units, window, seeds, meta, rid):
        a, b, c = window
        for chunk in _chunks(list(seeds), pack):
            ids = [rid(s) for s in chunk]
            jid = f"{prefix}__s{chunk[0]}-{chunk[-1]}"
            jobs.append(Job(jid, kind, index, target, rep, arch, units, (a, b, c), list(chunk), ids, dict(meta),
                            cost=(b - a) * E * ARCH_COST[arch] * len(chunk)))

    if "fixed" in include:
        for name, ex in cfg["experiments"].items():
            if experiments and name not in experiments:
                continue
            for index in ex["indices"]:
                n = n_of(index)
                for target in targets:
                    units = cli.units_for(cfg, target, ex["width"])
                    for rep in reprs:
                        for tf_ in ex["test_fracs"]:
                            w = fixed_origin_window(n, tf_)
                            for arch in archs or ex["archs"]:
                                seeds = seeds_override if seeds_override is not None else list(range(pr["seeds"]))
                                base = f"{name}_{index}_{target}_{rep}_{arch}{units}_tf{round(tf_ * 100)}"
                                add("fixed", base, index, target, rep, arch, units, w, seeds,
                                    {"experiment": name, "test_frac": tf_}, lambda s, base=base: f"{base}_s{s}")
    if "wf" in include:
        wf = cfg["walk_forward"]
        n = n_of(wf["index"])
        folds = walk_forward_windows(n, wf["train_window"], wf["test_window"], wf["step"])
        for target in targets:
            units = cli.units_for(cfg, target, "default")
            for rep in reprs:
                for arch in archs or wf["archs"]:
                    for k, w in enumerate(folds):
                        seeds = seeds_override if seeds_override is not None else wf["seeds"]
                        base = f"wf_{wf['index']}_{target}_{rep}_{arch}{units}_fold{k}"
                        add("wf", base, wf["index"], target, rep, arch, units, w, seeds, {"fold": k},
                            lambda s, base=base: f"{base}_s{s}")
    if "hist" in include:
        ha = cfg["history_ablation"]
        n = n_of(ha["index"])
        for target in targets:
            units = cli.units_for(cfg, target, "default")
            for rep in reprs:
                for arch in archs or ha["archs"]:
                    for w in history_windows(n, ha["test_frac"], ha["history_lengths"]):
                        L = w[1] - w[0]
                        seeds = seeds_override if seeds_override is not None else ha["seeds"]
                        base = f"hist_{ha['index']}_{target}_{rep}_{arch}{units}_L{L}"
                        add("hist", base, ha["index"], target, rep, arch, units, w, seeds,
                            {"hist_len": L, "hist_years": round(L / 252, 1)}, lambda s, base=base: f"{base}_s{s}")
    jobs.sort(key=lambda j: -j.cost)
    return jobs


def save(jobs: list[Job], path: Path) -> None:
    with open(path, "w") as fh:
        for j in jobs:
            fh.write(json.dumps(asdict(j)) + "\n")


def load(path: Path) -> list[Job]:
    out = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            d["window"] = tuple(d["window"])
            out.append(Job(**d))
    return out


def run_done(rows_dir: Path, run_id: str) -> bool:
    return (rows_dir / f"{run_id}.json").exists()


def job_done(rows_dir: Path, job: Job) -> bool:
    return all(run_done(rows_dir, r) for r in job.run_ids)


def walk_forward_tables(cfg: dict, data_dir=None):
    """Fold dates / regime labels and per-fold ADF tests (as written by the base 04 script)."""
    import pandas as pd
    from .data import regime_label
    from .stats import unit_root_table
    pr, wf = cfg["protocol"], cfg["walk_forward"]
    pairs = make_pairs(load_frame(cfg, wf["index"], data_dir), "ohlc", "levels", pr["feature_cols"], pr["add_weekday"])
    folds = walk_forward_windows(pairs.n, wf["train_window"], wf["test_window"], wf["step"])
    close = pairs.Y_level[:, pairs.close_col]
    d = lambda j: str(pd.Timestamp(pairs.target_dates[j]).date())
    rows, adf = [], []
    for k, (a, b, c) in enumerate(folds):
        lab, ret, vol = regime_label(close[b:c], wf["bull_threshold"], wf["bear_threshold"])
        rows.append({"fold": k, "train_start": d(a), "train_end": d(b - 1), "test_start": d(b),
                     "test_end": d(c - 1), "regime": lab, "ann_return": ret, "ann_vol": vol})
        adf.append({"fold": k, **unit_root_table(close[a:b])})
    return pd.DataFrame(rows), pd.DataFrame(adf)

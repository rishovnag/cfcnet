#!/usr/bin/env python
"""Pre-modelling data diagnostics and the data facts quoted in the manuscript (Section 6.1).
No model is trained.

    python scripts/01_diagnostics.py                               # results/tables/
    python scripts/01_diagnostics.py --paper-dir ../research-paper  # also copy the LaTeX files there

Writes to results/tables/:
    dataset_diagnostics.csv               per index: vendor span, analysed span, N, flat-OHLC and zero-volume
                                          counts, AdjClose == Close check, ADF / KPSS on levels
    dataset_diagnostics_common_period.csv the four cross-market indices on their common period, with
                                          Engle-Granger p-values in BOTH normalisations
    dataset_stats_rows.tex                rows of the dataset-diagnostics table
    test_windows.csv / test_windows_rows.tex   fixed-origin 95-5 partitions
    split_windows.csv                     80-20 ... 95-5 test windows (NIFTY 100 / 500) with regime labels
    walk_forward_folds.csv                walk-forward folds (dates, regime labels)
    data_facts.tex                        \\newcommand macros for every data-dependent number in the text
If data/manifest.json is missing (data not yet refreshed with data/download_data.py), the macros are
marked provisional and render in red in the mark-up build of the paper.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cfcnet import cli  # noqa: E402
from cfcnet.data import (fixed_origin_window, flat_ohlc, history_windows, load_index, make_pairs,  # noqa: E402
                         regime_label, walk_forward_windows)
from cfcnet.io import REPO, data_path  # noqa: E402
from cfcnet.stats import engle_granger, unit_root_table  # noqa: E402

NAMES = {"NIFTY50": "NIFTY 50", "NIFTY100": "NIFTY 100", "NIFTY500": "NIFTY 500",
         "SPX": "S\\&P 500", "N225": "Nikkei 225", "SSE": "SSE Composite"}
MKEY = {"NIFTY50": "NiftyFifty", "NIFTY100": "NiftyHundred", "NIFTY500": "NiftyFiveHundred",
        "SPX": "SandP", "N225": "Nikkei", "SSE": "SSE"}
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five".split()
SPLIT = {0.20: "Eighty", 0.15: "EightyFive", 0.10: "Ninety", 0.05: "NinetyFive"}


def dfmt(x) -> str:
    return pd.Timestamp(x).strftime("%-d %B %Y")


def dshort(x) -> str:
    return pd.Timestamp(x).strftime("%-d %b %Y")


def num(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--paper-dir", default=None, help="copy data_facts.tex and the table rows into this folder")
    args = ap.parse_args()
    cfg = cli.load(args)
    pr = cfg["protocol"]
    tdir = REPO / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    macros = {}

    raw = {k: load_index(data_path(cfg, k, args.data_dir)) for k in cfg["indices"]}
    frames = {k: load_index(data_path(cfg, k, args.data_dir), v.get("start")) for k, v in cfg["indices"].items()}

    # ---------------------------------------------------------------- per index
    diag, win = [], []
    for k, df in frames.items():
        r = raw[k]
        flat = flat_ohlc(r)
        row = {"index": k, "vendor_first": r.Date.iloc[0].date(), "first": df.Date.iloc[0].date(),
               "last": df.Date.iloc[-1].date(), "N": len(df), "dropped_before_start": len(r) - len(df),
               "flat_ohlc_vendor": int(flat.sum()), "flat_ohlc_in_sample": int(flat_ohlc(df).sum()),
               "zero_volume_in_sample": int((df.Volume == 0).sum()),
               "adjclose_equals_close": bool(np.allclose(df.AdjClose, df.Close)),
               **unit_root_table(df.AdjClose.to_numpy())}
        diag.append(row)
        m = MKEY[k]
        macros.update({f"Start{m}": dfmt(df.Date.iloc[0]), f"End{m}": dfmt(df.Date.iloc[-1]),
                       f"N{m}": num(len(df)), f"VendorStart{m}": dfmt(r.Date.iloc[0]),
                       f"Flat{m}": num(int(flat.sum())), f"Dropped{m}": num(len(r) - len(df)),
                       f"ZeroVol{m}": num(int((df.Volume == 0).sum()))})
        pairs = make_pairs(df, "ohlc", "levels", pr["feature_cols"], pr["add_weekday"])
        a, b, c = fixed_origin_window(pairs.n, pr["test_frac"])
        d = lambda j: pairs.target_dates[j]
        win.append({"index": NAMES[k], "N": len(df), "train_targets": f"{dshort(d(a))} -- {dshort(d(b - 1))}",
                    "test_targets": f"{dshort(d(b))} -- {dshort(d(c - 1))}", "n_test": c - b})
        macros.update({f"TestStart{m}": dfmt(d(b)), f"TestEnd{m}": dfmt(d(c - 1)), f"TestN{m}": num(c - b)})
    pd.DataFrame(diag).to_csv(tdir / "dataset_diagnostics.csv", index=False)
    w = pd.DataFrame(win)
    w.to_csv(tdir / "test_windows.csv", index=False)
    rows_tex = "\n".join(f"{r['index']} & {num(r['N'])} & {r['train_targets']} & {r['test_targets']} & "
                         f"{num(r['n_test'])} \\\\" for r in win) + "\n"

    # ---------------------------------------------------------------- common period (cross-market table)
    J = pd.concat({k: frames[k].set_index("Date")["AdjClose"] for k in ["NIFTY50", "SPX", "N225", "SSE"]},
                  axis=1, join="inner").dropna()
    common, stats_tex = [], []
    for k in J:
        ur = unit_root_table(J[k].to_numpy())
        row = {"index": k, "common_start": J.index.min().date(), "common_end": J.index.max().date(),
               "sessions": len(J), "mean": J[k].mean(), "median": J[k].median(), "std": J[k].std(), **ur}
        eg1 = eg2 = np.nan
        if k != "NIFTY50":
            eg1 = engle_granger(J[k].values, J["NIFTY50"].values)
            eg2 = engle_granger(J["NIFTY50"].values, J[k].values)
        row.update({"eg_p_index_on_nifty50": eg1, "eg_p_nifty50_on_index": eg2})
        common.append(row)
        fmt_p = lambda p: "---" if not np.isfinite(p) else ("$<\\!0.001$" if p < 0.001 else f"{p:.3f}")
        kp = "$<\\!0.01$" if ur["kpss_level_p"] <= 0.01 else f"{ur['kpss_level_p']:.2f}"
        stats_tex.append(f"{NAMES[k]} & {J[k].mean():,.2f} & {J[k].median():,.2f} & {J[k].std():,.2f} & "
                         f"{fmt_p(ur['adf_level_p'])} & {kp} & {fmt_p(eg1)} & {fmt_p(eg2)} \\\\"
                         .replace(",", "{,}"))
    pd.DataFrame(common).to_csv(tdir / "dataset_diagnostics_common_period.csv", index=False)
    macros.update({"CommonStart": dfmt(J.index.min()), "CommonEnd": dfmt(J.index.max()), "CommonN": num(len(J))})
    C = np.log(J).diff().dropna().corr()
    for a_, b_ in [("NIFTY50", "SPX"), ("NIFTY50", "N225"), ("NIFTY50", "SSE"), ("SPX", "N225"), ("SPX", "SSE"),
                   ("N225", "SSE")]:
        macros[f"Corr{MKEY[a_]}{MKEY[b_]}"] = f"{C.loc[a_, b_]:.2f}"

    # ---------------------------------------------------------------- split windows (regime composition)
    rows = []
    for k in ["NIFTY100", "NIFTY500"]:
        pairs = make_pairs(frames[k], "ohlc", "levels", pr["feature_cols"], pr["add_weekday"])
        for tf_ in cfg["experiments"]["split_sweep"]["test_fracs"]:
            a, b, c = fixed_origin_window(pairs.n, tf_)
            lab, ret, _ = regime_label(pairs.Y_level[b:c, pairs.close_col])
            rows.append({"index": k, "split": f"{round(100 - 100 * tf_)}-{round(100 * tf_)}",
                         "test_start": pd.Timestamp(pairs.target_dates[b]).date(),
                         "test_end": pd.Timestamp(pairs.target_dates[c - 1]).date(), "n_test": c - b,
                         "train_sessions": b - a, "ann_return": round(ret, 4), "regime": lab})
            if k == "NIFTY100":
                s = SPLIT[tf_]
                macros.update({f"Split{s}Start": dfmt(pairs.target_dates[b]), f"Split{s}N": num(c - b),
                               f"Split{s}Train": num(b - a), f"Split{s}Ret": f"{100 * ret:.1f}\\%", f"Split{s}Regime": lab})
    sw = pd.DataFrame(rows)
    sw.to_csv(tdir / "split_windows.csv", index=False)
    regs = sw[sw["index"] == "NIFTY100"].regime
    macros["SplitEnd"] = dfmt(pd.Timestamp(sw[sw["index"] == "NIFTY100"].test_end.iloc[0]))
    macros["SplitRegimeSummary"] = (f"all four windows are {regs.iloc[0]} windows" if regs.nunique() == 1 else
                                    "the windows are labelled " + ", ".join(regs) + " respectively")

    # ---------------------------------------------------------------- walk-forward folds
    wf = cfg["walk_forward"]
    pairs = make_pairs(frames[wf["index"]], "ohlc", "levels", pr["feature_cols"], pr["add_weekday"])
    folds = walk_forward_windows(pairs.n, wf["train_window"], wf["test_window"], wf["step"])
    fr = []
    for k, (a, b, c) in enumerate(folds):
        lab, ret, vol = regime_label(pairs.Y_level[b:c, pairs.close_col], wf["bull_threshold"], wf["bear_threshold"])
        fr.append({"fold": k, "train_start": pairs.target_dates[a], "test_start": pairs.target_dates[b],
                   "test_end": pairs.target_dates[c - 1], "regime": lab, "ann_return": ret})
    fdf = pd.DataFrame(fr)
    fdf.to_csv(tdir / "walk_forward_folds.csv", index=False)
    nF = len(folds)
    cnt = fdf.regime.value_counts()
    bear = fdf[fdf.regime == "bear"]
    narch, nseed = len(wf["archs"]), len(wf["seeds"])
    macros.update({
        "WFFolds": str(nF), "WFFoldsWord": WORDS[nF] if nF < len(WORDS) else str(nF),
        "WFFoldsWordCap": (WORDS[nF] if nF < len(WORDS) else str(nF)).capitalize(),
        "WFRunsPerCell": num(narch * nF * nseed),
        "WFSpanStart": dfmt(fdf.train_start.iloc[0]), "WFFirstTest": dfmt(fdf.test_start.iloc[0]),
        "WFLastTest": dfmt(fdf.test_end.iloc[-1]),
        "WFTestYears": f"{pd.Timestamp(fdf.test_start.iloc[0]).year}--{pd.Timestamp(fdf.test_end.iloc[-1]).year}",
        "WFBull": str(int(cnt.get("bull", 0))), "WFSideways": str(int(cnt.get("sideways", 0))),
        "WFBear": str(int(cnt.get("bear", 0))),
        "WFBearList": "; ".join(f"{pd.Timestamp(r.test_start):%B %Y} -- {pd.Timestamp(r.test_end):%B %Y}"
                                for r in bear.itertuples()) or "none"})

    # ---------------------------------------------------------------- history ablation window
    ha = cfg["history_ablation"]
    hw = history_windows(pairs.n, ha["test_frac"], ha["history_lengths"])
    a, b, c = hw[-1]
    macros.update({"HistTestStart": dfmt(pairs.target_dates[b]), "HistTestEnd": dfmt(pairs.target_dates[c - 1]),
                   "HistTestN": num(c - b), "HistMax": num(b - a), "HistMaxYears": f"{(b - a) / 252:.0f}",
                   "HistLengths": ", ".join(num(x[1] - x[0]) for x in hw)})

    # ---------------------------------------------------------------- provenance
    if manifest:
        macros["DataProvenance"] = (f"downloaded from Yahoo Finance with \\texttt{{yfinance}} "
                                    f"(version {manifest['yfinance_version']}) on {manifest['downloaded_utc']} UTC, "
                                    f"from the oldest session Yahoo provides")
    else:
        macros["DataProvenance"] = "taken from a local data snapshot (refresh it with \\texttt{data/download\\_data.py})"
    wrap = (lambda v: v) if manifest else (lambda v: f"\\PH{{{v}}}")
    lines = ["% Generated by scripts/01_diagnostics.py -- do not edit by hand.",
             f"% provisional = {manifest is None}",
             "\\providecommand{\\PH}[1]{#1}"]
    for k, v in macros.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{wrap(v) if k != 'DataProvenance' else v}}}")
    (tdir / "data_facts.tex").write_text("\n".join(lines) + "\n")
    (tdir / "test_windows_rows.tex").write_text(rows_tex)
    (tdir / "dataset_stats_rows.tex").write_text("\n".join(stats_tex) + "\n")
    if args.paper_dir:
        for fn in ["data_facts.tex", "test_windows_rows.tex", "dataset_stats_rows.tex"]:
            shutil.copy(tdir / fn, Path(args.paper_dir) / fn)
    print(pd.DataFrame(diag)[["index", "vendor_first", "first", "last", "N", "dropped_before_start",
                              "flat_ohlc_vendor", "zero_volume_in_sample"]].to_string(index=False))
    print(w.to_string(index=False))
    print(sw.to_string(index=False))
    print(f"walk-forward: {nF} folds; regimes {cnt.to_dict()}")
    print("provisional macros" if manifest is None else "final macros", "->", tdir / "data_facts.tex")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Aggregate the run logs into every results table of the paper, and the result macros.

Writes results/tables/<table>.csv and a LaTeX body results/tables/<table>.tex (booktabs rows,
with a panel line per target) that the manuscript reads with \\inputrows. It also writes
results/tables/results_facts.tex: one \\newcommand per number quoted in the text. No result
number is typed by hand.

Every table reports BOTH representations side by side (levels | returns). Cells summarise the
finished runs of a configuration by their MEDIAN (the distribution of R^2 across seeds is
heavy-tailed: a single unstable run can reach R^2 < -100); diverged runs (non-finite loss or
forecasts) are excluded from every statistic and counted in a separate column. Means and
standard deviations are kept in the CSV files.

    table_test            fixed-origin NIFTY 100 (95-5): 4 archs, both representations, + benchmarks
    table_dm              Diebold-Mariano counts vs the random walk and the close-anchored naive
    table_skill           per-column skill (OHLC): where any gain over the random walk comes from
    table_capacity        default vs ablation width
    table_params_latency  parameters and serial (uncontended) inference latency
    table_split_sweep     80-20 ... 95-5 for CFC on NIFTY 100 / 500
    table_cross_market    NIFTY 50, S&P 500, Nikkei 225, SSE + benchmarks
    table_wf_summary      walk-forward, distribution over folds
    table_wf_regime       walk-forward by regime
    table_history         history-length ablation
    table_repr            levels vs returns across all experiments
    table_robustness      robustness of the headline to the protocol (runs/primary_window.csv)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cfcnet import cli  # noqa: E402
from cfcnet.io import REPO, runs_dir  # noqa: E402
from cfcnet.models import label  # noqa: E402

ORDER = {"cfc": 0, "ltc": 1, "lstm": 2, "gru": 3}
ARCHS = list(ORDER)
REPRS = ["levels", "returns"]
TLAB = {"adjclose": "Panel A: adjusted close $\\tilde{C}_{t+1}$", "ohlc": "Panel B: joint OHLC $(O,H,L,C)_{t+1}$"}
TORD = {"adjclose": 0, "ohlc": 1}
INAME = {"NIFTY50": "NIFTY 50", "NIFTY100": "NIFTY 100", "NIFTY500": "NIFTY 500", "SPX": "S\\&P 500",
         "N225": "Nikkei 225", "SSE": "SSE Composite"}
MOTOR = {"adjclose": 1, "ohlc": 4}
PAPER_DIR = None
FACTS = {}


# --------------------------------------------------------------------------- formatting
def num(x, d=3):
    """Compact number: 3 decimals near 1, fewer for very large magnitudes (failed runs)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "---"
    if not np.isfinite(x):
        return "---"
    a = abs(x)
    if a >= 1e4:
        e = int(np.floor(np.log10(a)))
        return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}"
    return f"{x:.{d}f}"


def sci(x):
    e = int(np.floor(np.log10(abs(float(x)))))
    return f"${float(x) / 10 ** e:.1f}\\times10^{{{e}}}$"


def pct(x):
    return "---" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"


def fact(name, value):
    """Register a result macro \\<name> for results_facts.tex."""
    FACTS[name] = value


def write(df: pd.DataFrame, name: str, outdir: Path, panel_col: str = "target"):
    df.to_csv(outdir / f"{name}.csv", index=False)
    cols = [c for c in df.columns if c != panel_col and not c.startswith("_")]
    lines = []
    groups = df.groupby(panel_col, sort=False) if panel_col in df else [(None, df)]
    for k, (tgt, g) in enumerate(groups):
        if tgt is not None:
            if k:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{len(cols)}}}{{@{{}}l}}{{\\textit{{{TLAB.get(tgt, tgt)}}}}} \\\\")
        for r in g.to_dict("records"):
            if r.get("_rule"):
                lines.append("\\cmidrule(lr){1-%d}" % len(cols))
            lines.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    (outdir / f"{name}.tex").write_text("% " + " & ".join(cols) + "\n" + "\n".join(lines) + "\n")
    if PAPER_DIR:
        (Path(PAPER_DIR) / f"{name}.tex").write_text((outdir / f"{name}.tex").read_text())
    print(f"--- {name}\n{df.drop(columns=[c for c in df.columns if c.startswith('_')]).to_string(index=False)}\n")


def summ(g: pd.DataFrame) -> dict:
    """Median R2, median MASE, mean DA over finished runs; diverged count."""
    ok = g[g.status == "ok"]
    return {"r2": ok.r2_joint.median(), "mase": ok.mase.median(), "da": ok.da.mean(),
            "r2_mean": ok.r2_joint.mean(), "r2_sd": ok.r2_joint.std(), "mase_mean": ok.mase.mean(),
            "r2ret": ok.r2_ret.median() if "r2_ret" in ok else np.nan,
            "div": int((g.status != "ok").sum()), "n": len(g), "n_ok": len(ok)}


def both(g: pd.DataFrame, cols=("r2", "mase", "da"), div=True) -> dict:
    """Side-by-side cells for the two representations of one configuration."""
    out = {}
    for rep in REPRS:
        s = summ(g[g["repr"] == rep]) if len(g[g["repr"] == rep]) else None
        for c in cols:
            v = None if s is None else s[c]
            out[f"{c}_{rep[:3]}"] = pct(v) if c == "da" else num(v)
        out[f"_{rep}"] = s
    if div:
        d = [out[f"_{r}"]["div"] if out[f"_{r}"] else 0 for r in REPRS]
        out["div"] = f"{d[0]}/{d[1]}"
    return out


def bench_row(target, lab, b, cols=("r2", "mase", "da"), extra=None, div=True):
    row = {"target": target, "model": lab, **(extra or {})}
    for rep in REPRS:
        for c in cols:
            if c == "r2":
                row[f"{c}_{rep[:3]}"] = num(b.r2_joint)
            elif c == "mase":
                row[f"{c}_{rep[:3]}"] = num(b.mase)
            elif c == "da":
                row[f"{c}_{rep[:3]}"] = "---" if b.model in ("rw", "naive_close") else pct(b.da)
    if div:
        row["div"] = "---"
    return row


BLAB = {"rw": "Random walk $y_t$", "naive_close": "Close-anchored naive $C_t$", "arima": "ARIMA"}


def benchmarks(bl, index, target, tf_):
    if bl is None:
        return []
    out = []
    for m in ["rw", "naive_close", "arima"]:
        b = bl[(bl["index"] == index) & (bl.target == target) & (bl.test_frac == tf_) & (bl.model == m)]
        if len(b):
            out.append(b.iloc[0])
    return out


# --------------------------------------------------------------------------- tables
def t_test(fo, bl, tdir):
    rows = []
    hl = fo[fo.experiment == "headline"]
    for target in sorted(hl.target.unique(), key=TORD.get):
        for arch in ARCHS:
            g = hl[(hl.target == target) & (hl.arch == arch)]
            if not len(g):
                continue
            units = int(g.units.iloc[0])
            c = both(g)
            rows.append({"target": target, "model": label(arch, units, MOTOR[target]),
                         "params": f"{int(g.n_params.iloc[0]):,}".replace(",", "{,}"),
                         **{k: v for k, v in c.items() if not k.startswith("_")}})
            for rep in REPRS:
                s = c[f"_{rep}"]
                tag = f"{target}{rep}{arch}".replace("adjclose", "AC").replace("ohlc", "OHLC")
                fact(f"HL{tag}RtwoMed", num(s["r2"]))
                fact(f"HL{tag}MASE", num(s["mase"]))
                fact(f"HL{tag}Rtwo", f"{num(s['r2_mean'])} $\\pm$ {num(s['r2_sd'])}")
                fact(f"HL{tag}Div", s["div"])
        for k, b in enumerate(benchmarks(bl, "NIFTY100", target, 0.05)):
            r = bench_row(target, BLAB[b.model], b, extra={"params": "---"})
            r["_rule"] = k == 0
            rows.append(r)
            tag = f"{'AC' if target == 'adjclose' else 'OHLC'}{b.model.replace('_', '')}"
            fact(f"HLbench{tag}Rtwo", num(b.r2_joint))
            fact(f"HLbench{tag}MASE", num(b.mase))
    if rows:
        df = pd.DataFrame(rows)
        df = df[["target", "model", "params", "r2_lev", "mase_lev", "da_lev", "r2_ret", "mase_ret", "da_ret", "div",
                 "_rule"]].fillna({"_rule": False})
        write(df, "table_test", tdir)


def t_dm(dmr, sk, fo, wf, tdir):
    if dmr is None:
        return
    meta = pd.concat([fo[["run_id", "experiment"]], wf[["run_id"]].assign(experiment="walk_forward")]
                     if wf is not None else [fo[["run_id", "experiment"]]])
    d = dmr.merge(meta, on="run_id", how="left")
    if sk is not None:
        d = d.merge(sk[["run_id", "dm_naive_stat", "dm_naive_p"]], on="run_id", how="left")
    rows = []
    for target in ["adjclose", "ohlc"]:
        for rep in REPRS:
            for arch in ARCHS:
                row = {"target": target, "repr": rep if arch == "cfc" else "", "model": arch.upper()}
                for src, lab in [("headline", "hl"), ("cross_market", "cm"), ("walk_forward", "wf")]:
                    g = d[(d.target == target) & (d["repr"] == rep) & (d.arch == arch) & (d.experiment == src)]
                    b, w = int(g.significant_5pct.sum()), int(g.rw_significantly_better.sum())
                    row[f"{lab}_rw"] = f"{b} / {w} ({len(g)})"
                    if target == "ohlc" and sk is not None:
                        nb = int(((g.dm_naive_stat < 0) & (g.dm_naive_p < 0.05)).sum())
                        nw = int(((g.dm_naive_stat > 0) & (g.dm_naive_p < 0.05)).sum())
                        row[f"{lab}_nv"] = f"{nb} / {nw}"
                    else:
                        row[f"{lab}_nv"] = "---"
                    fact(f"DM{lab}{target}{rep}{arch}".replace("adjclose", "AC").replace("ohlc", "OHLC") + "better", b)
                row["_rule"] = arch == "cfc" and rep == "returns"
                rows.append(row)
    # global counts quoted in the text
    nets = d[d.experiment.notna()]
    fact("DMtotalRuns", f"{len(nets):,}".replace(",", "{,}"))
    close_like = nets[nets.target == "adjclose"]
    fact("DMadjcloseRuns", f"{len(close_like):,}".replace(",", "{,}"))
    fact("DMadjcloseSigBetter", int(close_like.significant_5pct.sum()))
    t5 = close_like[close_like.experiment.isin(["headline", "cross_market", "walk_forward"])]
    fact("DMadjcloseRunsTableFive", f"{len(t5):,}".replace(",", "{,}"))
    hl_oh = nets[(nets.target == "ohlc") & (nets["repr"] == "returns") & (nets.experiment == "headline")]
    if sk is not None:
        hh = hl_oh.merge(sk[["run_id", "dm_naive_stat", "dm_naive_p"]].rename(columns={"dm_naive_stat": "s2", "dm_naive_p": "p2"}),
                         on="run_id", how="left") if "dm_naive_stat" not in hl_oh else hl_oh
        c = hh.groupby("arch").apply(lambda g: int(((g.dm_naive_stat < 0) & (g.dm_naive_p < 0.05)).sum()))
        fact("DMhlNaiveBetterMin", int(c.min()))
        fact("DMhlNaiveBetterMax", int(c.max()))
    df = pd.DataFrame(rows)[["target", "repr", "model", "hl_rw", "hl_nv", "cm_rw", "cm_nv", "wf_rw", "wf_nv", "_rule"]]
    write(df, "table_dm", tdir)


def t_skill(sk, fo, wf, tdir):
    if sk is None:
        return
    s = sk[sk.target == "ohlc"]
    hl = s.merge(fo[["run_id", "experiment"]], on="run_id")
    hl = hl[hl.experiment == "headline"]
    w = s[s.source == "walk_forward"]
    rows = []
    for rep in REPRS:
        for arch in ARCHS:
            g, gw = hl[(hl["repr"] == rep) & (hl.arch == arch)], w[(w["repr"] == rep) & (w.arch == arch)]
            row = {"repr": rep if arch == "cfc" else "", "model": arch.upper()}
            for c in ["open", "high", "low", "close"]:
                row[f"hl_{c}"] = num(g[f"mase_{c}"].median())
            row["hl_nv"] = num(g.rel_naive.median())
            row["wf_close"] = num(gw.mase_close.median())
            row["wf_nv"] = num(gw.rel_naive.median())
            row["_rule"] = arch == "cfc" and rep == "returns"
            rows.append(row)
            tag = f"{rep}{arch}"
            for c in ["open", "high", "low", "close"]:
                fact(f"SK{tag}{c}", num(g[f"mase_{c}"].median()))
            fact(f"SK{tag}naive", num(g.rel_naive.median()))
            fact(f"SKwf{tag}naive", num(gw.rel_naive.median()))
            fact(f"SKwf{tag}close", num(gw.mase_close.median()))
    b = s[(s.source == "baseline") & s.run_id.str.contains("_NIFTY100_ohlc_tf5_")]
    for k, m in enumerate(["naive_close", "arima"]):
        g = b[b.arch == m]
        if len(g):
            g = g.iloc[0]
            rows.append({"repr": "", "model": BLAB[m], **{f"hl_{c}": num(g[f"mase_{c}"]) for c in ["open", "high", "low", "close"]},
                         "hl_nv": num(g.rel_naive), "wf_close": "---", "wf_nv": "---", "_rule": k == 0})
            fact(f"SKbench{m.replace('_', '')}open", num(g.mase_open))
    g_wf = w[w["repr"] == "returns"].groupby("arch").rel_naive.median()
    fact("SKwfGainMin", f"{100 * (1 - g_wf.max()):.0f}")
    fact("SKwfGainMax", f"{100 * (1 - g_wf.min()):.0f}")
    hl_r = hl[hl["repr"] == "returns"].groupby("arch").rel_naive.median()
    fact("SKhlRelMin", num(hl_r.min()))
    fact("SKhlRelMax", num(hl_r.max()))
    lat_ratio = None
    wn = w.groupby("run_id").naive_vs_rw.first()
    fact("SKwfNaiveVsRW", num(wn.median()))
    write(pd.DataFrame(rows), "table_skill", tdir, panel_col="__none__")


def t_capacity(fo, tdir):
    cap = fo[fo.experiment.isin(["headline", "capacity"])]
    rows = []
    for target in ["adjclose", "ohlc"]:
        for arch in ARCHS:
            for units in sorted(cap[(cap.target == target) & (cap.arch == arch)].units.unique(), reverse=True):
                g = cap[(cap.target == target) & (cap.arch == arch) & (cap.units == units)]
                c = both(g, cols=("r2", "mase"))
                rows.append({"target": target, "model": label(arch, int(units), MOTOR[target]),
                             "params": f"{int(g.n_params.iloc[0]):,}".replace(",", "{,}"),
                             **{k: v for k, v in c.items() if not k.startswith("_")}})
                for rep in REPRS:
                    fact(f"CAP{target}{rep}{arch}{int(units)}MASE".replace("adjclose", "AC").replace("ohlc", "OHLC"),
                         num(c[f"_{rep}"]["mase"]))
                    fact(f"CAP{target}{rep}{arch}{int(units)}Div".replace("adjclose", "AC").replace("ohlc", "OHLC"),
                         c[f"_{rep}"]["div"])
    write(pd.DataFrame(rows), "table_capacity", tdir)
    # parameter-accuracy frontier: configurations that no other configuration beats on both counts
    for target in ["adjclose", "ohlc"]:
        for rep in REPRS:
            g = cap[(cap.target == target) & (cap["repr"] == rep) & (cap.status == "ok")]
            pts = g.groupby(["arch", "units", "n_params"]).mase.median().reset_index()
            front = []
            for r in pts.itertuples():
                dominated = ((pts.n_params <= r.n_params) & (pts.mase <= r.mase) &
                             ((pts.n_params < r.n_params) | (pts.mase < r.mase))).any()
                if not dominated:
                    front.append((r.n_params, label(r.arch, int(r.units), MOTOR[target])))
            tag = f"{target}{rep}".replace("adjclose", "AC").replace("ohlc", "OHLC")
            fact(f"FRONT{tag}", ", ".join(l for _, l in sorted(front)))


def t_params_latency(fo, lat, tdir):
    cap = fo[fo.experiment.isin(["headline", "capacity"])]
    rows = []
    for target in ["adjclose", "ohlc"]:
        for arch in ARCHS:
            for units in sorted(cap[(cap.target == target) & (cap.arch == arch)].units.unique(), reverse=True):
                g = cap[(cap.target == target) & (cap.arch == arch) & (cap.units == units)]
                n = int(g.n_params.iloc[0])
                row = {"target": target, "model": label(arch, int(units), MOTOR[target]),
                       "params": f"{n:,}".replace(",", "{,}"), "ms_pass": "---", "p10_p90": "---", "ms_session": "---"}
                if lat is not None:
                    L = lat[(lat.target == target) & (lat.arch == arch) & (lat.units == units)]
                    if len(L):
                        L = L.iloc[0]
                        row.update(ms_pass=f"{L.median_ms:.1f}", p10_p90=f"{L.p10_ms:.1f}--{L.p90_ms:.1f}",
                                   ms_session=f"{L.ms_per_session:.3f}")
                        fact(f"LAT{target}{arch}{int(units)}".replace("adjclose", "AC").replace("ohlc", "OHLC"),
                             f"{L.ms_per_session:.3f}")
                fact(f"PAR{target}{arch}{int(units)}".replace("adjclose", "AC").replace("ohlc", "OHLC"),
                     f"{n:,}".replace(",", "{,}"))
                rows.append(row)
    p = {(t, a): int(fo[(fo.experiment == "headline") & (fo.target == t) & (fo.arch == a)].n_params.iloc[0])
         for t in ["ohlc", "adjclose"] for a in ARCHS}
    fact("PARcfcVsLstmPct", f"{100 * (1 - p['ohlc', 'cfc'] / p['ohlc', 'lstm']):.0f}")
    fact("PARcfcVsGruPct", f"{100 * (1 - p['ohlc', 'cfc'] / p['ohlc', 'gru']):.0f}")
    fact("PARlstmOverCfc", f"{p['ohlc', 'lstm'] / p['ohlc', 'cfc']:.2f}")
    if lat is not None:
        L = lambda a, u: float(lat[(lat.target == "ohlc") & (lat.arch == a) & (lat.units == u)].ms_per_session.iloc[0])
        fact("LATratioCfcLstm", f"{L('cfc', 80) / L('lstm', 80):.1f}")
        fact("LATratioCfcGru", f"{L('cfc', 80) / L('gru', 80):.1f}")
        fact("LATratioLtcLstm", f"{L('ltc', 80) / L('lstm', 80):.1f}")
        fact("LATdevice", str(lat.device.iloc[0]))
    write(pd.DataFrame(rows), "table_params_latency", tdir)


def t_split(fo, bl, tdir):
    sw = fo[fo.experiment.isin(["split_sweep"])]
    rows = []
    for target in ["adjclose", "ohlc"]:
        for index in sorted(sw["index"].unique()):
            for tf_ in sorted(sw.test_frac.unique(), reverse=True):
                g = sw[(sw.target == target) & (sw["index"] == index) & (sw.test_frac == tf_)]
                if not len(g):
                    continue
                c = both(g, cols=("r2", "mase"))
                for rep in REPRS:
                    fact(f"SP{target}{index}tf{round(tf_ * 100)}{rep}MASE".replace("adjclose", "AC").replace("ohlc", "OHLC"),
                         num(c[f"_{rep}"]["mase"]))
                rw = [b for b in benchmarks(bl, index, target, tf_) if b.model == "rw"]
                r0 = g.iloc[0]
                rows.append({"target": target, "index": INAME[index],
                             "split": f"{round(100 - 100 * tf_)}--{round(100 * tf_)}",
                             "test": f"{pd.Timestamp(r0.test_start):%b %Y}--{pd.Timestamp(r0.test_end):%b %Y}",
                             "rw_r2": num(rw[0].r2_joint) if rw else "---",
                             **{k: v for k, v in c.items() if not k.startswith("_")}})
    write(pd.DataFrame(rows), "table_split_sweep", tdir)


def t_cross(fo, bl, tdir):
    cm = fo[fo.experiment == "cross_market"]
    rows = []
    for target in ["adjclose", "ohlc"]:
        for index in ["NIFTY50", "SPX", "N225", "SSE"]:
            first = True
            for arch in ARCHS:
                g = cm[(cm.target == target) & (cm["index"] == index) & (cm.arch == arch)]
                if not len(g):
                    continue
                c = both(g, cols=("r2", "mase"))
                rows.append({"target": target, "index": INAME[index] if first else "", "model": arch.upper(),
                             **{k: v for k, v in c.items() if not k.startswith("_")}, "_rule": first})
                first = False
            for b in benchmarks(bl, index, target, 0.05):
                if b.model == "arima":
                    continue
                rows.append({"target": target, "index": "", **bench_row(target, BLAB[b.model], b, cols=("r2", "mase")),
                             "_rule": False})
    for target in ["adjclose", "ohlc"]:
        for rep in REPRS:
            for index in ["NIFTY50", "SPX", "N225", "SSE", "NIFTY100"]:
                src = fo[(fo.experiment == ("headline" if index == "NIFTY100" else "cross_market"))]
                g = src[(src.target == target) & (src["repr"] == rep) & (src["index"] == index) & (src.status == "ok")]
                m = g.groupby("arch").mase.median().sort_values()
                tag = f"BEST{target}{rep}{index}".replace("adjclose", "AC").replace("ohlc", "OHLC")
                fact(tag, m.index[0].upper())
                fact(tag + "Worst", m.index[-1].upper())
    cells = [(t, r, i) for t in ["AC", "OHLC"] for r in REPRS for i in ["N225", "NIFTY100",
                                                                          "NIFTY50", "SPX", "SSE"]]
    for r in REPRS:
        cs = [c for c in cells if c[1] == r]
        fact(f"NCELLS{r}", len(cs))
        for a in ARCHS:
            fact(f"NBEST{r}{a}", sum(FACTS[f"BEST{t}{rr}{i}"] == a.upper() for t, rr, i in cs))
            fact(f"NWORST{r}{a}", sum(FACTS[f"BEST{t}{rr}{i}Worst"] == a.upper() for t, rr, i in cs))
    df = pd.DataFrame(rows)
    df = df[["target", "index", "model", "r2_lev", "mase_lev", "r2_ret", "mase_ret", "div", "_rule"]]
    write(df, "table_cross_market", tdir)


def per_fold(w):
    """One value per fold and configuration: the median over seeds (finished runs)."""
    ok = w[w.status == "ok"]
    return ok.groupby(["target", "repr", "arch", "fold", "regime"]).agg(
        r2=("r2_joint", "median"), mase=("mase", "median")).reset_index()


def t_wf(wf, tdir):
    pf = per_fold(wf)
    rows, rows2 = [], []
    for target in ["adjclose", "ohlc"]:
        for arch in ARCHS:
            row = {"target": target, "model": label(arch, int(wf[(wf.target == target) & (wf.arch == arch)].units.iloc[0]),
                                                    MOTOR[target])}
            row2 = dict(row)
            for rep in REPRS:
                g = pf[(pf.target == target) & (pf["repr"] == rep) & (pf.arch == arch)]
                worst = g.loc[g.r2.idxmin()]
                r = rep[:3]
                row[f"r2_{r}"] = num(g.r2.median())
                row[f"mase_{r}"] = num(g.mase.median())
                row[f"worst_{r}"] = f"{int(worst.fold)}: {num(worst.r2)}"
                row[f"div_{r}"] = int(((wf.target == target) & (wf["repr"] == rep) & (wf.arch == arch) &
                                       (wf.status != "ok")).sum())
                tag = f"{target}{rep}{arch}".replace("adjclose", "AC").replace("ohlc", "OHLC")
                fact(f"WF{tag}MASE", num(g.mase.median()))
                fact(f"WF{tag}WorstFold", int(worst.fold))
                fact(f"WF{tag}WorstRtwo", num(worst.r2))
                for reg in ["bull", "sideways"]:
                    x = g[g.regime == reg].mase
                    row2[f"{reg}_{r}"] = num(x.median()) if len(x) else "---"
            rows.append(row)
            rows2.append(row2)
    cols = ["target", "model", "r2_lev", "mase_lev", "worst_lev", "div_lev", "r2_ret", "mase_ret", "worst_ret", "div_ret"]
    write(pd.DataFrame(rows)[cols], "table_wf_summary", tdir)
    write(pd.DataFrame(rows2)[["target", "model", "bull_lev", "sideways_lev", "bull_ret", "sideways_ret"]],
          "table_wf_regime", tdir)
    folds = wf.drop_duplicates("fold")
    fact("WFbullFolds", int((folds.regime == "bull").sum()))
    fact("WFsidewaysFolds", int((folds.regime == "sideways").sum()))
    fact("WFbearFolds", int((folds.regime == "bear").sum()))
    # the fold-11 (or worst) CFC episode, per seed, levels
    for target in ["adjclose", "ohlc"]:
        g = wf[(wf.target == target) & (wf["repr"] == "levels") & (wf.arch == "cfc")]
        k = int(pf[(pf.target == target) & (pf["repr"] == "levels") & (pf.arch == "cfc")].sort_values("r2").fold.iloc[0])
        e = g[g.fold == k].sort_values("seed")
        vals = ", ".join(num(v) if s == "ok" else "diverged" for v, s in zip(e.r2_joint, e.status))
        fact(f"WFcfcEpisode{'AC' if target == 'adjclose' else 'OHLC'}", vals)
        fact(f"WFcfcEpisodeFold{'AC' if target == 'adjclose' else 'OHLC'}", k)
        r = wf[(wf.fold == k)].iloc[0]
        fact("WFepisodeTest", f"{pd.Timestamp(r.test_start):%d %B %Y}--{pd.Timestamp(r.test_end):%d %B %Y}")
        ret_g = wf[(wf.target == target) & (wf["repr"] == "returns") & (wf.arch == "cfc") & (wf.fold == k) & (wf.status == "ok")]
        fact(f"WFcfcEpisodeRet{'AC' if target == 'adjclose' else 'OHLC'}MASE", num(ret_g.mase.median()))
    # systematic divergence of one seed
    dv = wf[wf.status != "ok"]
    fact("WFdivRuns", len(dv))
    fact("WFdivSeeds", ", ".join(str(s) for s in sorted(dv.seed.unique())))


def t_history(ha, tdir):
    ok = ha[ha.status == "ok"]
    rows = []
    for target in ["adjclose", "ohlc"]:
        for L in sorted(ha.hist_len.unique()):
            row = {"target": target, "years": f"{L / 252:.1f}", "sessions": f"{int(L):,}".replace(",", "{,}")}
            for rep in REPRS:
                for arch in ["cfc", "gru"]:
                    g = ha[(ha.target == target) & (ha["repr"] == rep) & (ha.arch == arch) & (ha.hist_len == L)]
                    o = g[g.status == "ok"]
                    d = int((g.status != "ok").sum())
                    row[f"{arch}_{rep[:3]}"] = num(o.mase.median()) + (f"$^{{{d}}}$" if d else "")
            rows.append(row)
    write(pd.DataFrame(rows), "table_history", tdir)
    for target in ["adjclose", "ohlc"]:
        for rep in REPRS:
            for arch in ["cfc", "gru"]:
                g = ok[(ok.target == target) & (ok["repr"] == rep) & (ok.arch == arch)].groupby("hist_len").mase.median()
                tag = f"{target}{rep}{arch}".replace("adjclose", "AC").replace("ohlc", "OHLC")
                fact(f"HA{tag}Short", num(g.iloc[0]))
                fact(f"HA{tag}Long", num(g.iloc[-1]))


def t_repr(fo, wf, tdir):
    rows = []
    for target in ["adjclose", "ohlc"]:
        for arch in ARCHS:
            row = {"target": target, "model": arch.upper()}
            for rep in REPRS:
                g = fo[(fo.target == target) & (fo.arch == arch) & (fo["repr"] == rep) & (fo.status == "ok")]
                row[f"hl_{rep[:3]}"] = num(g[g.experiment == "headline"].mase.median())
                cm = g[g.experiment == "cross_market"].groupby("index").mase.median()
                row[f"cm_{rep[:3]}"] = num(cm.median()) if len(cm) else "---"
                w = wf[(wf.target == target) & (wf.arch == arch) & (wf["repr"] == rep) & (wf.status == "ok")]
                row[f"wf_{rep[:3]}"] = num(w.mase.median())
            rows.append(row)
    write(pd.DataFrame(rows)[["target", "model", "hl_lev", "cm_lev", "wf_lev", "hl_ret", "cm_ret", "wf_ret"]],
          "table_repr", tdir)
    # range of median MASE across all (experiment, index, arch) cells, by representation
    for rep in REPRS:
        g = fo[(fo["repr"] == rep) & (fo.status == "ok") & (fo.experiment.isin(["headline", "cross_market"]))]
        cell = g.groupby(["target", "experiment", "index", "arch"]).mase.median()
        for target in ["adjclose", "ohlc"]:
            c = cell.xs(target, level="target")
            tag = f"{target}{rep}".replace("adjclose", "AC").replace("ohlc", "OHLC")
            fact(f"RNG{tag}Min", num(c.min()))
            fact(f"RNG{tag}Max", num(c.max()))




def t_primary(rr, tdir):
    """Headline on the primary NIFTY 100 sample (to 16 July 2024; 95-5 test window August 2023 - July 2024) under
    the one-step-ahead protocol: ten seeds per architecture, levels, both targets. Mean +- sd of R^2 over seeds,
    best run, RMSE and MAE (scaled), MASE, directional accuracy and Diebold-Mariano counts against the random walk."""
    if rr is None:
        return
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    out = runs_dir(None)
    dm = pd.read_csv(out / "dm_vs_rw.csv") if (out / "dm_vs_rw.csv").exists() else None
    g0 = rr[(rr.timing == "next_day") & (rr.scaler_fit == "train") & (rr["data"] == "primary")]
    t0 = g0.iloc[0]
    for k, c in [("TrainStart", "train_start"), ("TrainEnd", "train_end"), ("TestStart", "test_start"), ("TestEnd", "test_end")]:
        fact(f"PR{k}", f"{pd.Timestamp(t0[c]):%-d %B %Y}")
    fact("PRnTrain", f"{int(t0.n_train):,}".replace(",", "{,}"))
    fact("PRnTest", int(t0.n_test))
    fact("NNiftyHundredPrimary", f"{int(t0.n_train) + int(t0.n_test) + 2:,}".replace(",", "{,}"))
    rows = []
    for target in ["adjclose", "ohlc"]:
        tag = "OHLC" if target == "ohlc" else "AC"
        for arch in ARCHS:
            g = g0[(g0.target == target) & (g0.arch == arch)]
            ok = g[g.status == "ok"]
            units = int(ok.units.iloc[0])
            k = 4 if target == "ohlc" else 1
            label = (f"{arch.upper()} $({units}, {k})$" if arch in ("cfc", "ltc") else f"{arch.upper()}-{units}")
            row = {"target": target, "model": label, "r2": num(ok.r2_joint.median()), "best": num(ok.r2_joint.max()),
                   "rmse": f"{ok.rmse_joint.median():.4f}", "mae": f"{ok.mae_joint.median():.4f}",
                   "mase": num(ok.mase.median()), "da": f"{100 * ok.da.mean():.1f}", "_rule": False}
            if dm is not None:
                d = dm[dm.run_id.isin(ok.run_id)]
                row["dm"] = f"{int(d.rw_significantly_better.sum())}/{len(d)}"
            rows.append(row)
            fact(f"PR{tag}{arch}Rtwo", num(ok.r2_joint.median()))
            fact(f"PR{tag}{arch}RtwoMean", f"{ok.r2_joint.mean():.3f}")
            fact(f"PR{tag}{arch}RtwoBest", f"{ok.r2_joint.max():.3f}")
            fact(f"PR{tag}{arch}MASE", num(ok.mase.median()))
            fact(f"PR{tag}{arch}DA", f"{100 * ok.da.mean():.1f}")
            fact(f"PR{tag}{arch}N", len(ok))
            fact(f"PR{tag}{arch}Div", int((g.status != "ok").sum()))
            if target == "ohlc":
                fact(f"PR{tag}{arch}OpenBest", f"{ok.r2_open.max():.3f}")
                fact(f"PR{tag}{arch}OpenMean", f"{ok.r2_open.mean():.3f}")
        # random-walk row from the stored benchmark forecasts of one run
        z = np.load(out / "predictions" / f"{ok.run_id.iloc[0]}.npz")
        y, rw = z["actual_scaled"], z["rw_scaled"]
        r2 = r2_score(y.reshape(-1), rw.reshape(-1))
        rows.append({"target": target, "model": "Random walk", "r2": f"{r2:.3f}", "best": "---",
                     "rmse": f"{np.sqrt(mean_squared_error(y, rw)):.4f}", "mae": f"{mean_absolute_error(y, rw):.4f}",
                     "mase": "1", "da": "---", "dm": "---", "_rule": True})
        fact(f"PR{tag}rwRtwo", f"{r2:.3f}")
        # ranking by R^2 and by MASE
        m = g0[(g0.target == target) & (g0.status == "ok")].groupby("arch")
        fact(f"PR{tag}rankRtwo", ", ".join(a.upper() for a in m.r2_joint.median().sort_values(ascending=False).index))
        fact(f"PR{tag}rankMASE", ", ".join(a.upper() for a in m.mase.median().sort_values().index))
        cfc = g0[(g0.target == target) & (g0.arch == "cfc") & (g0.status == "ok")]
        lstm = g0[(g0.target == target) & (g0.arch == "lstm") & (g0.status == "ok")]
        gru = g0[(g0.target == target) & (g0.arch == "gru") & (g0.status == "ok")]
        fact(f"PR{tag}gainLSTM", f"{cfc.r2_joint.median() - lstm.r2_joint.median():.3f}")
        fact(f"PR{tag}gainGRU", f"{cfc.r2_joint.median() - gru.r2_joint.median():.3f}")
        fact(f"PR{tag}maseGainGRU", f"{100 * (1 - cfc.mase.median() / gru.mase.median()):.0f}")
        from scipy.stats import mannwhitneyu
        pv = mannwhitneyu(cfc.mase, gru.mase, alternative="less").pvalue
        fact(f"PR{tag}mwP", "$<$0.001" if pv < 0.001 else f"{pv:.3f}")
    write(pd.DataFrame(rows), "table_primary", tdir)



def t_robustness(rr, fo, tdir):
    """Robustness of the headline (Section 6.4): same-session target, full-series scaler, single column and
    best run, on the NIFTY 100 primary sample ending 16 July 2024."""
    rows = []
    for target in ["adjclose", "ohlc"]:
        for arch in ARCHS:
            row = {"target": target, "model": arch.upper()}
            if rr is not None:
                for (timing, scaler), lab in [(("same_day", "full"), "v1"), (("same_day", "train"), "v2"),
                                              (("next_day", "train"), "v3")]:
                    g = rr[(rr.target == target) & (rr.arch == arch) & (rr.timing == timing) & (rr.scaler_fit == scaler)
                           & (rr["data"] == "primary")]
                    ok = g[g.status == "ok"]
                    col = "r2_open" if target == "ohlc" else "r2_adjclose"
                    if lab == "v1":
                        row["v1_open_mean"] = num(ok[col].mean(), 4)
                        row["v1_open_best"] = num(ok[col].max(), 4)
                        row["v1_joint"] = num(ok.r2_joint.median())
                    else:
                        row[f"{lab}_joint"] = num(ok.r2_joint.median())
                    if lab == "v3":
                        row["v3_mase"] = num(ok.mase.median())
                    tag = f"{target}{arch}{lab}".replace("adjclose", "AC").replace("ohlc", "OHLC")
                    fact(f"RR{tag}Joint", num(ok.r2_joint.median()))
                    fact(f"RR{tag}Open", num(ok[col].mean(), 4))
                    fact(f"RR{tag}OpenBest", num(ok[col].max(), 4))
                    fact(f"RR{tag}MASE", num(ok.mase.median()))
                    fact(f"RR{tag}Div", int((g.status != "ok").sum()))
            else:
                for c in ["v1_open_mean", "v1_open_best", "v1_joint", "v2_joint", "v3_joint", "v3_mase"]:
                    row[c] = "\\TBD"
            g = fo[(fo.experiment == "headline") & (fo.target == target) & (fo.arch == arch) & (fo["repr"] == "levels")
                   & (fo.status == "ok")]
            row["r2_joint"] = num(g.r2_joint.median())
            row["r2_mase"] = num(g.mase.median())
            rows.append(row)
    write(pd.DataFrame(rows), "table_robustness", tdir)
    if rr is None:
        return
    ok = rr[rr.status == "ok"]
    fact("RRruns", len(rr))
    fact("RRok", len(ok))
    fact("RRdiv", int((rr.status != "ok").sum()))
    out = runs_dir(None)
    dm = pd.read_csv(out / "dm_vs_rw.csv") if (out / "dm_vs_rw.csv").exists() else None
    if dm is not None:
        d = dm[dm.source == "primary_window"]
        fact("RRdmBetter", int(d.significant_5pct.sum()))
        fact("RRdmWorse", int(d.rw_significantly_better.sum()))
    # the primary test window: random-walk R2 and how far it lies above the training range
    for timing in ["same_day", "next_day"]:
        g = ok[(ok.timing == timing) & (ok.target == "ohlc") & (ok.scaler_fit == "train")]
        z = np.load(out / "predictions" / f"{g.run_id.iloc[0]}.npz")
        y, rw = z["actual_scaled"], z["rw_scaled"]
        from sklearn.metrics import r2_score
        tag = "Same" if timing == "same_day" else "Next"
        fact(f"RRrwRtwo{tag}", num(r2_score(y.reshape(-1), rw.reshape(-1))))
        fact(f"RRwindowMax{tag}", f"{y[:, -1].max():.2f}")
        fact(f"RRwindowShare{tag}", f"{100 * (y[:, -1] > 1).mean():.0f}")
    t0 = ok.iloc[0]
    fact("RRtest", f"{pd.Timestamp(t0.test_start):%-d %B %Y}--{pd.Timestamp(t0.test_end):%-d %B %Y}")
    fact("RRnTest", int(t0.n_test))
    # ranking of the levels models on the ablation window under the present protocol (median MASE)
    for target in ["ohlc", "adjclose"]:
        g = ok[(ok.timing == "next_day") & (ok.target == target)].groupby("arch").mase.median().sort_values()
        tag = "OHLC" if target == "ohlc" else "AC"
        fact(f"RRrank{tag}", ", ".join(f"{a.upper()} {num(v)}" for a, v in g.items()))
        fact(f"RRbest{tag}", g.index[0].upper())
    # MASE range of CFC and GRU over the three protocols (OHLC)
    g = ok[(ok.target == "ohlc") & ok.arch.isin(["cfc", "gru"])].groupby(["arch", "timing", "scaler_fit"]).mase.median()
    fact("RRmaseMinCG", num(g.min()))
    fact("RRmaseMaxCG", num(g.max()))
    # change of the median joint R2 between protocol (1) and protocol (3)
    j = ok.groupby(["target", "arch", "timing", "scaler_fit"]).r2_joint.median()
    shift = {(t, a): abs(j[(t, a, "same_day", "full")] - j[(t, a, "next_day", "train")])
             for t in ["ohlc", "adjclose"] for a in ARCHS}
    fact("RRshiftMaxCG", f"{max(v for (t, a), v in shift.items() if a in ('cfc', 'gru')):.3f}")
    fact("RRshiftLSTM", f"{max(v for (t, a), v in shift.items() if a == 'lstm'):.3f}")


def data_facts(fo, bl, wf):
    """Window-level facts used in the text (test windows relative to the training range)."""
    for tf_ in [0.05, 0.10, 0.15, 0.20]:
        for index in ["NIFTY100", "NIFTY500"]:
            b = [x for x in benchmarks(bl, index, "ohlc", tf_) if x.model == "rw"]
            if b:
                fact(f"RW{index}tf{round(tf_ * 100)}Rtwo", num(b[0].r2_joint))
    fact("FOtotalRuns", f"{len(fo):,}".replace(",", "{,}"))
    fact("FOdivRuns", int((fo.status != "ok").sum()))
    fact("FOdivCfcRuns", int(((fo.status != "ok") & (fo.arch == "cfc")).sum()))
    fact("WFtotalRuns", f"{len(wf):,}".replace(",", "{,}"))
    adf = pd.read_csv(REPO / "runs" / "adf_per_fold.csv") if (REPO / "runs" / "adf_per_fold.csv").exists() else None
    if adf is not None:
        fact("ADFfoldsIone", int(((adf.adf_level_p > 0.05) & (adf.adf_diff_p < 0.05)).sum()))
        fact("ADFminLevelP", f"{adf.adf_level_p.min():.2f}")
        fact("ADFmaxDiffPexp", int(np.floor(np.log10(max(adf.adf_diff_p.max(), 1e-300)))) + 1)


def extra_facts(fo, out, fo_distinct=None):
    """Facts that need the saved predictions or histories: extrapolation beyond the training range,
    the anatomy of the diverged runs, directional accuracy."""
    import json
    ok = fo[fo.status == "ok"]
    seed0 = ok[(ok.target == "ohlc") & (ok["repr"] == "levels")].sort_values("seed").drop_duplicates(
        ["experiment", "index", "test_frac"])
    for r in seed0.itertuples():
        a = np.load(out / "predictions" / f"{r.run_id}.npz")["actual_scaled"][:, -1]
        tag = f"{r.experiment}{r.index}tf{round(r.test_frac * 100)}".replace("_", "")
        fact(f"EX{tag}Share", f"{100 * (a > 1).mean():.0f}")
        fact(f"EX{tag}Max", f"{a.max():.2f}")
    fd = fo if fo_distinct is None else fo_distinct
    d = fd[fd.status != "ok"]
    fact("DIVarchs", ", ".join(sorted(d.arch.str.upper().unique())))
    fact("DIVcfcShare", f"{100 * len(d[d.arch == 'cfc']) / max(1, (fd.arch == 'cfc').sum()):.1f}")
    fact("DIVcfcRuns", f"{int((fd.arch == 'cfc').sum()):,}".replace(",", "{,}"))
    cell = fd[fd.arch == "cfc"].groupby(["experiment", "index", "target", "repr", "units", "test_frac"]).apply(
        lambda g: int((g.status != "ok").sum()))
    fact("DIVcfcMaxPerCell", int(cell.max()))
    first_nan = []
    for rid in d.run_id:
        h = out / "histories" / f"{rid}.json"
        if h.exists():
            L = np.array(json.loads(h.read_text())["loss"], dtype=float)
            first_nan.append(int(np.argmax(~np.isfinite(L))) + 1)
    if first_nan:
        fact("DIVlastEpoch", max(first_nan))
        fact("DIVwithinTwelveShare", f"{100 * np.mean(np.array(first_nan) <= 13):.0f}")
    firsts_div, firsts_ok = [], []
    for rid, st in zip(fd[fd.arch == "cfc"].run_id, fd[fd.arch == "cfc"].status):
        h = out / "histories" / f"{rid}.json"
        if h.exists():
            v = json.loads(h.read_text())["loss"][0]
            v = float("nan") if v is None else float(v)
            (firsts_div if st != "ok" else firsts_ok).append(v)
    fd_ = np.array(firsts_div, float)
    fact("DIVfirstLossMedian", num(np.nanmedian(fd_)) if np.isfinite(fd_).any() else "---")
    fact("DIVfirstLossNaN", int((~np.isfinite(fd_)).sum()))
    fact("OKfirstLossMedian", num(np.median(firsts_ok)))
    fact("OKfirstLossMax", num(np.max(firsts_ok)))
    hlall = fo[(fo.experiment == "headline") & (fo.status == "ok")]
    for (target, rep, arch), g in hlall.groupby(["target", "repr", "arch"]):
        tag = f"LL{target}{rep}{arch}".replace("adjclose", "AC").replace("ohlc", "OHLC")
        e = int(np.floor(np.log10(g.last_loss.median())))
        fact(tag, f"${g.last_loss.median() / 10 ** e:.1f}\\times10^{{{e}}}$")
    hl = ok[ok.experiment == "headline"]
    for (target, rep), g in hl.groupby(["target", "repr"]):
        tag = f"{target}{rep}".replace("adjclose", "AC").replace("ohlc", "OHLC")
        m = g.groupby("arch").da.mean()
        fact(f"DA{tag}Min", pct(m.min()))
        fact(f"DA{tag}Max", pct(m.max()))
        fact(f"DA{tag}SeedMin", pct(g.da.min()))
        fact(f"DA{tag}SeedMax", pct(g.da.max()))
    n_test = int(hl.n_test.iloc[0])
    fact("DAbinomSE", f"{100 * np.sqrt(0.25 / n_test):.1f}")


def episode_facts(wf, out, target="ohlc"):
    """Anatomy of the walk-forward fold on which levels-trained CFC is worst (joint OHLC)."""
    lv = wf[(wf.target == target) & (wf["repr"] == "levels")]
    c = lv[(lv.arch == "cfc") & (lv.status == "ok")]
    fold = int(c.groupby("fold").r2_joint.median().idxmin())
    g = c[c.fold == fold]
    Z = [np.load(out / "predictions" / f"{r}.npz") for r in g.run_id]
    d = pd.to_datetime(Z[0]["dates"])
    err = np.median([((z["actual_scaled"] - z["pred_scaled"]) ** 2).mean(1) for z in Z], 0)
    rw = ((Z[0]["actual_scaled"] - Z[0]["rw_scaled"]) ** 2).mean(1)
    big = err > 100 * np.median(rw)
    fact("EPworstDay", f"{d[int(np.argmax(err))]:%-d %B %Y}")
    fact("EPbigDays", int(big.sum()))
    fact("EPbigFrom", f"{d[big].min():%-d %B}")
    fact("EPbigTo", f"{d[big].max():%-d %B %Y}")
    shares = [np.sort(((z["actual_scaled"] - z["pred_scaled"]) ** 2).mean(1))[::-1] for z in Z]
    shares = [v[:20].sum() / v.sum() for v in shares]
    fact("EPtopShareMin", f"{100 * min(shares):.1f}")
    fact("EPtopShareMax", f"{100 * max(shares):.1f}")
    fact("EPpredMin", f"{min(z['pred_scaled'].min() for z in Z):.1f}")
    fact("EPpredMax", f"{max(z['pred_scaled'].max() for z in Z):.1f}")
    a = Z[0]["actual_scaled"]
    fact("EPactMin", f"{a.min():.2f}")
    fact("EPactMax", f"{a.max():.2f}")
    fact("EPshareAbove", f"{100 * (a[:, -1] > 1).mean():.0f}")
    ll = c.groupby("fold").last_loss.median()
    fact("EPlastLoss", sci(g.last_loss.median()))
    fact("EPlastLossFoldMin", sci(ll.min()))
    fact("EPlastLossFoldMax", sci(ll.max()))
    r = g.iloc[0]
    fact("EPtrain", f"{pd.Timestamp(r.train_start):%B %Y}--{pd.Timestamp(r.train_end):%B %Y}")
    med = lv[(lv.fold == fold) & (lv.status == "ok")].groupby("arch").r2_joint.median()
    for a_ in ["gru", "lstm", "ltc"]:
        fact(f"EPRtwo{a_}", num(med[a_]))
    # extrapolation in every fold (share of test closes above the training maximum)
    sh = {}
    for k in sorted(lv.fold.unique()):
        rid = lv[(lv.fold == k) & (lv.arch == "gru")].run_id.iloc[0]
        x = np.load(out / "predictions" / f"{rid}.npz")["actual_scaled"][:, -1]
        sh[k] = (100 * (x > 1).mean(), x.max())
    kmax = max(sh, key=lambda k: sh[k][1])
    fact("EPmaxExFold", kmax)
    fact("EPmaxExShare", f"{sh[kmax][0]:.0f}")
    fact("EPmaxExMax", f"{sh[kmax][1]:.2f}")
    fact("EPmaxExCfcRtwo", num(c[c.fold == kmax].r2_joint.median()))


def audit_facts(fo, wf, ha, sk, lat, out):
    """Facts that qualify the main statements precisely (seed concentration of divergence, spreads,
    associations), so that the text never states more than the logs show."""
    import json
    from scipy.stats import spearmanr, mannwhitneyu
    # --- divergence: by seed and width, across all experiments
    allr = pd.concat([fo.assign(src="fo"), wf.assign(src="wf"), ha.assign(src="ha")], ignore_index=True)
    c = allr[allr.arch == "cfc"]
    fact("DIVallShare", f"{100 * (c.status != 'ok').mean():.1f}")
    fact("DIVallRuns", int((c.status != "ok").sum()))
    fact("DIVallCfcRuns", f"{len(c):,}".replace(",", "{,}"))
    for units, word in [(20, "Twenty"), (80, "Eighty")]:
        g = c[c.units == units]
        by = g.groupby("seed").apply(lambda x: ((x.status != "ok").sum(), len(x)))
        worst = max(by.items(), key=lambda kv: kv[1][0])
        fact(f"DIVseed{word}", int(worst[0]))
        fact(f"DIVseed{word}Div", int(worst[1][0]))
        fact(f"DIVseed{word}N", int(worst[1][1]))
        rest = g[g.seed != worst[0]]
        fact(f"DIVrest{word}Div", int((rest.status != "ok").sum()))
        fact(f"DIVrest{word}N", f"{len(rest):,}".replace(",", "{,}"))
    first = []
    for r in c.itertuples():
        h = out / "histories" / f"{r.run_id}.json"
        if h.exists():
            v = json.loads(h.read_text())["loss"][0]
            first.append((r.src, r.status != "ok", float("nan") if v is None else float(v)))
    fd = np.array([v for _, d, v in first if d]); fo_ = np.array([v for _, d, v in first if not d])
    fd_f = fd[np.isfinite(fd)]
    fact("DIVfirstLossMedianAll", num(np.median(fd_f)))
    fact("DIVfirstLossNaNAll", int((~np.isfinite(fd)).sum()))
    fact("OKfirstLossMedianAll", num(np.median(fo_[np.isfinite(fo_)])))
    ffo = [(d, v) for src, d, v in first if src == "fo"]
    a_ = np.array([v for d, v in ffo if d]); b_ = np.array([v for d, v in ffo if not d])
    pv = mannwhitneyu(a_[np.isfinite(a_)], b_[np.isfinite(b_)]).pvalue
    fact("DIVfoFirstMWp", "$<$0.001" if pv < 0.001 else f"{pv:.3f}")
    fact("DIVfirstLossMWp", "$<$0.001" if mannwhitneyu(fd_f, fo_[np.isfinite(fo_)]).pvalue < 0.001 else
         f"{mannwhitneyu(fd_f, fo_[np.isfinite(fo_)]).pvalue:.3f}")
    # --- spread among returns-trained architectures, per target, over the ten index-target cells
    ok = fo[fo.status == "ok"]
    for target in ["adjclose", "ohlc"]:
        spreads = []
        for (exp, index), g in ok[(ok.target == target) & (ok["repr"] == "returns") &
                                  ok.experiment.isin(["headline", "cross_market"])].groupby(["experiment", "index"]):
            m = g.groupby("arch").mase.median()
            spreads.append((100 * (m.max() / m.min() - 1), index))
        mx = max(spreads)
        tag = "AC" if target == "adjclose" else "OHLC"
        fact(f"SPREADret{tag}Max", f"{mx[0]:.0f}")
        fact(f"SPREADret{tag}MaxIndex", {"SPX": "S\\&P 500", "N225": "Nikkei 225", "SSE": "SSE Composite",
                                         "NIFTY50": "NIFTY 50", "NIFTY100": "NIFTY 100"}[mx[1]])
        fact(f"SPREADret{tag}Min", f"{min(spreads)[0]:.0f}")
    # --- returns-trained OHLC vs the close-anchored naive, per index (relative MAE, median over seeds)
    if sk is not None:
        s2 = sk.merge(fo[["run_id", "experiment", "index"]], on="run_id")
        s2 = s2[(s2.target == "ohlc") & (s2["repr"] == "returns") & s2.experiment.isin(["headline", "cross_market"])]
        for index, g in s2.groupby("index"):
            m = g.groupby("arch").rel_naive.median()
            fact(f"NVgain{index}Min", f"{100 * (1 - m.max()):.1f}")
            fact(f"NVgain{index}Max", f"{100 * (1 - m.min()):.1f}")
    # --- walk-forward: association between extrapolation share and levels MASE across folds
    rows = []
    for k in sorted(wf.fold.unique()):
        rid = wf[(wf.fold == k) & (wf.arch == "gru") & (wf.target == "ohlc") & (wf["repr"] == "levels")].run_id.iloc[0]
        x = np.load(out / "predictions" / f"{rid}.npz")["actual_scaled"][:, -1]
        rows.append((k, (x > 1).mean(), x.max()))
    sh = pd.DataFrame(rows, columns=["fold", "share", "mx"]).set_index("fold")
    rhos = []
    for target in ["adjclose", "ohlc"]:
        for arch in ARCHS:
            m = wf[(wf.target == target) & (wf["repr"] == "levels") & (wf.arch == arch) & (wf.status == "ok")
                   ].groupby("fold").mase.median()
            rhos.append(spearmanr(sh.share.reindex(m.index), m.values).correlation)
    fact("RHOshareMin", f"{min(rhos):.2f}")
    fact("RHOshareMax", f"{max(rhos):.2f}")
    folds = wf.drop_duplicates("fold").set_index("fold").sort_index()
    side_hi = [k for k in folds.index if folds.loc[k, "regime"] == "sideways" and sh.loc[k, "share"] > 0.4]
    bull_lo = [k for k in folds.index if folds.loc[k, "regime"] == "bull" and sh.loc[k, "share"] < 0.05]
    fact("FOLDsideHigh", ", ".join(str(k) for k in side_hi))
    fact("FOLDbullLow", ", ".join(str(k) for k in bull_lo))
    # fold table (dates, regime, share above the training maximum)
    trows = []
    for k in folds.index:
        r = folds.loc[k]
        trows.append({"fold": int(k), "train": f"{pd.Timestamp(r.train_start):%b %Y}--{pd.Timestamp(r.train_end):%b %Y}",
                      "test": f"{pd.Timestamp(r.test_start):%d %b %Y}--{pd.Timestamp(r.test_end):%d %b %Y}",
                      "regime": r.regime, "share": f"{100 * sh.loc[k, 'share']:.0f}", "mx": f"{sh.loc[k, 'mx']:.2f}"})
    write(pd.DataFrame(trows), "table_wf_folds", REPO / "results" / "tables", panel_col="__none__")
    # --- adjusted close, walk-forward, levels CFC: folds with a median MASE above 20
    m = wf[(wf.target == "adjclose") & (wf["repr"] == "levels") & (wf.arch == "cfc") & (wf.status == "ok")
           ].groupby("fold").mase.median()
    fact("WFacCfcBadFolds", ", ".join(str(k) for k in m[m > 20].index))
    # --- LTC returns adjusted-close folds below one (not significant)
    m = wf[(wf.target == "adjclose") & (wf["repr"] == "returns") & (wf.arch == "ltc") & (wf.status == "ok")
           ].groupby("fold").mase.median()
    fact("WFacLtcBelowOne", int((m < 1).sum()))
    # --- loss convergence on returns: epoch at which the median loss is within 10% of its final value
    hl = fo[(fo.experiment == "headline") & (fo.target == "ohlc") & (fo["repr"] == "returns") & (fo.status == "ok")]
    eps = []
    for arch in ["cfc", "ltc", "gru"]:
        C = []
        for rid in hl[hl.arch == arch].run_id:
            h = out / "histories" / f"{rid}.json"
            if h.exists():
                C.append(np.array(json.loads(h.read_text())["loss"], float))
        L = np.median(np.array(C), axis=0)
        eps.append(int(np.argmax(L <= 1.1 * L[-1])) + 1)
    fact("LOSSretWithinEpoch", max(eps))
    # --- split sweep, returns: range of MASE across splits
    sw = pd.read_csv(out / "fixed_origin.csv")
    sw = sw[(sw.experiment == "split_sweep") & (sw.status == "ok") & (sw["repr"] == "returns")]
    for (target, index), g in sw.groupby(["target", "index"]):
        m = g.groupby("test_frac").mase.median()
        tag = f"SPLITret{target}{index}".replace("adjclose", "AC").replace("ohlc", "OHLC")
        fact(tag + "Min", num(m.min()))
        fact(tag + "Max", num(m.max()))
    # --- latency ratio range: CFC against the gated cells at the same width, all configurations
    if lat is not None:
        rr = []
        for (target, units), g in lat.groupby(["target", "units"]):
            g = g.set_index("arch").ms_per_session
            rr += [g["cfc"] / g["lstm"], g["cfc"] / g["gru"]]
        fact("LATratioMin", f"{min(rr):.1f}")
        fact("LATratioMax", f"{max(rr):.1f}")
        if "platform" in lat.columns:
            fact("LATplatform", str(lat.platform.iloc[0]))
            fact("LATtf", str(lat.tf_version.iloc[0]))
    # --- episode: median-seed R2 and the implied SSE/SST ratio
    lv = wf[(wf.target == "ohlc") & (wf["repr"] == "levels") & (wf.arch == "cfc") & (wf.status == "ok")]
    k = int(lv.groupby("fold").r2_joint.median().idxmin())
    med = float(lv[lv.fold == k].r2_joint.median())
    fact("EPmedRtwo", f"{med:.0f}")
    fact("EPratio", f"{1 - med:.0f}")
    pre = []
    for arch in ARCHS:
        g = wf[(wf.target == "ohlc") & (wf["repr"] == "levels") & (wf.arch == arch) & (wf.fold == k) & (wf.status == "ok")]
        vals = []
        for rid in g.run_id:
            z = np.load(out / "predictions" / f"{rid}.npz")
            d = pd.to_datetime(z["dates"])
            sel = d < pd.Timestamp("2023-01-01")
            vals.append(np.abs(z["pred_scaled"][sel] - z["actual_scaled"][sel]).mean() /
                        np.abs(z["rw_scaled"][sel] - z["actual_scaled"][sel]).mean())
        fact(f"EPpre{arch}", f"{min(vals):.1f}--{max(vals):.1f}")


def pairwise_facts(pw):
    """Pairwise DM p-values between architectures on the NIFTY 100 headline window (seed-averaged losses)."""
    if pw is None:
        return
    h = pw[pw.experiment == "headline"]
    for r in h.itertuples():
        a, b = sorted([r.model_a, r.model_b])
        tag = f"PW{r.target}{a}{b}".replace("adjclose", "AC").replace("ohlc", "OHLC").replace("_", "")
        p = r.p_value
        fact(tag + "P", "$<$0.001" if p < 0.001 else f"{p:.3f}")
        better = r.model_a if r.dm_stat < 0 else r.model_b
        fact(tag + "Better", better.split("_")[0].upper())


def write_facts(tdir):
    lines = ["%% generated by scripts/07_make_tables.py -- do not edit"]
    for k, v in sorted(FACTS.items()):
        words = "Zero One Two Three Four Five Six Seven Eight Nine".split()
        name = "".join(words[int(ch)] if ch.isdigit() else ch for ch in k if ch.isalnum())
        lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    (tdir / "results_facts.tex").write_text("\n".join(lines) + "\n")
    if PAPER_DIR:
        (Path(PAPER_DIR) / "results_facts.tex").write_text((tdir / "results_facts.tex").read_text())
    print(f"results_facts.tex: {len(FACTS)} macros")


def main():
    global PAPER_DIR
    ap = cli.base_parser(__doc__)
    ap.add_argument("--paper-dir", default=None, help="also write the .tex row files into this folder")
    args = ap.parse_args()
    PAPER_DIR = args.paper_dir
    out = runs_dir(args.out)
    tdir = REPO / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    rd = lambda n: pd.read_csv(out / n) if (out / n).exists() else None
    fo, bl, wf, ha = rd("fixed_origin.csv"), rd("baselines.csv"), rd("walk_forward.csv"), rd("history_ablation.csv")
    dmr, sk, rr = rd("dm_vs_rw.csv"), rd("skill.csv"), rd("primary_window.csv")
    lat = rd("latency_cpu.csv")             # the serial CPU benchmark reported in the paper
    if lat is None:
        lat = rd("latency_gpu.csv")
    if fo is not None:
        # the NIFTY 100 95-5 cell of the split sweep re-runs the headline CFC cell (bit-identical); drop only it
        fo = fo[~((fo.experiment == "split_sweep") & (fo.test_frac == 0.05) & (fo["index"] == "NIFTY100"))].copy()
        t_test(fo, bl, tdir)
        t_capacity(fo, tdir)
        t_params_latency(fo, lat, tdir)
        sw = pd.read_csv(out / "fixed_origin.csv")
        t_split(sw, bl, tdir)
        t_cross(fo, bl, tdir)
        t_skill(sk, fo, wf, tdir)
        t_robustness(rr, fo, tdir)
        t_primary(rr, tdir)
    if wf is not None:
        t_wf(wf, tdir)
        episode_facts(wf, out)
    if ha is not None:
        t_history(ha, tdir)
    if fo is not None and wf is not None:
        t_dm(dmr, sk, fo, wf, tdir)
        t_repr(fo, wf, tdir)
        data_facts(fo, bl, wf)
        extra_facts(pd.read_csv(out / "fixed_origin.csv"), out, fo)
    pairwise_facts(rd("dm_pairwise.csv"))
    if fo is not None and wf is not None and ha is not None:
        audit_facts(fo, wf, ha, sk, lat, out)
    write_facts(tdir)


if __name__ == "__main__":
    main()

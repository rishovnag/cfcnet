#!/usr/bin/env python
"""Regenerate every results figure of the paper from the run logs, in the journal style of
src/cfcnet/style.py (final printed size, TeX Gyre Heros, colour-blind-safe palette with markers).

Accuracy is shown as MASE (mean absolute scaled error against the random walk; < 1 beats it) on a log
axis, with the random walk as the reference line, because R^2 on persistent price levels is high even
for persistence; R^2 is shown where it is the subject (headline and robustness figures).

    python scripts/08_make_figures.py                           # results/figures/
    python scripts/08_make_figures.py --paper-dir ../research-paper

    fig_primary.pdf            headline window (NIFTY 100 primary sample): forecasts and cumulative MAE vs random walk
    fig_seed_variability.pdf   per-seed MASE, extended NIFTY 100 95-5, targets x representations
    fig_actual_predicted.pdf   extended NIFTY 100 test window: close forecasts (median seed) vs actual
    fig_skill.pdf              per-column MASE (Open, High, Low, Close) of the returns-trained OHLC models
    fig_pareto.pdf             trainable parameters vs median MASE, both widths
    fig_extrapolation.pdf      test levels beyond the training range and the resulting levels-model errors
    fig_cross_market.pdf       median MASE per index and architecture
    fig_ltc_loss.pdf           training-loss curves, LTC vs CFC vs GRU, levels vs returns
    fig_wf_timeline.pdf        walk-forward per-fold MASE with regimes
    fig_wf_history.pdf         history-length ablation, CFC vs GRU
    fig_wf_fold_anatomy.pdf    the walk-forward fold of the CFC regime-shift episode
    fig_robustness.pdf         robustness of the headline to protocol choices (runs/primary_window.csv)
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from cfcnet import cli, style as S  # noqa: E402
from cfcnet.io import REPO, runs_dir  # noqa: E402

ARCHS = ["cfc", "ltc", "gru", "lstm"]          # continuous-time first, then gated
TARGETS = ["adjclose", "ohlc"]
REPRS = ["levels", "returns"]
OHLC_NAMES = ["Open", "High", "Low", "Close"]
LETTERS = "abcdefgh"

RW_H = (Line2D([], [], **{k: S.BENCH["rw"][k] for k in ("color", "ls", "lw")}), "Random walk")
NAIVE_H = (Line2D([], [], **{k: S.BENCH["naive"][k] for k in ("color", "ls", "lw")}), "Close-anchored naive")
ACT_H = (Line2D([], [], color=S.INK, lw=1.0), "Realised")


def arch_handles(archs=ARCHS, line=True):
    return [(S.arch_handle(a, line=line), S.ARCH_LABEL[a]) for a in archs]


def legend(fig, items, ncol=None, y=1.0):
    S.top_legend(fig, [h for h, _ in items], [l for _, l in items], ncol=ncol, y=y)


# ------------------------------------------------------------------ axes helpers
NICE = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500]


def _nice_ticks(lo, hi):
    t = [v for v in NICE if lo <= v <= hi * 1.0001]
    if 2 < hi / max(lo, 1e-9) <= 4:
        t = [v for v in t if v in (0.5, 0.6, 0.8, 1, 1.5, 2, 3, 5)]
    elif hi / max(lo, 1e-9) > 4:
        t = [v for v in t if v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)]
    elif len(t) > 6:
        t = [v for v in t if v in (0.6, 0.8, 1, 1.2, 1.5, 2, 3, 5)]
    return t


def mase_axis(ax, label=True, rw=True):
    ax.set_yscale("log")
    if rw:
        ax.axhline(1, **{k: S.BENCH["rw"][k] for k in ("color", "ls", "lw")}, zorder=1)
    ax._mase = True
    if label:
        ax.set_ylabel("MASE (log scale)")


def finish_log_axes(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        if getattr(ax, "_mase", False):
            lo, hi = ax.get_ylim()
            ax.yaxis.set_major_locator(mticker.FixedLocator(_nice_ticks(lo, hi)))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        if ax.get_xscale() == "log":
            lo, hi = ax.get_xlim()
            t = [v for v in [100, 300, 1000, 3000, 10000, 30000] if lo <= v <= hi]
            ax.xaxis.set_major_locator(mticker.FixedLocator(t))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.xaxis.set_minor_locator(mticker.NullLocator())


def save(fig, path):
    finish_log_axes(fig)
    S.save(fig, path)


def date_axis(ax, maxticks=6):
    loc = matplotlib.dates.AutoDateLocator(minticks=3, maxticks=maxticks)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(loc, show_offset=False))


def thousands(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def no_xgrid_ticks(ax):
    ax.tick_params(axis="x", length=0)


def grid4(h, sharey="row", sharex=False):
    fig, axs = plt.subplots(2, 2, figsize=(S.FULL_WIDTH, h), sharey=sharey, sharex=sharex, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.05, hspace=0.08, wspace=0.05)
    return fig, axs


def label4(axs):
    k = 0
    for i, target in enumerate(TARGETS):
        for j, rep in enumerate(REPRS):
            S.panel(axs[i, j], LETTERS[k], f"{S.TARGET_LABEL[target]}, trained on {S.REPR_LABEL[rep]}")
            k += 1


class Runs:
    def __init__(self, out: Path):
        self.out = out
        rd = lambda n: pd.read_csv(out / n) if (out / n).exists() else None
        self.fo, self.wf, self.ha = rd("fixed_origin.csv"), rd("walk_forward.csv"), rd("history_ablation.csv")
        self.folds, self.bl, self.sk = rd("walk_forward_folds.csv"), rd("baselines.csv"), rd("skill.csv")
        self.rr = rd("primary_window.csv")

    def npz(self, run_id):
        return np.load(self.out / "predictions" / f"{run_id}.npz")

    def hist(self, run_id):
        p = self.out / "histories" / f"{run_id}.json"
        return np.array(json.loads(p.read_text())["loss"], dtype=float) if p.exists() else None

    def naive_mase(self, index, target, tf_=0.05):
        if self.bl is None or target != "ohlc":
            return None
        b = self.bl[(self.bl["index"] == index) & (self.bl.target == target) & (self.bl.test_frac == tf_) &
                    (self.bl.model == "naive_close")]
        return float(b.mase.iloc[0]) if len(b) else None


def median_run(g):
    """Run id of the seed whose MASE is the median (finished runs)."""
    g = g[g.status == "ok"].sort_values("mase")
    return None if g.empty else g.run_id.iloc[(len(g) - 1) // 2]


# ------------------------------------------------------------------ headline (primary sample)
def fig_primary(R, path):
    rr = R.rr
    if rr is None:
        return False
    g0 = rr[(rr.timing == "next_day") & (rr.scaler_fit == "train") & (rr["data"] == "primary") & (rr.target == "ohlc")]
    fig, axs = plt.subplots(2, 1, figsize=(S.FULL_WIDTH, 3.9), sharex=True, layout="constrained",
                            gridspec_kw={"height_ratios": [1.45, 1]})
    fig.get_layout_engine().set(h_pad=0.05, hspace=0.06)
    z0 = None
    for arch in ["lstm", "gru", "cfc"]:                  # CFC drawn last, on top
        rid = median_run(g0[g0.arch == arch])
        z = R.npz(rid)
        z0 = z
        d = pd.to_datetime(z["dates"])
        axs[0].plot(d, z["pred"][:, -1], color=S.ARCH[arch], lw=0.85 if arch != "cfc" else 1.05,
                    zorder=3 if arch == "cfc" else 2)
        cum = np.cumsum(np.abs(z["actual_scaled"] - z["pred_scaled"]).mean(axis=1))
        cum_rw = np.cumsum(np.abs(z["actual_scaled"] - z["rw_scaled"]).mean(axis=1))
        axs[1].plot(d, cum / cum_rw, color=S.ARCH[arch], lw=1.0 if arch != "cfc" else 1.2, zorder=3 if arch == "cfc" else 2)
        S.direct_label(axs[1], d[-1], cum[-1] / cum_rw[-1], S.ARCH_LABEL[arch], color=S.ARCH[arch], dx=3, size=7)
    d = pd.to_datetime(z0["dates"])
    axs[0].plot(d, z0["actual"][:, -1], color=S.INK, lw=1.0, zorder=4)
    thousands(axs[0])
    axs[0].set_ylabel("NIFTY 100 close")
    axs[1].axhline(1, **{k: S.BENCH["rw"][k] for k in ("color", "ls", "lw")})
    axs[1].set_ylim(0, None)
    axs[1].set_ylabel("cumulative MAE /\nrandom walk")
    date_axis(axs[1], maxticks=8)
    S.panel(axs[0], "a", "Close forecasts of the median seed, joint OHLC target")
    S.panel(axs[1], "b", "Accuracy relative to the random walk, accumulated over the test window")
    legend(fig, [ACT_H] + arch_handles(["cfc", "gru", "lstm"]) + [RW_H], ncol=5)
    save(fig, path)
    return True


# ------------------------------------------------------------------ extended fixed origin
def fig_seed_variability(R, path):
    fo = R.fo[(R.fo.experiment == "headline")]
    fig, axs = grid4(4.1)
    rng = np.random.default_rng(0)
    for i, target in enumerate(TARGETS):
        for j, rep in enumerate(REPRS):
            ax = axs[i, j]
            for x, arch in enumerate(ARCHS):
                g = fo[(fo.target == target) & (fo["repr"] == rep) & (fo.arch == arch)]
                v = g[g.status == "ok"].mase.to_numpy()
                ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=7, color=S.ARCH[arch], marker=S.MARKER[arch],
                           edgecolor="white", linewidth=0.25, alpha=0.85, zorder=3)
                ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=S.INK, lw=1.1, zorder=4,
                        solid_capstyle="butt")
                nd = int((g.status != "ok").sum())
                if nd:
                    ax.annotate(f"{nd} div.", (x, 1), xycoords=("data", "axes fraction"), xytext=(0, -1),
                                textcoords="offset points", ha="center", va="top", fontsize=6.5, color=S.INK3)
            mase_axis(ax, label=j == 0)
            nv = R.naive_mase("NIFTY100", target)
            if nv:
                ax.axhline(nv, **{k: S.BENCH["naive"][k] for k in ("color", "ls", "lw")}, zorder=1)
            ax.set_xticks(range(4), [S.ARCH_LABEL[a] for a in ARCHS])
            no_xgrid_ticks(ax)
            ax.set_xlim(-0.6, 3.6)
    label4(axs)
    legend(fig, [(Line2D([], [], color=S.INK, lw=1.1, solid_capstyle="butt"), "Median over 30 seeds"), RW_H,
                 (NAIVE_H[0], "Close-anchored naive (OHLC)")], ncol=3)
    save(fig, path)
    return True


def fig_actual_predicted(R, path, index="NIFTY100"):
    fo = R.fo[(R.fo.experiment == "headline") & (R.fo["index"] == index)]
    fig, axs = grid4(3.9, sharey=False, sharex=True)
    for i, target in enumerate(TARGETS):
        for j, rep in enumerate(REPRS):
            ax = axs[i, j]
            z0 = None
            for arch in ["ltc", "lstm", "gru", "cfc"]:
                rid = median_run(fo[(fo.target == target) & (fo["repr"] == rep) & (fo.arch == arch)])
                if rid is None:
                    continue
                z = R.npz(rid)
                z0 = z
                ax.plot(pd.to_datetime(z["dates"]), z["pred"][:, -1], color=S.ARCH[arch], lw=0.75)
            ax.plot(pd.to_datetime(z0["dates"]), z0["actual"][:, -1], color=S.INK, lw=1.0, zorder=5)
            lo, hi = z0["actual"][:, -1].min(), z0["actual"][:, -1].max()
            pad = 0.35 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            date_axis(ax, maxticks=5)
            thousands(ax)
            if j == 0:
                ax.set_ylabel("close (index points)")
    label4(axs)
    legend(fig, [ACT_H] + arch_handles(line=True), ncol=5)
    save(fig, path)
    return True


def fig_skill(R, path):
    if R.sk is None:
        return False
    s = R.sk[(R.sk.target == "ohlc")]
    hl = s.merge(R.fo[["run_id", "experiment"]], on="run_id")
    panels = [("Fixed origin, extended NIFTY 100 (30 seeds)", hl[(hl.experiment == "headline") & (hl["repr"] == "returns")]),
              ("Walk-forward, 15 folds × 3 seeds", s[(s.source == "walk_forward") & (s["repr"] == "returns")])]
    nv = s[(s.source == "baseline") & s.run_id.str.contains("_NIFTY100_ohlc_tf5_naive_close")]
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL_WIDTH, 2.35), sharey=True, layout="constrained")
    cols = ["open", "high", "low", "close"]
    off = 0.14
    for k, (ax, (title, g)) in enumerate(zip(axs, panels)):
        for c_i in range(4):
            ax.axvspan(c_i - 0.42, c_i + 0.42, color="#f3f3f3", lw=0, zorder=0)
        for a_i, arch in enumerate(ARCHS):
            v = [g[g.arch == arch][f"mase_{c}"].median() for c in cols]
            x = np.arange(4) + (a_i - 1.5) * off
            ax.plot(x, v, ls="none", **S.arch_kw(arch, ms=S.MSIZE[arch] + 0.6), zorder=3)
        if k == 0 and len(nv):
            for c_i, c in enumerate(cols):
                vv = float(nv.iloc[0][f"mase_{c}"])
                ax.plot([c_i - 0.38, c_i + 0.38], [vv, vv], **{kk: S.BENCH["naive"][kk] for kk in ("color", "ls")},
                        lw=1.2, zorder=2)
        ax.axhline(1, **{kk: S.BENCH["rw"][kk] for kk in ("color", "ls", "lw")}, zorder=1)
        ax.set_xticks(range(4), OHLC_NAMES)
        no_xgrid_ticks(ax)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(0.4, 1.12)
        S.panel(ax, LETTERS[k], title)
    axs[0].set_ylabel("median MASE of the column")
    legend(fig, arch_handles(line=False) + [RW_H, (NAIVE_H[0], "Close-anchored naive")], ncol=6)
    S.save(fig, path)
    return True


def fig_pareto(R, path):
    fo = R.fo[R.fo.experiment.isin(["headline", "capacity"])]
    fig, axs = grid4(3.9, sharey=False)
    for i, target in enumerate(TARGETS):
        for j, rep in enumerate(REPRS):
            ax = axs[i, j]
            for arch in ARCHS:
                g = fo[(fo.target == target) & (fo["repr"] == rep) & (fo.arch == arch) & (fo.status == "ok")]
                s = g.groupby("n_params").mase.agg(med="median", lo=lambda v: v.quantile(0.25),
                                                   hi=lambda v: v.quantile(0.75)).reset_index()
                ax.vlines(s.n_params, s.lo, s.hi, color=S.ARCH[arch], lw=0.7, alpha=0.8)
                ax.plot(s.n_params, s.med, **S.arch_kw(arch, lw=0.9, ms=S.MSIZE[arch] + 0.6))
            ax.set_xscale("log")
            mase_axis(ax, label=j == 0)
            if i == 1:
                ax.set_xlabel("trainable parameters (log scale)")
    label4(axs)
    legend(fig, arch_handles() + [RW_H], ncol=5)
    save(fig, path)
    return True


def fig_extrapolation(R, path):
    """Levels-trained models out of sample: the test window leaves the training range."""
    fo = R.fo
    sw = fo[(fo.experiment == "split_sweep") & (fo["index"] == "NIFTY100") & (fo.target == "ohlc")]
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL_WIDTH, 2.45), layout="constrained",
                            gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axs[0]
    g = sw[sw.test_frac == 0.20]
    for rep, ls in [("levels", "-"), ("returns", (0, (3, 1.5)))]:
        z = R.npz(median_run(g[g["repr"] == rep]))
        d = pd.to_datetime(z["dates"])
        ax.plot(d, z["pred_scaled"][:, -1], color=S.ARCH["cfc"], lw=0.9, ls=ls)
    ax.plot(d, z["actual_scaled"][:, -1], color=S.INK, lw=1.0)
    ax.axhline(1.0, color=S.RULE, lw=0.7, zorder=0)
    S.direct_label(ax, d[-1], 1.0, "training maximum", dy=-3, dx=0, ha="right", va="top", size=6.5, color=S.INK3)
    ax.set_ylim(-0.2, 2.2)
    date_axis(ax, maxticks=5)
    ax.set_ylabel("close, training-scaled")
    S.panel(ax, "a", "NIFTY 100, 80–20 split: CFC close forecasts")
    ax.legend([Line2D([], [], color=S.INK, lw=1.0), Line2D([], [], color=S.ARCH["cfc"], lw=0.9),
               Line2D([], [], color=S.ARCH["cfc"], lw=0.9, ls=(0, (3, 1.5)))],
              ["Realised", "CFC, price levels", "CFC, log returns"], loc="lower right", fontsize=7)
    ax = axs[1]
    for index, mk in [("NIFTY100", "o"), ("NIFTY500", "s")]:
        for tf_ in sorted(fo.test_frac.unique()):
            g = fo[(fo.experiment == "split_sweep") & (fo["index"] == index) & (fo.target == "ohlc") &
                   (fo.test_frac == tf_)]
            if g.empty:
                continue
            z = R.npz(g.run_id.iloc[0])
            share = float((z["actual_scaled"][:, -1] > 1.0).mean())
            for rep, fill in [("levels", S.ARCH["cfc"]), ("returns", "white")]:
                m = g[(g["repr"] == rep) & (g.status == "ok")].mase.median()
                ax.plot(share, m, marker=mk, color=S.ARCH["cfc"], mfc=fill, mec=S.ARCH["cfc"], mew=0.8, ms=4.2,
                        ls="none")
    mase_axis(ax)
    ax.set_xlabel("test sessions above the training maximum")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    S.panel(ax, "b", "CFC, all splits, joint OHLC")
    ax.legend([Line2D([], [], marker="o", color=S.ARCH["cfc"], mec=S.ARCH["cfc"], ls="none", ms=4.2),
               Line2D([], [], marker="s", color=S.ARCH["cfc"], mec=S.ARCH["cfc"], ls="none", ms=4.2),
               Line2D([], [], marker="o", mfc="white", mec=S.ARCH["cfc"], mew=0.8, ls="none", ms=4.2)],
              ["NIFTY 100, levels", "NIFTY 500, levels", "Open: log returns"], loc="upper left", fontsize=6.8,
              handletextpad=0.2)
    save(fig, path)
    return True


def fig_cross_market(R, path):
    cm = R.fo[(R.fo.experiment == "cross_market") & (R.fo.status == "ok")]
    idx = ["NIFTY50", "SSE", "N225", "SPX"]          # ordered by the share of the test window above the training range
    fig, axs = grid4(3.9, sharey=False)
    for i, target in enumerate(TARGETS):
        for j, rep in enumerate(REPRS):
            ax = axs[i, j]
            for c_i in range(4):
                ax.axvspan(c_i - 0.42, c_i + 0.42, color="#f3f3f3", lw=0, zorder=0)
            for a_i, arch in enumerate(ARCHS):
                g = cm[(cm.target == target) & (cm["repr"] == rep) & (cm.arch == arch)]
                s = g.groupby("index").mase.agg(med="median", lo=lambda v: v.quantile(0.25),
                                                hi=lambda v: v.quantile(0.75)).reindex(idx)
                x = np.arange(4) + (a_i - 1.5) * 0.15
                ax.vlines(x, s.lo, s.hi, color=S.ARCH[arch], lw=0.7, alpha=0.8, zorder=2)
                ax.plot(x, s.med, ls="none", **S.arch_kw(arch, ms=S.MSIZE[arch] + 0.4), zorder=3)
            for k, index in enumerate(idx):
                nv = R.naive_mase(index, target)
                if nv:
                    ax.plot([k - 0.38, k + 0.38], [nv, nv], **{kk: S.BENCH["naive"][kk] for kk in ("color", "ls", "lw")},
                            zorder=1)
            mase_axis(ax, label=j == 0)
            ax.set_xticks(range(4), ["NIFTY 50", "SSE", "Nikkei 225", "S&P 500"], fontsize=7)
            no_xgrid_ticks(ax)
            ax.set_xlim(-0.55, 3.55)
    label4(axs)
    legend(fig, arch_handles(line=False) + [RW_H, (NAIVE_H[0], "Close-anchored naive (OHLC)")], ncol=6)
    save(fig, path)
    return True


def fig_ltc_loss(R, path):
    fo = R.fo[(R.fo.experiment == "headline") & (R.fo.target == "ohlc")]
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL_WIDTH, 2.3), sharey=True, layout="constrained")
    for k, (ax, rep) in enumerate(zip(axs, REPRS)):
        for arch in ["gru", "cfc", "ltc"]:
            curves = [R.hist(r) for r in fo[(fo["repr"] == rep) & (fo.arch == arch)].run_id]
            curves = [c for c in curves if c is not None and np.isfinite(c).all()]
            if not curves:
                continue
            L = min(map(len, curves))
            C = np.array([c[:L] for c in curves])
            e = np.arange(1, L + 1)
            ax.fill_between(e, np.quantile(C, 0.1, axis=0), np.quantile(C, 0.9, axis=0), color=S.ARCH[arch],
                            alpha=0.14, lw=0)
            ax.plot(e, np.median(C, axis=0), color=S.ARCH[arch], lw=1.1)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_xlim(0, L)
        S.panel(ax, LETTERS[k], f"Joint OHLC, trained on {S.REPR_LABEL[rep]}")
    axs[0].set_ylabel("training loss (log scale)")
    legend(fig, [(Line2D([], [], color=S.ARCH[a], lw=1.1), S.ARCH_LABEL[a]) for a in ["cfc", "ltc", "gru"]] +
           [(Patch(color=S.INK2, alpha=0.14, lw=0), "10th to 90th percentile over 30 seeds (line: median)")], ncol=4)
    S.save(fig, path)
    return True


# ------------------------------------------------------------------ walk-forward / history
def _regime_strip(ax, folds):
    for _, f in folds.iterrows():
        if f.regime != "sideways":
            ax.axvspan(f.fold - 0.5, f.fold + 0.5, color=S.REGIME[f.regime], lw=0, zorder=0)


def fig_wf_timeline(R, path):
    wf = R.wf[R.wf.status == "ok"]
    fig, axs = grid4(3.9, sharey="col", sharex=True)
    for i, target in enumerate(TARGETS):
        for j, rep in enumerate(REPRS):
            ax = axs[i, j]
            _regime_strip(ax, R.folds)
            g = wf[(wf.target == target) & (wf["repr"] == rep)]
            for arch in ARCHS:
                s = g[g.arch == arch].groupby("fold").mase.median()
                ax.plot(s.index, s.values, **S.arch_kw(arch, lw=0.85, ms=S.MSIZE[arch] - 0.4))
            mase_axis(ax, label=j == 0)
            ax.set_xticks(R.folds.fold[::2], [f"{pd.Timestamp(t):%Y}" for t in R.folds.test_start[::2]])
            ax.set_xlim(R.folds.fold.min() - 0.5, R.folds.fold.max() + 0.5)
            if i == 1:
                ax.set_xlabel("start of the one-year test window")
    label4(axs)
    regimes = [r for r in ["bull", "bear"] if (R.folds.regime == r).any()]
    legend(fig, arch_handles() + [RW_H] + [(Patch(color=S.REGIME[r], ec=S.REGIME_EDGE[r], lw=0.4), f"{r.capitalize()} window")
                                           for r in regimes], ncol=6 + len(regimes))
    save(fig, path)
    return True


def fig_history(R, path):
    ha = R.ha[R.ha.status == "ok"]
    fig, axs = grid4(3.7, sharey=False, sharex=True)
    for i, target in enumerate(TARGETS):
        for j, rep in enumerate(REPRS):
            ax = axs[i, j]
            for arch in ["gru", "cfc"]:
                s = ha[(ha.target == target) & (ha["repr"] == rep) & (ha.arch == arch)].groupby("hist_years").mase
                m = s.median()
                ax.fill_between(m.index, s.min(), s.max(), color=S.ARCH[arch], alpha=0.13, lw=0)
                ax.plot(m.index, m.values, **S.arch_kw(arch, lw=0.9))
            mase_axis(ax, label=j == 0)
            if i == 1:
                ax.set_xlabel("training history (trading years)")
    label4(axs)
    legend(fig, arch_handles(["cfc", "gru"]) + [(Patch(color=S.INK2, alpha=0.13, lw=0), "Range over 3 seeds"), RW_H],
           ncol=4)
    save(fig, path)
    return True


def fig_fold_anatomy(R, path, target="ohlc"):
    wf = R.wf[(R.wf.target == target)]
    lv = wf[wf["repr"] == "levels"]
    c = lv[(lv.arch == "cfc") & (lv.status == "ok")]
    fold = int(c.groupby("fold").r2_joint.median().idxmin())
    fig, axs = plt.subplots(2, 1, figsize=(S.FULL_WIDTH, 3.9), sharex=True, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.05, hspace=0.07)
    d0, worst = None, 0
    for arch in ["lstm", "gru", "ltc", "cfc"]:
        g = lv[(lv.arch == arch) & (lv.fold == fold) & (lv.status == "ok")]
        Z = [R.npz(r) for r in g.run_id]
        d0 = pd.to_datetime(Z[0]["dates"])
        err = np.median([((z["actual_scaled"] - z["pred_scaled"]) ** 2).mean(axis=1) for z in Z], axis=0)
        axs[0].plot(d0, err, color=S.ARCH[arch], lw=0.8 if arch != "cfc" else 1.0, zorder=3 if arch == "cfc" else 2)
        if arch == "cfc":
            worst = int(np.argmax(err))
            for z in Z:
                axs[1].plot(d0, z["pred_scaled"][:, -1], color=S.ARCH[arch], lw=0.7, alpha=0.8)
    gr = wf[(wf["repr"] == "returns") & (wf.arch == "cfc") & (wf.fold == fold) & (wf.status == "ok")]
    for r in gr.run_id:
        axs[1].plot(d0, R.npz(r)["pred_scaled"][:, -1], color=S.ARCH["cfc"], lw=0.8, ls=(0, (3, 1.5)))
    z = R.npz(c[c.fold == fold].run_id.iloc[0])
    axs[0].plot(d0, ((z["actual_scaled"] - z["rw_scaled"]) ** 2).mean(axis=1),
                **{k: S.BENCH["rw"][k] for k in ("color", "ls")}, lw=0.8)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("daily squared error")
    axs[1].plot(d0, z["actual_scaled"][:, -1], color=S.INK, lw=1.1, zorder=5)
    axs[1].axhline(1.0, color=S.RULE, lw=0.7, zorder=0)
    S.direct_label(axs[1], d0[2], 1.0, "training maximum", dy=3, dx=0, ha="left", va="bottom", size=6.5, color=S.INK3)
    lo, hi = z["actual_scaled"][:, -1].min(), z["actual_scaled"][:, -1].max()
    axs[1].set_ylim(lo - 0.6 * (hi - lo), hi + 0.6 * (hi - lo))
    axs[1].set_ylabel("close, training-scaled")
    for ax in axs:
        ax.axvline(d0[worst], color=S.INK3, lw=0.6, ls=(0, (2, 2)), zorder=0)
    date_axis(axs[1], maxticks=8)
    f = R.folds.set_index("fold").loc[fold]
    S.panel(axs[0], "a", f"Fold {fold} (test {pd.Timestamp(f.test_start):%b %Y}–{pd.Timestamp(f.test_end):%b %Y}), "
                         "trained on levels: median daily error over seeds")
    S.panel(axs[1], "b", "CFC close forecasts, every seed")
    legend(fig, arch_handles(line=True) + [RW_H], ncol=5, y=1.0)
    axs[1].legend([Line2D([], [], color=S.INK, lw=1.1), Line2D([], [], color=S.ARCH["cfc"], lw=0.7),
                   Line2D([], [], color=S.ARCH["cfc"], lw=0.8, ls=(0, (3, 1.5)))],
                  ["Realised", "CFC, trained on price levels", "CFC, trained on log returns"], ncol=3,
                  loc="lower right", fontsize=7, facecolor="white", framealpha=1, frameon=True, edgecolor="none")
    save(fig, path)
    return True


# ------------------------------------------------------------------ robustness of the headline
def fig_robustness(R, path):
    """Robustness of the headline to the protocol: median joint R^2 (top) and MASE (bottom) of each
    architecture under the three protocols, connected across protocols (slope chart)."""
    rr = R.rr
    if rr is None:
        return False
    from sklearn.metrics import r2_score
    variants = [("same_day", "full", "(1) same session,\nfull-series scaler"),
                ("same_day", "train", "(2) same session,\ntrain-only scaler"),
                ("next_day", "train", "(3) next session,\ntrain-only scaler")]
    fig, axs = plt.subplots(2, 2, figsize=(S.FULL_WIDTH, 3.9), sharex=True, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.05, hspace=0.08, wspace=0.06)
    for j, target in enumerate(TARGETS):
        okt = rr[(rr.target == target) & (rr.status == "ok") & (rr["data"] == "primary")]
        for arch in ARCHS:
            r2v, mv = [], []
            for timing, scaler, _ in variants:
                g = okt[(okt.arch == arch) & (okt.timing == timing) & (okt.scaler_fit == scaler)]
                r2v.append(g.r2_joint.median())
                mv.append(g.mase.median())
            if arch != "ltc":
                axs[0, j].plot(range(3), r2v, **S.arch_kw(arch, lw=0.9, ms=S.MSIZE[arch] + 0.5))
            axs[1, j].plot(range(3), mv, **S.arch_kw(arch, lw=0.9, ms=S.MSIZE[arch] + 0.5))
        g = okt[(okt.timing == "next_day") & (okt.scaler_fit == "train")]
        z = R.npz(g.run_id.iloc[0])
        rw_r2 = r2_score(z["actual_scaled"].reshape(-1), z["rw_scaled"].reshape(-1))
        axs[0, j].axhline(rw_r2, **{k: S.BENCH["rw"][k] for k in ("color", "ls", "lw")})
        S.direct_label(axs[0, j], 2.15, rw_r2, f"random walk, {rw_r2:.3f}", dx=0, dy=-3, ha="right", va="top", size=6.5)
        axs[0, j].set_ylim(0.0 if target == "adjclose" else 0.5, 1.02)
        mase_axis(axs[1, j], label=j == 0)
        axs[1, j].set_xticks(range(3), [v[2] for v in variants], fontsize=6.8)
        axs[1, j].set_xlim(-0.25, 2.25)
        no_xgrid_ticks(axs[1, j])
        S.panel(axs[0, j], LETTERS[j], f"{S.TARGET_LABEL[target]}: median joint $R^2$ (LTC < 0)")
        S.panel(axs[1, j], LETTERS[j + 2], f"{S.TARGET_LABEL[target]}: median MASE")
    axs[0, 0].set_ylabel("$R^2$")
    legend(fig, arch_handles() + [RW_H], ncol=5)
    save(fig, path)
    return True


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--paper-dir", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    S.apply()
    R = Runs(runs_dir(args.out))
    fdir = REPO / "results" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    jobs = {"fig_primary.pdf": fig_primary, "fig_seed_variability.pdf": fig_seed_variability,
            "fig_actual_predicted.pdf": fig_actual_predicted,
            "fig_skill.pdf": fig_skill, "fig_pareto.pdf": fig_pareto, "fig_extrapolation.pdf": fig_extrapolation,
            "fig_cross_market.pdf": fig_cross_market, "fig_ltc_loss.pdf": fig_ltc_loss,
            "fig_wf_timeline.pdf": fig_wf_timeline, "fig_wf_history.pdf": fig_history,
            "fig_wf_fold_anatomy.pdf": fig_fold_anatomy, "fig_robustness.pdf": fig_robustness}
    for name, fn in jobs.items():
        if args.only and name[:-4] not in args.only:
            continue
        ok = fn(R, fdir / name)
        print(("wrote " if ok else "skipped (no runs) ") + name)
        if ok and args.paper_dir:
            shutil.copy(fdir / name, Path(args.paper_dir) / name)


if __name__ == "__main__":
    main()

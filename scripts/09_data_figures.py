#!/usr/bin/env python
"""Figures that depend only on the data and the architecture definitions (no training runs).

    python scripts/09_data_figures.py                          # results/figures/
    python scripts/09_data_figures.py --paper-dir ../research-paper

    fig_flow.pdf                 discrete residual updates vs a continuous-time trajectory
    fig_ncp_architecture.pdf     the four neuron classes of a neural circuit policy
    fig_wiring_cfc.pdf           the actual AutoNCP(28,4) and AutoNCP(80,4) wirings used for CFC/LTC
    fig_index_histories.pdf      adjusted close of the six indices (vendor history vs analysed sample)
    fig_indices_normalised.pdf   cross-market indices on their common period, start = 100
    fig_return_correlation.pdf   correlation of daily log returns on the common period
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from cfcnet import cli, style  # noqa: E402
from cfcnet.data import fixed_origin_window, load_index, make_pairs  # noqa: E402
from cfcnet.io import REPO, data_path  # noqa: E402

S = style


# ------------------------------------------------------------------ schematics
def fig_flow(path):
    f = lambda z, t: -0.55 * z + 0.9 * np.sin(1.6 * t + 0.8 * z)
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL_WIDTH, 2.5), sharey=True)
    T, Z = np.meshgrid(np.linspace(0, 5, 21), np.linspace(-2.2, 2.2, 15))
    starts = [-1.8, -0.9, 0.0, 0.9, 1.8]
    cols = [S.BLUE_RAMP[i] for i in (2, 3, 4, 5, 6)]
    # (a) residual network: updates only at integer depths
    ax = axs[0]
    for k in range(6):
        zz = np.linspace(-2.2, 2.2, 15)
        ax.quiver(np.full_like(zz, k), zz, np.ones_like(zz) * 0.35, f(zz, k) * 0.35, angles="xy", scale_units="xy",
                  scale=1, width=0.004, color=S.INK3, alpha=0.6)
    for z0, c in zip(starts, cols):
        z = [z0]
        for k in range(5):
            z.append(z[-1] + f(z[-1], k))
        ax.plot(range(6), z, color=c, marker="o", ms=4.5, lw=1.4)
    ax.set_xlabel("depth (layer index)")
    ax.set_ylabel("hidden state")
    S.panel_label(ax, "a", "Discrete updates  $x_{k+1} = x_k + f(x_k)$")
    # (b) continuous dynamics
    ax = axs[1]
    U, V = np.ones_like(T), f(Z, T)
    nrm = np.hypot(U, V)
    ax.quiver(T, Z, U / nrm * 0.18, V / nrm * 0.18, angles="xy", scale_units="xy", scale=1, width=0.003,
              color=S.INK3, alpha=0.6)
    ts = np.linspace(0, 5, 501)
    for z0, c in zip(starts, cols):
        z = np.empty_like(ts)
        z[0] = z0
        for i in range(1, len(ts)):
            h = ts[i] - ts[i - 1]
            k1 = f(z[i - 1], ts[i - 1]); k2 = f(z[i - 1] + h * k1 / 2, ts[i - 1] + h / 2)
            k3 = f(z[i - 1] + h * k2 / 2, ts[i - 1] + h / 2); k4 = f(z[i - 1] + h * k3, ts[i])
            z[i] = z[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        ax.plot(ts, z, color=c, lw=1.6)
        ax.plot(ts[[0, -1]], z[[0, -1]], ls="none", marker="o", ms=4.5, color=c)
    ax.set_xlabel("time $t$")
    S.panel_label(ax, "b", "Continuous dynamics  $dx/dt = f(x(t), t)$")
    for ax in axs:
        ax.grid(False)
        ax.set_ylim(-2.6, 2.6)
    fig.tight_layout(w_pad=1.5)
    S.save(fig, path)


def _layer_positions(n, x, height=1.0):
    ys = np.linspace(-height / 2, height / 2, n) if n > 1 else np.array([0.0])
    return np.column_stack([np.full(n, x), ys])


def fig_ncp_architecture(path):
    rng = np.random.default_rng(7)
    sizes = {"sensory": 7, "inter": 6, "command": 4, "motor": 4}
    xs = {"sensory": 0, "inter": 1, "command": 2, "motor": 3}
    pos = {k: _layer_positions(n, xs[k], 0.18 * (n - 1)) for k, n in sizes.items()}
    fig, ax = plt.subplots(figsize=(S.FULL_WIDTH, 2.6))
    ax.set_axis_off()
    order = ["sensory", "inter", "command", "motor"]
    for src, dst in zip(order[:-1], order[1:]):
        for i, p in enumerate(pos[src]):
            for j in rng.choice(sizes[dst], size=min(2, sizes[dst]), replace=False):
                q = pos[dst][j]
                ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.6,
                                             color=S.INK3, alpha=0.7, shrinkA=6, shrinkB=6))
    cp = pos["command"]
    for i in range(sizes["command"]):                      # recurrent command connections
        j = (i + 1) % sizes["command"]
        ax.add_patch(FancyArrowPatch(cp[i], cp[j], connectionstyle="arc3,rad=-0.9", arrowstyle="-|>",
                                     mutation_scale=6, lw=0.6, color=S.NEURON["command"], shrinkA=6, shrinkB=6))
    names = {"sensory": "Sensory\ninputs $u_t$ (7 channels)", "inter": "Inter\nneurons",
             "command": "Command\nneurons (recurrent)", "motor": "Motor\nneurons = outputs"}
    for k in order:
        ax.scatter(pos[k][:, 0], pos[k][:, 1], s=95, color=S.NEURON[k], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(xs[k], pos["sensory"][:, 1].min() - 0.22, names[k], ha="center", va="top", fontsize=7.5, color=S.INK2)
    ax.annotate("", xy=(3.35, 0.72), xytext=(-0.35, 0.72),
                arrowprops=dict(arrowstyle="-|>", color=S.INK2, lw=0.8))
    ax.text(1.5, 0.78, "direction of information flow", ha="center", va="bottom", fontsize=7.5, color=S.INK2)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-1.05, 0.95)
    S.save(fig, path)


def _wiring(units, motor, n_in=7):
    from ncps import wirings
    w = wirings.AutoNCP(units, motor)
    w.build(n_in)
    types = [w.get_type_of_neuron(i) for i in range(units)]
    return w, types


def fig_wiring(path):
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL_WIDTH, 3.0))
    for ax, (units, letter) in zip(axs, [(28, "a"), (80, "b")]):
        w, types = _wiring(units, 4)
        groups = {"sensory": list(range(7))}
        for t in ("inter", "command", "motor"):
            groups[t] = [i for i, ty in enumerate(types) if ty == t]
        xs = {"sensory": 0, "inter": 1, "command": 2, "motor": 3}
        P = {}
        for g, idx in groups.items():
            ys = np.linspace(-1, 1, len(idx)) if len(idx) > 1 else [0]
            for k, i in enumerate(idx):
                P[(g == "sensory", i)] = (xs[g], ys[k])
        A, Sa = w.adjacency_matrix, w.sensory_adjacency_matrix
        lw, al = (0.22, 0.45) if units > 30 else (0.5, 0.7)
        for i, j in zip(*np.nonzero(Sa)):
            p, q = P[(True, i)], P[(False, j)]
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#86b6ef" if Sa[i, j] > 0 else "#f19a98", lw=lw, alpha=al, zorder=1)
        for i, j in zip(*np.nonzero(A)):
            p, q = P[(False, i)], P[(False, j)]
            if p[0] == q[0]:
                ax.add_patch(FancyArrowPatch(p, q, connectionstyle="arc3,rad=0.6", arrowstyle="-", lw=lw,
                                             color="#86b6ef" if A[i, j] > 0 else "#f19a98", alpha=al, zorder=1))
            else:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#86b6ef" if A[i, j] > 0 else "#f19a98", lw=lw,
                        alpha=al, zorder=1)
        for g, idx in groups.items():
            pts = np.array([P[(g == "sensory", i)] for i in idx])
            ax.scatter(pts[:, 0], pts[:, 1], s=30 if units > 30 else 55, color=S.NEURON[g], edgecolor="white",
                       linewidth=0.8, zorder=3)
            ax.text(xs[g], 1.12, f"{len(idx)}", ha="center", va="bottom", fontsize=8, color=S.INK2)
        ax.set_axis_off()
        ax.set_xlim(-0.4, 3.4)
        ax.set_ylim(-1.1, 1.3)
        S.panel_label(ax, letter, f"AutoNCP({units}, 4): {int((A != 0).sum() + (Sa != 0).sum())} synapses")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="none", marker="o", ms=6, color=S.NEURON[g], markeredgecolor="white") for g in S.NEURON]
    h += [Line2D([], [], color="#86b6ef", lw=1.4), Line2D([], [], color="#f19a98", lw=1.4)]
    fig.legend(h, ["sensory (inputs)", "inter", "command", "motor (outputs)", "excitatory", "inhibitory"],
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, path)


# ------------------------------------------------------------------ data figures
def fig_index_histories(cfg, path, data_dir=None):
    pr = cfg["protocol"]
    keys = ["NIFTY50", "NIFTY100", "NIFTY500", "SPX", "N225", "SSE"]
    fig, axs = plt.subplots(2, 3, figsize=(S.FULL_WIDTH, 3.6))
    for ax, k in zip(axs.flat, keys):
        raw = load_index(data_path(cfg, k, data_dir))
        df = load_index(data_path(cfg, k, data_dir), cfg["indices"][k].get("start"))
        excl = raw[raw.Date < df.Date.iloc[0]]
        if len(excl):
            ax.plot(excl.Date, excl.AdjClose, color=S.INK3, lw=0.9)
        ax.plot(df.Date, df.AdjClose, color=S.ARCH["cfc"], lw=1.0)
        pairs = make_pairs(df, "ohlc", "levels", pr["feature_cols"], pr["add_weekday"])
        _, b, c = fixed_origin_window(pairs.n, pr["test_frac"])
        ax.axvspan(pairs.target_dates[b], pairs.target_dates[c - 1], color="#e9e9e9", zorder=0)
        ax.set_yscale("log")
        ax.set_title(S.MARKET_LABEL[k])
        lo, hi = df.AdjClose.min(), df.AdjClose.max()
        if len(excl):
            lo = min(lo, excl.AdjClose.min())
        subs = (1.0,) if hi / lo > 300 else ((1.0, 2.0, 5.0) if hi / lo > 8 else (1.0, 1.5, 2.0, 3.0, 5.0, 7.0))
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=subs))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(axis="x", labelrotation=0)
        ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Line2D([], [], color=S.ARCH["cfc"], lw=1.2), Line2D([], [], color=S.INK3, lw=1.2),
                Patch(color="#e9e9e9")],
               ["analysed sample", "excluded: vendor reports O = H = L = C", "fixed-origin test window (95-5)"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.supylabel("adjusted close (log scale)", fontsize=8.5, color=S.INK2, x=0.005)
    fig.tight_layout(rect=(0.01, 0.05, 1, 1))
    S.save(fig, path)


def _common(cfg, data_dir=None):
    fr = {k: load_index(data_path(cfg, k, data_dir), cfg["indices"][k].get("start")).set_index("Date")["AdjClose"]
          for k in ["NIFTY50", "SPX", "N225", "SSE"]}
    return pd.concat(fr, axis=1, join="inner").dropna()


def fig_indices_normalised(cfg, path, data_dir=None):
    J = _common(cfg, data_dir)
    N = (100 * J / J.iloc[0]).resample("W-FRI").last().dropna()     # weekly closes: same story, less ink
    fig, ax = plt.subplots(figsize=(S.FULL_WIDTH, 2.7))
    ax.axhline(100, color=S.INK3, lw=0.6)
    for k in ["SPX", "N225", "SSE", "NIFTY50"]:
        ax.plot(N.index, N[k], color=S.MARKET[k], lw=1.1 if k == "NIFTY50" else 0.9, zorder=3 if k == "NIFTY50" else 2)
    ax.set_ylabel(f"level, {J.index[0]:%-d %b %Y} = 100")
    ax.set_xlim(N.index[0], N.index[-1])
    ticks = [t for t in pd.date_range(f"{N.index[0].year + 1}-01-01", N.index[-1], freq="2YS")]
    ax.set_xticks(ticks, [f"{t:%Y}" for t in ticks])
    ax.legend([plt.Line2D([], [], color=S.MARKET[k], lw=1.4) for k in ["NIFTY50", "SPX", "N225", "SSE"]],
              [S.MARKET_LABEL[k] for k in ["NIFTY50", "SPX", "N225", "SSE"]], ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    S.save(fig, path)


def fig_return_correlation(cfg, path, data_dir=None):
    J = _common(cfg, data_dir)
    R = np.log(J).diff().dropna()
    C = R.corr().to_numpy()
    labs = [S.MARKET_LABEL[k] for k in J.columns]
    n = len(labs)
    fig, ax = plt.subplots(figsize=(S.HALF_WIDTH + 0.6, S.HALF_WIDTH - 0.2))
    M = C[1:, :-1].copy()
    M[np.triu_indices(n - 1, 1)] = np.nan                        # strictly lower triangle, no diagonal
    im = ax.imshow(M, cmap=S.DIVERGING, vmin=-1, vmax=1)
    for i in range(n - 1):
        for j in range(i + 1):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(M[i, j]) > 0.6 else S.INK)
    ax.set_xticks(range(n - 1), labs[:-1], rotation=30, ha="right")
    ax.set_yticks(range(n - 1), labs[1:])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Pearson correlation of daily log returns", fontsize=7.5, color=S.INK2)
    S.save(fig, path)


def main():
    ap = cli.base_parser(__doc__)
    ap.add_argument("--paper-dir", default=None)
    args = ap.parse_args()
    cfg = cli.load(args)
    S.apply()
    fdir = REPO / "results" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    jobs = {"fig_flow.pdf": lambda p: fig_flow(p), "fig_ncp_architecture.pdf": lambda p: fig_ncp_architecture(p),
            "fig_wiring_cfc.pdf": lambda p: fig_wiring(p),
            "fig_index_histories.pdf": lambda p: fig_index_histories(cfg, p, args.data_dir),
            "fig_indices_normalised.pdf": lambda p: fig_indices_normalised(cfg, p, args.data_dir),
            "fig_return_correlation.pdf": lambda p: fig_return_correlation(cfg, p, args.data_dir)}
    for name, fn in jobs.items():
        fn(fdir / name)
        if args.paper_dir:
            shutil.copy(fdir / name, Path(args.paper_dir) / name)
        print("wrote", fdir / name)


if __name__ == "__main__":
    main()

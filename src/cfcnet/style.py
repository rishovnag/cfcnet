"""One visual system for every figure of the paper (journal print and screen).

Typography
    TeX Gyre Heros (a Helvetica clone under the GUST font licence, bundled in ./fonts as TrueType so
    that the PDF embeds Type 42 fonts), 7.5 pt tick labels and 8 pt axis labels at the final printed
    size. Figures are drawn at the journal text width (372 pt = 5.17 in), so nothing is rescaled.

Colour
    Architectures carry their family in the hue: the continuous-time recurrences are blues (CFC dark,
    LTC light), the gated recurrences are warm (GRU brick red, LSTM ochre). The palette passes the
    lightness-band, chroma, colour-vision-deficiency (all pairs, deltaE >= 15) and normal-vision checks;
    every architecture also has its own marker, so identity never rests on colour alone and the
    figures remain legible in greyscale print.
    Benchmarks are neutral ink: random walk = black dashed, close-anchored naive = grey dotted,
    ARIMA = grey dash-dot. The realised series is near-black and slightly heavier than forecasts.

Layout
    No chart titles inside the axes: each panel carries a bold lower-case letter and a short
    descriptor; one shared legend sits above the panels; only left and bottom spines; outward ticks;
    no background grid (reference lines carry the comparisons).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT = "CFCNet Heros"           # TeX Gyre Heros, renamed so that the bundled TrueType is used

INK = "#1a1a1a"
INK2 = "#4d4d4d"          # secondary text, axes
INK3 = "#8c8c8c"          # muted (naive benchmark, annotations)
RULE = "#bfbfbf"          # reference rules (training maximum etc.)
SURFACE = "#ffffff"

ARCH = {"cfc": "#1d4f91", "ltc": "#4a90d9", "gru": "#a83a32", "lstm": "#c68a1d"}
ARCH_LABEL = {"cfc": "CFC", "ltc": "LTC", "lstm": "LSTM", "gru": "GRU"}
MARKER = {"cfc": "o", "ltc": "^", "lstm": "s", "gru": "D"}
MSIZE = {"cfc": 3.6, "ltc": 3.9, "lstm": 3.3, "gru": 3.2}

BENCH = {"rw": dict(color=INK, ls=(0, (4, 2.5)), lw=0.75, label="Random walk"),
         "naive": dict(color=INK3, ls=(0, (1, 1.6)), lw=1.0, label="Close-anchored naive"),
         "arima": dict(color=INK3, ls=(0, (5, 1.5, 1, 1.5)), lw=0.8, label="ARIMA")}
ACTUAL = dict(color=INK, lw=1.0, label="Realised")

MARKET = {"NIFTY50": INK, "SPX": "#1d4f91", "N225": "#a83a32", "SSE": "#c68a1d"}
MARKET_LABEL = {"NIFTY50": "NIFTY 50", "NIFTY100": "NIFTY 100", "NIFTY500": "NIFTY 500",
                "SPX": "S&P 500", "N225": "Nikkei 225", "SSE": "SSE Composite"}

# market regimes: very light tints, legible in greyscale
REGIME = {"bull": "#e3ecf7", "sideways": "#ffffff", "bear": "#f6e3e1"}
REGIME_EDGE = {"bull": "#b9cde6", "sideways": "#d9d9d9", "bear": "#e4b9b4"}

BLUE_RAMP = ["#e3ecf7", "#c3d6ee", "#9dbce3", "#6f9bd2", "#4a7fc0", "#2f64a8", "#1d4f91", "#123766"]
NEURON = {"sensory": "#c3d6ee", "inter": "#6f9bd2", "command": "#2f64a8", "motor": "#123766"}

DIVERGING = LinearSegmentedColormap.from_list(
    "cfc_div", ["#a83a32", "#d99790", "#f5f5f5", "#9dbce3", "#1d4f91"], N=256)
SEQUENTIAL = LinearSegmentedColormap.from_list("cfc_seq", BLUE_RAMP, N=256)

TARGET_LABEL = {"adjclose": "Adjusted close", "ohlc": "Joint OHLC"}
REPR_LABEL = {"levels": "price levels", "returns": "log returns"}

FULL_WIDTH = 5.17         # inches: the journal text width (372 pt)
HALF_WIDTH = 2.5


def _register_fonts() -> None:
    for f in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))


def apply() -> None:
    """Install the paper style for all subsequent matplotlib figures."""
    _register_fonts()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.cal": FONT, "mathtext.fallback": "stixsans",
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 5,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": INK2, "axes.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.axisbelow": True, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 3,
        "lines.linewidth": 1.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 3.5, "lines.markeredgewidth": 0.4, "lines.markeredgecolor": SURFACE,
        "legend.frameon": False, "legend.handlelength": 2.0, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.4, "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
        "axes.unicode_minus": True,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=list(ARCH.values())),
    })


def arch_kw(arch: str, **extra) -> dict:
    """Line/marker style of one architecture."""
    kw = dict(color=ARCH[arch], marker=MARKER[arch], ms=MSIZE[arch], label=ARCH_LABEL[arch],
              markeredgecolor=SURFACE, markeredgewidth=0.4)
    kw.update(extra)
    return kw


def arch_handle(arch: str, line: bool = True, **extra) -> Line2D:
    kw = arch_kw(arch, **extra)
    if not line:
        kw["ls"] = "none"
    return Line2D([], [], **kw)


def panel(ax, letter: str, text: str | None = None, x: float = 0.0, y: float = 1.0) -> None:
    """Bold lower-case panel letter followed by a short descriptor, above the top-left corner."""
    ax.annotate(letter, (x, y), xycoords="axes fraction", xytext=(-2, 5), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, fontweight="bold", color=INK)
    if text:
        ax.annotate(text, (x, y), xycoords="axes fraction", xytext=(9, 5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8, color=INK)


def panel_label(ax, letter: str, title: str | None = None) -> None:
    """Backward-compatible alias of panel()."""
    panel(ax, letter, title)


def direct_label(ax, x, y, text, color=INK2, dx=3, dy=0, ha="left", va="center", size=7, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                color=color, fontsize=size, **kw)


def top_legend(fig, handles, labels, ncol=None, y=1.0):
    fig.legend(handles, labels, loc="lower center", ncol=ncol or len(handles), bbox_to_anchor=(0.5, y),
               frameon=False)


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)

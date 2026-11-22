# ruff: noqa: E501
"""Paper figures: one shared style (order, colour, marker, label per source), two-panel line
figures (open models left, hosted right, humans and Bayes in both), direct end-of-line labels,
vector PDF at the final placed width plus a PNG copy. Style only: every number comes from the
CSVs the paper generator already writes under paper/data.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .task import D_LEVELS, Q_LEVELS  # noqa: E402

TEXT_WIDTH_IN = 6.27  # A4, 1-inch margins
HALF_WIDTH_IN = 3.05
FONT_PT = 8.5
HUMAN = "human"
BAYES = "Bayes"
INK = "#111111"
GRID = "#e6e6e6"

# ---------------------------------------------------------------------------- style table
# family -> (marker, ordered members light -> dark, ramp)
FAMILIES: dict[str, dict] = {
    "qwen": {
        "marker": "o",
        "members": ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"],
        "ramp": ["#9ecae1", "#6baed6", "#2171b5", "#08306b"],
        "group": "open",
    },
    "open": {
        "marker": "s",
        "members": ["llama3.1:8b", "gemma2:9b", "granite4.2:8b", "phi3.5:3.8b"],
        "ramp": ["#e6550d", "#31a354", "#8c510a", "#238b45"],
        "group": "open",
    },
    "haiku": {
        "marker": "^",
        "members": ["claude-haiku-4-5", "claude-haiku-5-5"],
        "ramp": ["#9e9ac8", "#54278f"],
        "group": "hosted",
    },
    "sonnet": {
        "marker": "D",
        "members": ["claude-sonnet-4-6", "claude-sonnet-5", "claude-sonnet-5-5"],
        "ramp": ["#fb6a4a", "#cb181d", "#67000d"],
        "group": "hosted",
    },
}
SHORT = {
    "qwen2.5:3b": "Qwen2.5 3B", "qwen2.5:7b": "Qwen2.5 7B", "qwen2.5:14b": "Qwen2.5 14B", "qwen2.5:32b": "Qwen2.5 32B",
    "llama3.1:8b": "Llama 3.1 8B", "gemma2:9b": "Gemma 2 9B", "granite4.2:8b": "Granite 4.2 8B", "phi3.5:3.8b": "Phi 3.5",
    "claude-haiku-4-5": "Haiku 4.5", "claude-haiku-5-5": "Haiku 5.5",
    "claude-sonnet-4-6": "Sonnet 4.6", "claude-sonnet-5": "Sonnet 5", "claude-sonnet-5-5": "Sonnet 5.5",
    HUMAN: "Humans", BAYES: "Bayes",
}  # fmt: skip


def parse(source: str) -> dict:
    """Tag, mode, condition, partial label from a source string such as
    'claude-haiku-5-5 [main, T1, reasoning allowed]' or 'qwen2.5:3b [stateless, T0.7]'."""
    m = re.match(r"^(?P<tag>[^\[]+?)\s*(?:\[(?P<inside>[^\]]*)\])?$", source)
    tag = m.group("tag").strip()
    parts = [p.strip() for p in (m.group("inside") or "").split(",") if p.strip()]
    mode = "main"
    cond = ""
    partial = ""
    for p in parts:
        if p in ("main", "stateless"):
            mode = p
        elif p.startswith("reasoning"):
            cond = "reasoning"
        elif p == "number only":
            cond = "number only"
        elif p.startswith("PARTIAL"):
            partial = p.replace("PARTIAL ", "")
        elif p == "greedy":
            mode = "greedy"
    return {"tag": tag, "mode": mode, "cond": cond, "partial": partial}


def spec(source: str) -> dict:
    """Fixed style for a source: label, group (open/hosted/human/bayes), colour, marker,
    linestyle, sort key."""
    p = parse(source)
    tag = p["tag"]
    if tag == HUMAN or tag.startswith("human"):
        return {
            "label": "Humans",
            "group": "human",
            "color": "#000000",
            "marker": None,
            "ls": "-",
            "lw": 2.0,
            "key": (0, 0, 0),
            **p,
        }
    if tag == BAYES:
        return {
            "label": "Bayes",
            "group": "bayes",
            "color": "#7a7a7a",
            "marker": None,
            "ls": "--",
            "lw": 1.1,
            "key": (9, 0, 0),
            **p,
        }
    for fi, (fam, f) in enumerate(FAMILIES.items()):
        if tag in f["members"]:
            i = f["members"].index(tag)
            label = SHORT.get(tag, tag)
            ls = "-"
            if p["mode"] == "stateless":
                label += ", stateless"
                ls = (0, (4, 2))
            elif p["mode"] == "greedy":
                label += ", greedy"
                ls = (0, (1, 1.5))
            if p["cond"] == "reasoning":
                label += ", reasoning"
                ls = (0, (1, 1.5))
            if p["partial"]:
                label += f" ({p['partial']})"
            sub = 0 if p["mode"] == "main" and p["cond"] != "reasoning" else 1
            return {
                "label": label,
                "group": f["group"],
                "color": f["ramp"][i],
                "marker": f["marker"],
                "ls": ls,
                "lw": 1.1,
                "key": (1 + fi, i, sub),
                "family": fam,
                **p,
            }
    return {
        "label": source,
        "group": "open",
        "color": "#999999",
        "marker": "x",
        "ls": "-",
        "lw": 1.0,
        "key": (8, 0, 0),
        **p,
    }


def order(sources: Sequence[str]) -> list[str]:
    return sorted(dict.fromkeys(sources), key=lambda s: spec(s)["key"])


def split_groups(sources: Sequence[str]) -> tuple[list[str], list[str]]:
    """(open models, hosted models), each in fixed order, without humans and Bayes."""
    sp = {s: spec(s) for s in sources}
    return order([s for s in sources if sp[s]["group"] == "open"]), order(
        [s for s in sources if sp[s]["group"] == "hosted"]
    )


def rc() -> dict:
    return {
        "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman", "STIXGeneral"],
        "mathtext.fontset": "cm", "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
        "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT, "xtick.labelsize": FONT_PT - 0.5,
        "ytick.labelsize": FONT_PT - 0.5, "legend.fontsize": FONT_PT - 0.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.markersize": 3.6, "lines.linewidth": 1.1, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "legend.frameon": False,
        "figure.dpi": 100,
    }  # fmt: skip


def line_kwargs(source: str, markers: bool = True) -> dict:
    s = spec(source)
    kw = {
        "color": s["color"],
        "linestyle": s["ls"],
        "linewidth": s["lw"],
        "zorder": 5 if s["group"] == "human" else 3,
    }
    if markers and s["marker"]:
        kw.update(
            marker=s["marker"],
            markersize=3.6,
            markeredgewidth=0.6,
            markerfacecolor="white"
            if s["mode"] != "main" or s["cond"] == "reasoning"
            else s["color"],
            markeredgecolor=s["color"],
        )
    return kw


# ---------------------------------------------------------------------------- direct labels
#
# Every line figure labels its series in a margin to the right of the panel: a short swatch in
# the series style (colour, dash, marker) followed by the name in near-black ink. Labels never
# sit over the data; when two line ends are too close the labels are pushed apart and a thin
# grey leader joins the line end to its label.

LABEL_PT = FONT_PT - 1
SWATCH_IN = 0.17  # length of the swatch line in the margin
LABEL_PAD_IN = 0.05  # gap between the panel edge and the swatch, and between swatch and text


def multi_turn(sources: Sequence[str]) -> list[str]:
    """The complete multi-turn sources only: drop every stateless run (appendix line figures)."""
    return [s for s in sources if "stateless" not in s and "stateless" not in spec(s)["label"]]


def _text_width_in(fig: plt.Figure, text: str, fontsize: float) -> float:
    """Rendered width of ``text`` in inches (measured, so it is right for mathtext too)."""
    t = fig.text(0, 0, text, fontsize=fontsize)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def label_margin_in(fig: plt.Figure, texts: Sequence[str], fontsize: float = LABEL_PT) -> float:
    """Width of the margin a label column needs: swatch, pads and the widest text."""
    widest = max((_text_width_in(fig, t, fontsize) for t in texts), default=0.0)
    return LABEL_PAD_IN + SWATCH_IN + LABEL_PAD_IN + widest + 0.04


def layout_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    label_cols: Sequence[float],
    left_in: float = 0.6,
    right_in: float = 0.02,
    top_in: float = 0.25,
    bottom_in: float = 0.45,
    hgap_in: float = 0.12,
    vgap_in: float = 0.4,
) -> None:
    """Place a (nrows x ncols) grid of axes with explicit margins in inches. Every panel has the
    same size; ``label_cols[c]`` inches are left free right of column ``c`` for its labels."""
    W, H = fig.get_size_inches()
    nrow, ncol = axes.shape
    w = (W - left_in - right_in - float(sum(label_cols)) - hgap_in * (ncol - 1)) / ncol
    h = (H - top_in - bottom_in - vgap_in * (nrow - 1)) / nrow
    x = left_in
    for c in range(ncol):
        for r in range(nrow):
            y = H - top_in - (r + 1) * h - r * vgap_in
            axes[r, c].set_position([x / W, y / H, w / W, h / H])
        x += w + label_cols[c] + hgap_in


def _spread(desired: Sequence[float], sep: float, lo: float, hi: float) -> list[float]:
    """1-D label positions at least ``sep`` apart inside [lo, hi], as close as possible to the
    sorted ``desired`` positions (clusters of touching labels are centred on their mean)."""
    n = len(desired)
    if n == 0:
        return []
    if hi - lo < (n - 1) * sep:  # no room: spread evenly over the whole range
        return list(np.linspace(lo, hi, n)) if n > 1 else [(lo + hi) / 2]
    clusters: list[list] = []  # [first index, last index, centre]
    for i, d in enumerate(desired):
        clusters.append([i, i, d])
        while len(clusters) > 1:
            a, b = clusters[-2], clusters[-1]
            a_top = a[2] + (a[1] - a[0]) * sep / 2
            b_bot = b[2] - (b[1] - b[0]) * sep / 2
            if b_bot >= a_top + sep:
                break
            idx = range(a[0], b[1] + 1)
            clusters[-2:] = [[a[0], b[1], sum(desired[k] for k in idx) / len(idx)]]
    pos: list[float] = []
    for a0, a1, c in clusters:
        m = a1 - a0 + 1
        pos.extend(c - (m - 1) * sep / 2 + k * sep for k in range(m))
    for k in range(n):  # clamp into the range, keeping the separation
        pos[k] = max(pos[k], lo if k == 0 else max(lo, pos[k - 1] + sep))
    for k in range(n - 1, -1, -1):
        pos[k] = min(pos[k], hi if k == n - 1 else min(hi, pos[k + 1] - sep))
    return pos


def direct_labels(
    ax: plt.Axes,
    items: Sequence[tuple],
    x: float | Sequence[float] | None = None,
    fontsize: float = LABEL_PT,
    min_sep_pt: float | None = None,
    leader_from: str = "data",
) -> None:
    """Label line ends in the margin right of ``ax``. ``items`` are ``(y, source)`` or
    ``(y, text, source)`` tuples: the swatch takes the style of ``source`` and the text is
    ``spec(source)['label']`` unless given. ``x`` is the data x of the line ends (one value or
    one per item); a leader starts there (or at the panel edge when ``leader_from == 'edge'``
    or ``x`` is None) whenever a label had to move away from its line. Text is drawn in ink."""
    xs = list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x] * len(items)
    rows = []
    for it, xi in zip(items, xs, strict=True):
        y, src = it[0], it[-1]
        text = it[1] if len(it) == 3 else spec(src)["label"]
        if y is None or not np.isfinite(y):
            continue
        rows.append((float(y), text, src, xi))
    if not rows:
        return
    fig = ax.figure
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72
    px_per_in = fig.dpi
    sep = (min_sep_pt or fontsize * 1.25) * px_per_pt
    xmid = np.mean(ax.get_xlim()) if ax.get_xscale() != "log" else np.sqrt(np.prod(ax.get_xlim()))
    rows.sort(key=lambda r: ax.transData.transform((xmid, r[0]))[1])
    desired = [ax.transData.transform((xmid, r[0]))[1] for r in rows]
    bb = ax.bbox
    desired = [min(max(v, bb.y0), bb.y1) for v in desired]
    placed = _spread(desired, sep, bb.y0 + 0.45 * sep, bb.y1 - 0.45 * sep)
    aw, ah = bb.width, bb.height
    x_sw0 = 1 + LABEL_PAD_IN * px_per_in / aw
    x_sw1 = x_sw0 + SWATCH_IN * px_per_in / aw
    x_txt = x_sw1 + LABEL_PAD_IN * px_per_in / aw
    for (y, text, src, xi), yd, yp in zip(rows, desired, placed, strict=True):
        ya = (yp - bb.y0) / ah
        kw = line_kwargs(src)
        kw["zorder"] = 6
        ax.plot([x_sw0, x_sw1], [ya, ya], transform=ax.transAxes, clip_on=False, **kw)
        ax.text(
            x_txt,
            ya,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
            color=INK,
            clip_on=False,
            zorder=10,
        )
        if abs(yp - yd) > 1.5 * px_per_pt:
            yda = (yd - bb.y0) / ah
            if leader_from == "data" and xi is not None and np.isfinite(xi):
                xa = (ax.transData.transform((xi, y))[0] - bb.x0) / aw
            else:
                xa = 1.0
            ax.plot(
                [xa, x_sw0 - 0.3 * LABEL_PAD_IN * px_per_in / aw],
                [yda, ya],
                transform=ax.transAxes,
                color="#9a9a9a",
                lw=0.5,
                clip_on=False,
                zorder=2,
            )


def _save(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(pdf, format="pdf")  # no bbox_inches: the figure is placed at its own width
    fig.savefig(png, format="png", dpi=220)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------- Figure 1: dot plot


def fig_distance_dotplot(
    acc: pd.DataFrame, out_stem: Path, n_subjects: int = 30
) -> tuple[Path, Path]:
    """Sorted dot plot of mean |answer - Bayesian posterior| with bootstrap 95% intervals, one
    row per complete ``n_subjects`` source, open and hosted groups separated; human mean and
    interval as a shaded band; a line at zero labelled Bayes."""
    df = acc[acc.n_subjects == n_subjects].copy()
    hum = df[df.source == HUMAN].iloc[0]
    df = df[df.source != HUMAN]
    open_s, hosted_s = split_groups(df.source.tolist())
    open_s = sorted(open_s, key=lambda s: -float(df.set_index("source").loc[s, "mean_abs_dev"]))
    hosted_s = sorted(hosted_s, key=lambda s: -float(df.set_index("source").loc[s, "mean_abs_dev"]))
    rows = [(s, "open") for s in open_s] + [(s, "hosted") for s in hosted_s]
    with plt.rc_context(rc()):
        h = 0.17 * (len(rows) + 3) + 0.5
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, h))
        ax.grid(axis="y", visible=False)
        d = df.set_index("source")
        ys = []
        y = 0.0
        last_group = None
        for s, g in rows:
            if last_group is not None and g != last_group:
                y -= 1.9  # gap between the groups, with room for the next heading
            last_group = g
            ys.append(y)
            r = d.loc[s]
            st = spec(s)
            ax.plot(
                [r.ci_low, r.ci_high],
                [y, y],
                color=st["color"],
                lw=1.2,
                solid_capstyle="butt",
                zorder=3,
            )
            ax.plot(
                [r.mean_abs_dev],
                [y],
                marker=st["marker"] or "o",
                color=st["color"],
                markersize=5,
                markeredgewidth=0.7,
                markerfacecolor="white"
                if st["mode"] != "main" or st["cond"] == "reasoning"
                else st["color"],
                markeredgecolor=st["color"],
                linestyle="none",
                zorder=4,
            )
            y -= 1.0
        ax.axvspan(hum.ci_low, hum.ci_high, color="#000000", alpha=0.10, lw=0, zorder=1)
        ax.axvline(hum.mean_abs_dev, color="#000000", lw=1.4, zorder=2)
        ax.axvline(0, color="#7a7a7a", lw=1.0, ls="--", zorder=2)
        # row labels in ink; the marker in the row is the swatch
        ax.set_yticks(ys, [spec(s)["label"] for s, _ in rows])
        for tick in ax.get_yticklabels():
            tick.set_color(INK)
        ax.set_ylim(min(ys) - 0.7, 1.7)
        xmax = max(float(df.ci_high.max()), float(hum.ci_high)) * 1.08
        ax.set_xlim(-0.004, xmax)
        ax.set_xlabel(r"mean $|$stated probability $-$ Bayesian posterior$|$")
        ax.text(
            0,
            0.95,
            " Bayes",
            color=INK,
            fontsize=FONT_PT - 1,
            ha="left",
            va="center",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            hum.mean_abs_dev,
            0.95,
            " Humans",
            color=INK,
            fontsize=FONT_PT - 1,
            ha="left",
            va="center",
            transform=ax.get_xaxis_transform(),
        )
        # horizontal group headings above each block, in the label column
        n_open = len(open_s)
        for label, y_head in (
            ("Open-weight models", ys[0] + 1.0),
            ("Claude models, number only", ys[n_open] + 1.0),
        ):
            ax.annotate(
                label,
                xy=(0.02, y_head),
                xycoords=("figure fraction", "data"),
                ha="left",
                va="center",
                fontsize=FONT_PT,
                color=INK,
                annotation_clip=False,
            )
        fig.subplots_adjust(left=0.26, right=0.95, top=0.97, bottom=0.42 / h)
        return _save(fig, out_stem)


# ---------------------------------------------------------------------------- two-column line figures
#
# Open-weight models left, Claude models right, humans and Bayes in both. Both columns get
# the same panel size and the same label margin (the widest label of either column).


def _grid(nrows: int, height: float, sharey: str | bool = "row") -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        nrows, 2, figsize=(TEXT_WIDTH_IN, height), sharey=sharey, squeeze=False
    )
    axes[0, 0].set_title("Open-weight models", loc="left", fontsize=FONT_PT, color=INK)
    axes[0, 1].set_title("Claude models", loc="left", fontsize=FONT_PT, color=INK)
    return fig, axes


def _finish_grid(
    fig: plt.Figure,
    axes: np.ndarray,
    items: dict[tuple[int, int], list[tuple]],
    x_end: dict[tuple[int, int], float] | None = None,
    label_pt: float = LABEL_PT,
    **layout: float,
) -> None:
    """Lay the grid out with one shared label margin and place every panel's labels."""
    texts = [
        (it[1] if len(it) == 3 else spec(it[-1])["label"]) for lst in items.values() for it in lst
    ]
    margin = label_margin_in(fig, texts, label_pt)
    layout_grid(fig, axes, [margin] * axes.shape[1], **layout)
    for (r, c), lst in items.items():
        xe = (x_end or {}).get((r, c))
        direct_labels(axes[r, c], lst, x=xe, fontsize=label_pt)


# ---------------------------------------------------------------------------- 2 x 2 line figures


def fig_effective_grid(
    eff: pd.DataFrame, sources: Sequence[str], out_stem: Path
) -> tuple[Path, Path]:
    """Main results figure: effective switch rate (top) and effective diagnosticity (bottom),
    open models left, hosted right, humans and the Bayes identity in every panel. Multi-turn
    sources only."""
    open_s, hosted_s = split_groups(multi_turn(sources))
    specs = [
        (
            "effective switch",
            Q_LEVELS,
            "stated switch probability $q$",
            r"effective switch rate $\alpha q$",
        ),
        (
            "effective diag",
            D_LEVELS,
            "stated diagnosticity $d$",
            r"effective diagnosticity $d^{\beta}$",
        ),
    ]
    with plt.rc_context(rc()):
        fig, axes = _grid(2, 4.0)
        items: dict = {}
        x_end: dict = {}
        for r, (quantity, levels, xlabel, ylabel) in enumerate(specs):
            lv = np.array(levels, dtype=float)
            for c, group in enumerate((open_s, hosted_s)):
                ax = axes[r, c]
                ax.set_xscale("log")
                ax.set_yscale("log")
                lo, hi = lv.min() / 1.25, lv.max() * 1.25
                ax.plot([lo, hi], [lo, hi], color="#7a7a7a", ls="--", lw=1.0, zorder=1)
                lst = []
                for src in [HUMAN, *group]:
                    sub = (
                        eff[(eff.source == src) & eff.quantity.str.startswith(quantity)]
                        .set_index("level")
                        .reindex(list(levels))
                    )
                    if sub["mean"].isna().all():
                        continue
                    ax.errorbar(
                        lv,
                        sub["mean"],
                        yerr=sub["sem"],
                        capsize=1.5,
                        elinewidth=0.6,
                        **line_kwargs(src),
                    )
                    lst.append((float(sub["mean"].iloc[-1]), src))
                lst.append((float(hi), BAYES))
                items[(r, c)] = lst
                x_end[(r, c)] = [lv.max()] * (len(lst) - 1) + [hi]
                ax.set_xticks(lv, [f"{v:g}" for v in levels])
                ax.minorticks_off()
                ax.set_xlim(lo, hi)
                if r == 1:
                    ax.set_xlabel(xlabel)
            axes[r, 0].set_ylabel(ylabel)
        _finish_grid(
            fig, axes, items, x_end, left_in=0.62, top_in=0.28, bottom_in=0.42, vgap_in=0.38
        )
        return _save(fig, out_stem)


def fig_parameters_grid(
    params: pd.DataFrame, sources: Sequence[str], out_stem: Path
) -> tuple[Path, Path]:
    """Raw alpha by q (top) and beta by d (bottom), means with SEM, open left, hosted right.
    Multi-turn sources only."""
    open_s, hosted_s = split_groups(multi_turn(sources))
    means = params.pivot(index="source", columns="name", values="mean")
    sems = params.pivot(index="source", columns="name", values="sem")
    with plt.rc_context(rc()):
        fig, axes = _grid(2, 4.0)
        items: dict = {}
        x_end: dict = {}
        for r, (name, levels, xlabel, ylabel) in enumerate(
            (
                ("alpha", Q_LEVELS, "stated switch probability $q$", r"$\alpha$"),
                ("beta", D_LEVELS, "stated diagnosticity $d$", r"$\beta$"),
            )
        ):
            x = np.arange(3)
            for c, group in enumerate((open_s, hosted_s)):
                ax = axes[r, c]
                ax.axhline(1, color="#7a7a7a", ls="--", lw=1.0, zorder=1)
                lst = [(1.0, BAYES)]
                for src in [HUMAN, *group]:
                    if src not in means.index:
                        continue
                    y = [means.loc[src, f"{name}_{i}"] for i in (1, 2, 3)]
                    e = [sems.loc[src, f"{name}_{i}"] for i in (1, 2, 3)]
                    ax.errorbar(x, y, yerr=e, capsize=1.5, elinewidth=0.6, **line_kwargs(src))
                    lst.append((float(y[-1]), src))
                items[(r, c)] = lst
                x_end[(r, c)] = [2.25] + [2.0] * (len(lst) - 1)
                ax.set_xticks(x, [f"{v:g}" for v in levels])
                ax.set_xlim(-0.25, 2.25)
                if name == "alpha":
                    ax.set_yscale("log")
                if r == 1:
                    ax.set_xlabel(xlabel)
            axes[r, 0].set_ylabel(ylabel)
        _finish_grid(
            fig, axes, items, x_end, left_in=0.55, top_in=0.28, bottom_in=0.42, vgap_in=0.38
        )
        return _save(fig, out_stem)


def fig_distance_by_condition(
    dist: pd.DataFrame, sources: Sequence[str], out_stem: Path
) -> tuple[Path, Path]:
    """Mean |answer - posterior| per condition, lines within each q block, open left, hosted
    right; both panels the same width. Multi-turn sources only."""
    open_s, hosted_s = split_groups(multi_turn(sources))
    conds = [(q, d) for q in Q_LEVELS for d in D_LEVELS]
    x = np.arange(len(conds), dtype=float)
    with plt.rc_context(rc()):
        fig, axes = _grid(1, 3.0, sharey=True)
        items: dict = {}
        x_end: dict = {}
        for c, group in enumerate((open_s, hosted_s)):
            ax = axes[0, c]
            lst = []
            for src in [HUMAN, *group]:
                sub = dist[dist.source == src].set_index(["q", "d"]).reindex(conds)
                if sub["mean"].isna().all():
                    continue
                kw = line_kwargs(src)
                for b in range(3):
                    sl = slice(3 * b, 3 * b + 3)
                    ax.errorbar(
                        x[sl],
                        sub["mean"].to_numpy()[sl],
                        yerr=sub["sem"].to_numpy()[sl],
                        capsize=1.2,
                        elinewidth=0.5,
                        **kw,
                    )
                lst.append((float(sub["mean"].iloc[-1]), src))
            items[(0, c)] = lst
            x_end[(0, c)] = 8.0
            for k in (2.5, 5.5):
                ax.axvline(k, color=GRID, lw=0.8)
            ax.set_xticks(x, [f"{d:g}" for _, d in conds])
            ax.set_xlim(-0.4, 8.4)
            for b, q in enumerate(Q_LEVELS):
                ax.text(
                    3 * b + 1,
                    -0.21,
                    f"$q$ = {q:g}",
                    ha="center",
                    va="top",
                    fontsize=FONT_PT - 0.5,
                    color=INK,
                    transform=ax.get_xaxis_transform(),
                )
            ax.set_xlabel("stated diagnosticity $d$ within each $q$ block", labelpad=27)
            ax.set_ylim(bottom=0)
        axes[0, 0].set_ylabel(r"mean $|$stated $-$ posterior$|$")
        _finish_grid(fig, axes, items, x_end, left_in=0.55, top_in=0.28, bottom_in=0.8)
        return _save(fig, out_stem)


def fig_calibration_two_panel(
    cal: pd.DataFrame, sources: Sequence[str], out_stem: Path
) -> tuple[Path, Path]:
    """Mean stated probability against the mean Bayesian posterior in five posterior bins; the
    dashed diagonal is perfect calibration. Open left, hosted right, humans in both. Multi-turn
    sources only."""
    open_s, hosted_s = split_groups(multi_turn(sources))
    with plt.rc_context(rc()):
        fig, axes = _grid(1, 3.0, sharey=True)
        items: dict = {}
        x_end: dict = {}
        for c, group in enumerate((open_s, hosted_s)):
            ax = axes[0, c]
            ax.plot([0, 1], [0, 1], color="#7a7a7a", ls="--", lw=1.0, zorder=1)
            lst = []
            xe = []
            for src in [HUMAN, *group]:
                sub = cal[cal.source == src].sort_values("mean_bayes")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub.mean_bayes,
                    sub.mean_stated,
                    yerr=sub.sem_stated,
                    capsize=1.2,
                    elinewidth=0.5,
                    **line_kwargs(src),
                )
                lst.append((float(sub.mean_stated.iloc[-1]), src))
                xe.append(float(sub.mean_bayes.iloc[-1]))
            lst.append((1.0, BAYES))
            xe.append(1.0)
            items[(0, c)] = lst
            x_end[(0, c)] = xe
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("mean Bayesian posterior in bin")
        axes[0, 0].set_ylabel("mean stated probability")
        _finish_grid(fig, axes, items, x_end, left_in=0.55, top_in=0.28, bottom_in=0.42)
        return _save(fig, out_stem)


def fig_signal_response_grid(
    sig: pd.DataFrame, sources: Sequence[str], out_stem: Path
) -> tuple[Path, Path]:
    """Mean change in the stated probability after a blue ball (top) and after a red ball
    (bottom) at each d; open left, hosted right; Bayes dashed, humans black. Multi-turn sources
    only."""
    open_s, hosted_s = split_groups(multi_turn(sources))
    sig = sig.copy()
    sig["d"] = pd.to_numeric(sig["d"], errors="coerce")  # the frame also carries an 'all' row
    sig = sig.dropna(subset=["d"])
    x = np.arange(3, dtype=float)
    with plt.rc_context(rc()):
        fig, axes = _grid(2, 4.0)
        items: dict = {}
        x_end: dict = {}
        for r, after in enumerate(("blue", "red")):
            for c, group in enumerate((open_s, hosted_s)):
                ax = axes[r, c]
                ax.axhline(0, color=GRID, lw=0.8)
                lst = []
                for src in [BAYES, HUMAN, *group]:
                    sub = (
                        sig[(sig.source == src) & (sig.after == after)]
                        .set_index("d")
                        .reindex(list(D_LEVELS))
                    )
                    if sub.mean_delta.isna().all():
                        continue
                    ax.errorbar(
                        x,
                        sub.mean_delta,
                        yerr=sub["sem"],
                        capsize=1.2,
                        elinewidth=0.5,
                        **line_kwargs(src, markers=src != BAYES),
                    )
                    lst.append((float(sub.mean_delta.iloc[-1]), src))
                items[(r, c)] = lst
                x_end[(r, c)] = 2.0
                ax.set_xticks(x, [f"{d:g}" for d in D_LEVELS])
                ax.set_xlim(-0.25, 2.25)
                if r == 1:
                    ax.set_xlabel("stated diagnosticity $d$")
            axes[r, 0].set_ylabel(f"mean change after a {after} ball")
        _finish_grid(
            fig, axes, items, x_end, left_in=0.62, top_in=0.28, bottom_in=0.42, vgap_in=0.38
        )
        return _save(fig, out_stem)


# ---------------------------------------------------------------------------- overreaction heatmaps


DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#f0efec", "#d03b3b"])


IO_WIDTH_IN = 4.95  # narrower than the text: keeps the 4 x 4 grid of square panels short enough to share a page


def _is_dark(rgba) -> bool:
    """True when white text reads better than ink on this colour (relative luminance below 0.45)."""
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45


def fig_io_by_family(
    io: pd.DataFrame, sources: Sequence[str], out_stem: Path
) -> tuple[Path, Path, float]:
    """Index-of-overreaction grids (rows d, columns q): one regular grid of equal square
    panels, four per row, humans first and then every source in the fixed family order. One
    symmetric colour scale set to the largest absolute cell mean in the data; the colour bar
    sits under the grid."""
    panels = [HUMAN, *order(multi_turn(sources))]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    piv = {
        s: io[io.source == s]
        .pivot(index="d", columns="q", values="mean")
        .reindex(index=list(D_LEVELS), columns=list(Q_LEVELS))
        for s in panels
    }
    vmax = float(max(np.nanmax(np.abs(p.to_numpy())) for p in piv.values()))
    vmax = float(np.ceil(vmax * 100) / 100)
    # geometry in inches: square panels, room for a title above and tick labels below each
    left_in, right_in, hgap_in, vgap_in = 0.42, 0.06, 0.30, 0.52
    top_in, cbar_in, bottom_in = 0.22, 0.95, 0.12
    panel_in = (IO_WIDTH_IN - left_in - right_in - hgap_in * (ncol - 1)) / ncol
    H = top_in + nrow * panel_in + (nrow - 1) * vgap_in + cbar_in + bottom_in
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(nrow, ncol, figsize=(IO_WIDTH_IN, H), squeeze=False)
        im = None
        for k, ax in enumerate(axes.ravel()):
            r, c = divmod(k, ncol)
            x0 = left_in + c * (panel_in + hgap_in)
            y0 = H - top_in - (r + 1) * panel_in - r * vgap_in
            ax.set_position([x0 / IO_WIDTH_IN, y0 / H, panel_in / IO_WIDTH_IN, panel_in / H])
            if k >= len(panels):
                ax.set_visible(False)
                continue
            s_ = panels[k]
            p_ = piv[s_].to_numpy()
            im = ax.imshow(p_, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
            for (a, b), v in np.ndenumerate(p_):
                ax.text(
                    b,
                    a,
                    f"{v:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=FONT_PT - 1.5,
                    color="white" if _is_dark(DIVERGING((v + vmax) / (2 * vmax))) else INK,
                )
            ax.set_xticks(range(3), [f"{q:g}" for q in Q_LEVELS])
            ax.set_yticks(range(3), [f"{d:g}" for d in D_LEVELS])
            ax.tick_params(length=0, pad=2)
            ax.grid(False)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            # title in ink, a marker in the series style as the swatch before it
            st = spec(s_)
            if st["marker"]:
                ax.plot(
                    [0.0],
                    [1.08],
                    transform=ax.transAxes,
                    clip_on=False,
                    linestyle="none",
                    marker=st["marker"],
                    markersize=3.6,
                    markeredgewidth=0.6,
                    markerfacecolor="white" if st["mode"] != "main" else st["color"],
                    markeredgecolor=st["color"],
                )
                ax.text(
                    0.08,
                    1.08,
                    st["label"],
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=FONT_PT - 0.5,
                    color=INK,
                )
            else:
                ax.text(
                    0.0,
                    1.08,
                    st["label"],
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=FONT_PT - 0.5,
                    color=INK,
                )
            if c == 0:
                ax.set_ylabel("$d$")
            ax.set_xlabel("$q$", labelpad=1)
        cax = fig.add_axes([0.30, (bottom_in + 0.36) / H, 0.40, 0.11 / H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("index of overreaction (+ = over-react)", fontsize=FONT_PT - 1, color=INK)
        cb.ax.tick_params(labelsize=FONT_PT - 1.5, length=2)
        cb.outline.set_linewidth(0.5)
        pdf, png = _save(fig, out_stem)
        return pdf, png, vmax


# ---------------------------------------------------------------------------- driver


def make_all(data: dict, out_dir: Path, sources: Sequence[str]) -> dict:
    """Render every paper figure. ``data`` holds the frames the generator already has:
    accuracy, eff, parameters, io, dist, calibration, period_means, signal."""
    out = Path(out_dir)
    made: dict = {}
    made["dotplot"] = fig_distance_dotplot(data["accuracy"], out / "fig_distance_dotplot")
    made["effective"] = fig_effective_grid(data["eff"], sources, out / "fig_effective_grid")
    made["parameters"] = fig_parameters_grid(
        data["parameters"], sources, out / "fig_parameters_grid"
    )
    made["distance"] = fig_distance_by_condition(
        data["dist"], sources, out / "fig_distance_by_condition"
    )
    made["calibration"] = fig_calibration_two_panel(
        data["calibration"], sources, out / "fig_calibration"
    )
    made["signal"] = fig_signal_response_grid(data["signal"], sources, out / "fig_signal_response")
    made["io"] = fig_io_by_family(data["io"], sources, out / "fig_io_by_family")
    return made


# ---------------------------------------------------------------------------- sensitivity map


def _halo():
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=2.0, foreground="white")]


def place_labels_2d(
    ax: plt.Axes,
    items: Sequence[tuple[float, float, str, str, float]],
    obstacles: Sequence[tuple[float, float, float, float]] = (),
    boxes: Sequence = (),
    weights: Sequence[float] | None = None,
) -> None:
    """Label points so that labels avoid each other, every point (and its error-bar box, given
    in data coordinates as x0, x1, y0, y1), the extra display-space ``boxes`` (e.g. an inset)
    and the axes edges. Each label tries eight directions at four distances and keeps the
    cheapest; a thin grey leader joins the point to a label that had to move away. Text is ink
    with a white halo. ``items`` are ``(x, y, text, colour, fontsize)``; ``colour`` is kept for
    compatibility and not used for the text."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tr = ax.transData
    px_per_pt = fig.dpi / 72

    def box_of_obstacle(o):
        (x0, y0) = tr.transform((o[0], o[2]))
        (x1, y1) = tr.transform((o[1], o[3]))
        return Bbox.from_extents(min(x0, x1) - 2, min(y0, y1) - 2, max(x0, x1) + 2, max(y0, y1) + 2)

    bars = [box_of_obstacle(o) for o in obstacles]
    pts = [tr.transform((x, y)) for x, y, *_ in items]
    marks = [Bbox.from_extents(px - 5, py - 5, px + 5, py + 5) for px, py in pts]
    hard = [Bbox.from_extents(b.x0 - 3, b.y0 - 3, b.x1 + 3, b.y1 + 3) for b in boxes]
    axes_box = ax.bbox
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 0.7), (-1, 0.7), (1, -0.7), (-1, -0.7)]
    dists = (4.5, 11, 19, 29, 40)
    # measure every text once
    sizes = []
    for _, _, text, _, fs in items:
        t = ax.text(0, 0, text, fontsize=fs)
        bb = t.get_window_extent(renderer=renderer)
        sizes.append((bb.width + 2, bb.height + 2))
        t.remove()
    placed: list[Bbox] = []
    order_idx = sorted(
        range(len(items)),
        key=lambda k: (
            -sum(1 for q in pts if abs(q[0] - pts[k][0]) < 45 and abs(q[1] - pts[k][1]) < 45)
        ),
    )
    for k in order_idx:
        x, y, text, _colour, fs = items[k]
        px, py = pts[k]
        w, h = sizes[k]
        wk = 1.0 if weights is None else weights[k]
        best = None
        for dist in dists:
            for dx, dy in dirs:
                ox, oy = dist * dx * px_per_pt, dist * dy * px_per_pt
                ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
                va = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
                cx, cy = px + ox, py + oy
                x0 = cx if ha == "left" else (cx - w if ha == "right" else cx - w / 2)
                y0 = cy if va == "bottom" else (cy - h if va == "top" else cy - h / 2)
                bb = Bbox.from_extents(x0, y0, x0 + w, y0 + h)
                cost = 0.03 * dist
                for b in placed:
                    if bb.overlaps(b):
                        cost += 25
                for j, b in enumerate(marks):
                    if j != k and bb.overlaps(b):
                        cost += 15
                for b in bars:
                    if bb.overlaps(b):
                        cost += 3
                for b in hard:
                    if bb.overlaps(b):
                        cost += 40
                if (
                    bb.x0 < axes_box.x0
                    or bb.x1 > axes_box.x1
                    or bb.y0 < axes_box.y0
                    or bb.y1 > axes_box.y1
                ):
                    cost += 40
                if dist > 8:  # a leader will be drawn: keep it off other markers
                    lx = min(max(px, bb.x0), bb.x1)
                    ly = min(max(py, bb.y0), bb.y1)
                    for j, b in enumerate(marks):
                        if j != k and _segment_hits_box(px, py, lx, ly, b):
                            cost += 6
                cost *= wk
                if best is None or cost < best[0]:
                    best = (cost, (ox, oy, ha, va), bb)
        cost, (ox, oy, ha, va), bb = best
        ax.annotate(
            text,
            (x, y),
            xytext=(ox / px_per_pt, oy / px_per_pt),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fs,
            color=INK,
            zorder=10,
            path_effects=_halo(),
        )
        if (ox * ox + oy * oy) ** 0.5 > 8 * px_per_pt:
            lx = min(max(px, bb.x0), bb.x1)
            ly = min(max(py, bb.y0), bb.y1)
            inv = tr.inverted()
            (lxd, lyd) = inv.transform((lx, ly))
            ax.plot([x, lxd], [y, lyd], color="#9a9a9a", lw=0.5, zorder=1, clip_on=False)
        placed.append(bb)


def _segment_hits_box(x0, y0, x1, y1, b) -> bool:
    """Whether the segment (x0, y0)-(x1, y1) passes through the display box ``b``."""
    for t in np.linspace(0.08, 0.92, 12):
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        if b.x0 <= x <= b.x1 and b.y0 <= y <= b.y1:
            return True
    return False


def _point_style(s: str, d: pd.DataFrame) -> dict:
    """Marker style of one source on the sensitivity map (fake agents grey, humans black)."""
    if s.startswith("fake"):
        return {"marker": "o", "color": "#9a9a9a", "mfc": "#9a9a9a", "ms": 3.4}
    if s == "human":
        return {"marker": "s", "color": "#000000", "mfc": "#000000", "ms": 6.5}
    st = spec(s)
    hollow = st["mode"] != "main" or st["cond"] == "reasoning" or bool(st["partial"])
    return {
        "marker": st["marker"] or "o",
        "color": st["color"],
        "mfc": "white" if hollow else st["color"],
        "ms": 5.2,
    }


def fig_sensitivity_map(
    summ: pd.DataFrame,
    out_stem: Path,
    sources: Sequence[str],
    fake_anchors: Sequence[str] = ("fake-fixedstep", "fake-switchprob"),
    all_fakes: bool = False,
    height: float = 4.8,
    inset: tuple[float, float, float, float] | None = None,
    numbered: bool | None = None,
) -> tuple[Path, Path]:
    """Scatter of fitted sensitivities: s_q (x) against s_d (y) with bootstrap intervals; Bayes at
    (1, 1), full neglect at (0, 0), humans as a marked point, fake agents as small grey markers.
    ``summ`` has source, s_q, s_q_lo, s_q_hi, s_d, s_d_lo, s_d_hi. Points are labelled directly
    (ink text, leaders where a label had to move) or, when there are many (``numbered``, by
    default above 18 labelled points), by a number next to the marker with a key under the plot."""
    d = summ.set_index("source")
    fakes = (
        [s for s in d.index if s.startswith("fake")]
        if all_fakes
        else [s for s in d.index if any(s.startswith(a) for a in fake_anchors)]
    )
    models = order(
        [x for x in sources if x in d.index and x != "human" and not x.startswith("fake")]
    )
    labelled = ([] if "human" not in d.index else ["human"]) + models + list(fakes)
    if numbered is None:
        numbered = len(labelled) > 18
    key_rows = [
        (s, FAKE_SHORT.get(s.split(" [")[0], s) if s.startswith("fake") else spec(s)["label"])
        for s in labelled
    ]
    numbers = {s: k + 1 for k, (s, _) in enumerate(key_rows)} if numbered else {}
    key_row_in = 0.135
    # the key sits in a column to the right of the map, one entry per row
    key_w_in = 2.05 if numbered else 0.0
    key_in = 0.0
    lab_pt = FONT_PT - 1

    def label_of(s: str) -> str:
        if numbered:
            return str(numbers[s])
        return FAKE_SHORT.get(s.split(" [")[0], s) if s.startswith("fake") else spec(s)["label"]

    with plt.rc_context(rc()):
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, height))
        ax.axhline(1, color="#e2e2e2", lw=0.6)
        ax.axvline(1, color="#e2e2e2", lw=0.6)
        ax.plot([1], [1], marker="*", color="#7a7a7a", markersize=10, linestyle="none", zorder=3)
        ax.plot(
            [0],
            [0],
            marker="x",
            color="#7a7a7a",
            markersize=7,
            markeredgewidth=1.2,
            linestyle="none",
            zorder=3,
        )
        items = [
            (1.0, 1.0, "Bayes (1, 1)", INK, lab_pt),
            (0.0, 0.0, "full neglect (0, 0)", INK, lab_pt),
        ]
        weights = [1.0, 1.0]
        obstacles = []
        for s in labelled:
            r = d.loc[s]
            ps = _point_style(s, d)
            if s.startswith("fake"):
                ax.plot(
                    r.s_q,
                    r.s_d,
                    marker=ps["marker"],
                    color=ps["color"],
                    markersize=ps["ms"],
                    linestyle="none",
                    zorder=2,
                )
                fs = lab_pt
            else:
                ax.errorbar(
                    r.s_q,
                    r.s_d,
                    xerr=[[r.s_q - r.s_q_lo], [r.s_q_hi - r.s_q]],
                    yerr=[[r.s_d - r.s_d_lo], [r.s_d_hi - r.s_d]],
                    fmt=ps["marker"],
                    color=ps["color"],
                    markerfacecolor=ps["mfc"],
                    markeredgecolor=ps["color"],
                    markersize=ps["ms"],
                    capsize=2 if s == "human" else 1.5,
                    elinewidth=1.1 if s == "human" else 0.7,
                    zorder=5 if s == "human" else 4,
                )
                obstacles.append((r.s_q_lo, r.s_q_hi, r.s_d_lo, r.s_d_hi))
                fs = FONT_PT if s == "human" and not numbered else lab_pt
            text = "Humans" if (s == "human" and not numbered) else label_of(s)
            items.append((float(r.s_q), float(r.s_d), text, INK, fs))
            weights.append(1.0)
        xs = [x for x, *_ in items]
        ys = [y for _, y, *_ in items]
        ax.set_xlim(min(-0.3, min(xs) - 0.05), max(1.2, max(xs) + 0.3))
        ax.set_ylim(min(-0.3, min(ys) - 0.08), max(1.45, max(ys) + 0.1))
        ax.set_xlabel(r"$s_q$: sensitivity to the stated switch probability")
        ax.set_ylabel(r"$s_d$: sensitivity to the stated diagnosticity")
        bottom_in = 0.45 + key_in
        fig.subplots_adjust(
            left=0.09,
            right=1 - (key_w_in + 0.05) / TEXT_WIDTH_IN if numbered else 0.98,
            top=0.97,
            bottom=bottom_in / height,
        )
        boxes = []
        if inset is not None:
            x0, x1, y0, y1 = inset
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    lw=0.6,
                    color="#999999",
                    ls=(0, (2, 2)),
                    zorder=1,
                )
            )
            # a free band above the data for the detail panel
            ylo_, yhi_ = ax.get_ylim()
            ax.set_ylim(ylo_, yhi_ + 0.75 * (yhi_ - ylo_) * 0.5)
            axin = ax.inset_axes([0.30, 0.67, 0.44, 0.31])
            axin.set_xlim(x0, x1)
            axin.set_ylim(y0, y1)
            axin.tick_params(labelsize=FONT_PT - 2, length=2)
            axin.grid(True, color=GRID, lw=0.4)
            for sp_ in axin.spines.values():
                sp_.set_color("#999999")
            in_items = []
            in_obstacles = []
            for s in labelled:
                r = d.loc[s]
                if not (x0 <= r.s_q <= x1 and y0 <= r.s_d <= y1):
                    continue
                ps = _point_style(s, d)
                if s.startswith("fake"):
                    axin.plot(
                        r.s_q,
                        r.s_d,
                        marker=ps["marker"],
                        color=ps["color"],
                        markersize=ps["ms"],
                        linestyle="none",
                        zorder=2,
                    )
                else:
                    axin.errorbar(
                        r.s_q,
                        r.s_d,
                        xerr=[[r.s_q - r.s_q_lo], [r.s_q_hi - r.s_q]],
                        yerr=[[r.s_d - r.s_d_lo], [r.s_d_hi - r.s_d]],
                        fmt=ps["marker"],
                        color=ps["color"],
                        markerfacecolor=ps["mfc"],
                        markeredgecolor=ps["color"],
                        markersize=ps["ms"] - 0.2,
                        capsize=1.2,
                        elinewidth=0.6,
                        zorder=5 if s == "human" else 4,
                    )
                    in_obstacles.append((r.s_q_lo, r.s_q_hi, r.s_d_lo, r.s_d_hi))
                text = "Humans" if (s == "human" and not numbered) else label_of(s)
                if not numbered and "(s1-10)" in text:
                    continue  # the 10-subject stateless partial stays unlabelled in the detail
                in_items.append((float(r.s_q), float(r.s_d), text, INK, FONT_PT - 1.2))
            neglect_in_inset = (x0 <= 0.0) and (x1 >= 0.0) and (y0 <= 0.0) and (y1 >= 0.0)
            if neglect_in_inset:
                axin.plot(
                    [0],
                    [0],
                    marker="x",
                    color="#7a7a7a",
                    markersize=7,
                    markeredgewidth=1.2,
                    linestyle="none",
                    zorder=3,
                )
                if not numbered:  # the crowded numbered detail labels neglect in the main panel
                    in_items.append((0.0, 0.0, "full neglect (0, 0)", INK, FONT_PT - 1.2))
            axin.set_title("detail", fontsize=FONT_PT - 1.5, loc="left", pad=2, color=INK)
            fig.canvas.draw()
            place_labels_2d(axin, in_items, in_obstacles)
            keep = [
                k
                for k, it in enumerate(items)
                if not (x0 <= it[0] <= x1 and y0 <= it[1] <= y1)
                or (numbered and it[2].startswith("full neglect"))
            ]
            items = [items[k] for k in keep]
            weights = [weights[k] for k in keep]
            # the inset (with its tick labels and title) is a no-go area for main-panel labels
            fig.canvas.draw()
            ib = axin.get_tightbbox(fig.canvas.get_renderer())
            boxes.append(ib)
        place_labels_2d(ax, items, obstacles, boxes, weights)
        if numbered:
            # key beside the plot: number, marker swatch, name; one entry per row
            W = TEXT_WIDTH_IN
            for k, (s, name) in enumerate(key_rows):
                xk = (W - key_w_in + 0.22) / W
                yk = (height - 0.3 - k * key_row_in) / height
                ps = _point_style(s, d)
                fig.text(
                    xk, yk, f"{k + 1}", ha="right", va="center", fontsize=FONT_PT - 1, color=INK
                )
                fig.add_artist(
                    plt.Line2D(
                        [xk + 0.11 / W],
                        [yk],
                        transform=fig.transFigure,
                        marker=ps["marker"],
                        color=ps["color"],
                        markerfacecolor=ps["mfc"],
                        markeredgecolor=ps["color"],
                        markersize=ps["ms"] - 0.5,
                        linestyle="none",
                    )
                )
                fig.text(
                    xk + 0.2 / W, yk, name, ha="left", va="center", fontsize=FONT_PT - 1, color=INK
                )
        return _save(fig, out_stem)


FAKE_SHORT = {
    "fake-minority": "agent: minority share", "fake-switchprob": "agent: switch probability", "fake-random30": "agent: random around 30",
    "fake-countup": "agent: count-up", "fake-fixedstep": "agent: fixed step", "fake-wrongbayes": "agent: Bayes, wrong q and d",
    "fake-bayesnoise": "Bayes with the true q, d plus noise",
}  # fmt: skip


def fig_hist_3b_by_d(
    counts: pd.DataFrame, source: str, out_stem: Path, q: float = 0.05
) -> tuple[Path, Path]:
    """Raw-answer histogram of one source, one panel per d at a fixed q, with the printed minority
    share marked. ``counts`` has source, q, d, answer, n (the sanity raw-answer counts)."""
    sub = counts[(counts.source == source) & (counts.q == q)]
    st = spec(source)
    with plt.rc_context(rc()):
        fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 1.65), sharey=True)
        for ax, d in zip(axes, D_LEVELS, strict=True):
            g = sub[sub.d == d].groupby("answer").n.sum().reindex(range(0, 100), fill_value=0)
            share = g / max(1, g.sum())
            ax.bar(share.index, share.to_numpy(), width=1.0, color=st["color"], linewidth=0)
            printed = round(100 / (1 + d))
            ax.axvline(printed, color=INK, ls=(0, (1, 1.5)), lw=0.8)
            ax.set_title(
                f"$d$ = {d:g}: printed minority share {printed}",
                loc="left",
                fontsize=FONT_PT - 0.5,
                color=INK,
            )
            ax.set_xlim(-1, 100)
            ax.set_xlabel("stated probability, 0 to 99")
        axes[0].set_ylabel("share of answers")
        fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.22, wspace=0.12)
        return _save(fig, out_stem)


# ---------------------------------------------------------------- source labels for the tables
# (the builder's row labels and source order; the figure labels above use ``spec``)

FAMILY_ORDER = ("human", "qwen", "llama", "gemma", "phi", "granite", "claude", "mock", "other")
CLAUDE_TIERS = {"haiku": 0, "sonnet": 1, "opus": 2, "fable": 3, "mythos": 3}
MODE_ORDER = ("main", "stateless")
PLAIN_TAGS = {
    "human": "Humans",
    "human (all 99 trials)": "Humans (all 99 trials)",
    "Bayes": "Bayesian observer",
    "qwen2.5:3b": "Qwen2.5 3B",
    "qwen2.5:7b": "Qwen2.5 7B",
    "qwen2.5:14b": "Qwen2.5 14B",
    "qwen2.5:32b": "Qwen2.5 32B",
    "llama3.1:8b": "Llama 3.1 8B",
    "gemma2:9b": "Gemma 2 9B",
    "phi3.5:3.8b": "Phi 3.5 3.8B",
    "granite4.2:8b": "Granite 4.2 8B",
    "claude-haiku-5-5": "Haiku 5.5",
    "claude-sonnet-5-5": "Sonnet 5.5",
    "claude-opus-5-5": "Opus 5.5",
    "claude-haiku-4-5": "Haiku 4.5",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-sonnet-5": "Sonnet 5",
}


def parse_source(source: str) -> dict:
    """Split a source label into family, size/tier rank, model tag, mode and the remaining text.

    A label looks like ``'qwen2.5:7b [stateless, T0.7, PARTIAL s1-10]'``: the model tag,
    then in brackets the mode followed by free-form extras.
    """
    if source.startswith("human"):
        return {"family": "human", "rank": 0, "tag": "human", "mode": "main", "extra": ""}
    if source == "Bayes":
        return {"family": "bayes", "rank": 0, "tag": "Bayes", "mode": "main", "extra": ""}
    m = re.match(r"^(?P<tag>[^\[]+?)\s*(?:\[(?P<inside>[^\]]*)\])?$", source)
    tag = (m.group("tag") if m else source).strip()
    inside = (m.group("inside") or "") if m else ""
    parts = [p.strip() for p in inside.split(",")] if inside else []
    mode = parts[0] if parts else "main"
    extra = ", ".join(parts[1:])
    low = tag.lower()
    if low.startswith("qwen"):
        family = "qwen"
    elif low.startswith("llama"):
        family = "llama"
    elif low.startswith("gemma"):
        family = "gemma"
    elif low.startswith("phi"):
        family = "phi"
    elif low.startswith("claude"):
        family = "claude"
    elif low.startswith("granite"):
        family = "granite"
    elif low.startswith("mock"):
        family = "mock"
    else:
        family = "other"
    rank = 0.0
    if family == "claude":
        rank = next((v for k, v in CLAUDE_TIERS.items() if k in low), 0)
    else:
        size = re.search(r"(\d+(?:\.\d+)?)\s*b\b", low)
        rank = float(size.group(1)) if size else 0.0
    return {"family": family, "rank": rank, "tag": tag, "mode": mode, "extra": extra}


def plain_name(source: str) -> str:
    """Legend text: the plain model name, with the mode and extras in parentheses when the
    run is not the default multi-turn one (``'Qwen2.5 7B (stateless, subjects 1-10)'``)."""
    if source in PLAIN_TAGS:
        return PLAIN_TAGS[source]
    p = parse_source(source)
    name = PLAIN_TAGS.get(p["tag"], p["tag"])
    bits = []
    if p["mode"] != "main":
        bits.append(p["mode"])
    for part in [x.strip() for x in p["extra"].split(",") if x.strip()]:
        if re.fullmatch(r"T[\d.]+", part):
            continue
        if part == "greedy":
            bits.append("greedy")
        elif part.startswith("PARTIAL"):
            bits.append(part.replace("PARTIAL ", "subjects ").replace("s1-", "1-"))
        else:
            bits.append(part)
    return name + (" (" + ", ".join(bits) + ")" if bits else "")


def order_sources(sources: Sequence[str]) -> list[str]:
    """Humans first, then by family, size/tier, mode and the remaining label."""

    def key(s: str):
        p = parse_source(s)
        fam = FAMILY_ORDER.index(p["family"]) if p["family"] in FAMILY_ORDER else 99
        mode = MODE_ORDER.index(p["mode"]) if p["mode"] in MODE_ORDER else 9
        return (fam, p["rank"], p["tag"], mode, p["extra"], s)

    return sorted(dict.fromkeys(sources), key=key)

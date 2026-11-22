"""System-sensitivity pair (s_q, s_d) for every complete source, the humans, the rule-based agents
and the complete partial runs. No model calls.

Fitted version (from the per-subject system-neglect fits):
  s_q = OLS slope of log(alpha_i q_i) on log(q_i) over the three q levels (1 = uses the stated
        switch probability fully, 0 = ignores it);
  s_d = OLS slope of log(d_j^beta_j) = beta_j log d_j on log(d_j) over the three d levels.
Model-free version (no fit): per subject, m_q = mean stated probability over all periods and all
d at each q level and b_q = mean Bayesian posterior on the same periods; s_q(free) = OLS slope of
m_q on b_q over the three q levels (1 = the answer moves across q levels exactly as the posterior
does, 0 = flat); s_d(free) = OLS slope, across the three d levels, of the answer's signal response
(mean move after a blue ball minus after a red ball, periods 2-10) on the posterior's signal
response on the same sequences (1 = the response to a ball grows with d as the posterior's does,
0 = no change across d). A mean-answer slope is ill-conditioned for d because the mean posterior
barely differs between d levels. Means and percentile bootstrap 95% intervals over subjects
(2,000 draws, seed 0); paired differences from the human with the same subject index.

    .venv/bin/python scripts/explore_sensitivity.py

Reads the analysis CSVs that scripts/build_paper_numbers.py writes under paper/data/ and the raw
rows under results/; writes results/explore/sensitivity_per_subject.csv,
results/explore/sensitivity_summary.csv, results/explore/SENSITIVITY.md and a preview figure under
paper/figures/preview/. ``fitted_pairs``, ``free_pairs`` and ``summarise`` are imported by the
paper builder.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_shift import analysis as an  # noqa: E402
from regime_shift import paper_figures as pf  # noqa: E402
from regime_shift import pipeline  # noqa: E402
from regime_shift.results import load_results  # noqa: E402
from regime_shift.task import D_LEVELS, Q_LEVELS  # noqa: E402

OUT = ROOT / "results" / "explore"
Q = np.array(Q_LEVELS, dtype=float)
D = np.array(D_LEVELS, dtype=float)
LQ, LD = np.log(Q), np.log(D)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on x; NaN when either has a non-finite value or x is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return float("nan")
    xc = x - x.mean()
    den = float((xc**2).sum())
    return float((xc * (y - y.mean())).sum() / den) if den > 0 else float("nan")


def fitted_pairs(fits: pd.DataFrame) -> pd.DataFrame:
    """Per-subject (s_q, s_d) from a fits table with alpha_1..3 and beta_1..3 per row."""
    rows = []
    for _, r in fits.iterrows():
        a = np.array([r.alpha_1, r.alpha_2, r.alpha_3], dtype=float)
        b = np.array([r.beta_1, r.beta_2, r.beta_3], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sq = slope(LQ, np.log(a * Q)) if np.all(a > 0) else float("nan")
        sd = slope(LD, b * LD)
        rows.append({"source": r.source, "subject": int(r.subject), "s_q": sq, "s_d": sd})
    return pd.DataFrame(rows)


def free_pairs(long: pd.DataFrame) -> pd.DataFrame:
    """``long`` has source, subject, trial, period, signal, q, d, estimate, p_bayes.

    s_q(free): slope of the mean answer on the mean posterior across the three q levels.
    s_d(free): slope, across the three d levels, of the answer's signal response (mean move
    after a blue ball minus after a red ball, periods 2-10) on the posterior's signal response.
    """
    from regime_shift.task import BLUE, RED

    rows = []
    df = long.sort_values(["source", "subject", "trial", "period"]).copy()
    grp = df.groupby(["source", "subject", "trial"])
    df["move"] = df.estimate - grp.estimate.shift(1)
    df["bmove"] = df.p_bayes - grp.p_bayes.shift(1)
    for (src, subj), g in df.groupby(["source", "subject"]):
        gq = g.dropna(subset=["estimate", "p_bayes"])
        mq = gq.groupby("q").estimate.mean().reindex(Q_LEVELS)
        bq = gq.groupby("q").p_bayes.mean().reindex(Q_LEVELS)
        gm = g[g.period > 1].dropna(subset=["move", "bmove"])
        rd, bd = [], []
        for d in D_LEVELS:
            h = gm[gm.d == d]
            rd.append(h[h.signal == BLUE].move.mean() - h[h.signal == RED].move.mean())
            bd.append(h[h.signal == BLUE].bmove.mean() - h[h.signal == RED].bmove.mean())
        rows.append(
            {
                "source": src,
                "subject": int(subj),
                "s_q_free": slope(bq.to_numpy(), mq.to_numpy()),
                "s_d_free": slope(np.array(bd), np.array(rd)),
            }
        )
    return pd.DataFrame(rows)


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap 95% interval of the mean over the finite values of ``x``."""
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(per: pd.DataFrame, cols: list[str], human_label: str = "human") -> pd.DataFrame:
    """Per source: mean and bootstrap interval of every column in ``cols``, and for every
    non-human source the paired difference from the human with the same subject index."""
    rng = np.random.default_rng(0)
    hum = per[per.source == human_label].set_index("subject")
    out = []
    for src, g in per.groupby("source", sort=False):
        row = {"source": src, "n_subjects": g.subject.nunique()}
        g = g.set_index("subject")
        for c in cols:
            x = g[c].to_numpy(dtype=float)
            lo, hi = boot_ci(x, rng)
            row[c] = float(np.nanmean(x))
            row[f"{c}_lo"], row[f"{c}_hi"] = lo, hi
            if src != human_label:
                common = g.index.intersection(hum.index)
                diff = (g.loc[common, c] - hum.loc[common, c]).to_numpy(dtype=float)
                dlo, dhi = boot_ci(diff, rng)
                row[f"{c}_diff"] = float(np.nanmean(diff)) if len(diff) else float("nan")
                row[f"{c}_diff_lo"], row[f"{c}_diff_hi"] = dlo, dhi
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    human = pipeline.prepare_human("data/human", "behavioral")
    hs = human["human_sub"].copy()
    hs["source"] = "human"

    # ---- fits: main (complete 30-subject sources + humans), partial (complete files), agents
    fits_main = pd.read_csv(ROOT / "paper/data/main/analysis/fits.csv")
    fits_part = pd.read_csv(ROOT / "paper/data/partial/analysis/partial_fits.csv")
    fits_fake = pd.read_csv(ROOT / "results/fake/analysis/fake_fits.csv")
    acc = pd.read_csv(ROOT / "paper/data/main/analysis/accuracy_overall.csv")
    complete = set(acc[acc.n_subjects == 30].source) - {"human"}
    main_fits = fits_main[fits_main.source.isin(complete | {"human"})]
    # complete partial files only (rows == subjects x 720, or the stopped reasoning run)
    part_rows = load_results(ROOT / "results/partial")
    counts = part_rows.groupby("source").size()
    keep_part = []
    for src, n in counts.items():
        m = re.search(r"s1-(\d+)", src)
        n_sub = int(m.group(1)) if m else 30
        # the protocol variants only: not the multi-turn subset views or the pilot
        wanted = ("stateless" in src) or ("reasoning allowed" in src)
        if wanted and (n >= n_sub * 720 or "reasoning allowed, PARTIAL s1-3" in src):
            keep_part.append(src)
    part_fits = fits_part[
        fits_part.source.isin(keep_part) & ~fits_part.source.str.startswith("human")
    ]
    fake_fits = fits_fake[~fits_fake.source.str.startswith("human")]
    fitted = pd.concat(
        [fitted_pairs(main_fits), fitted_pairs(part_fits), fitted_pairs(fake_fits)],
        ignore_index=True,
    )

    # ---- model-free: rows with the Bayesian posterior
    main_rows = an.add_bayes(load_results(ROOT / "results"))
    main_rows = main_rows[main_rows.source.isin(complete)]
    part_rows = an.add_bayes(part_rows[part_rows.source.isin(keep_part)])
    fake_rows = an.add_bayes(load_results(ROOT / "results/fake"))
    cols = ["source", "subject", "trial", "period", "signal", "q", "d", "estimate", "p_bayes"]
    free = free_pairs(
        pd.concat([hs[cols], main_rows[cols], part_rows[cols], fake_rows[cols]], ignore_index=True)
    )

    per = fitted.merge(free, on=["source", "subject"], how="outer")
    per.to_csv(OUT / "sensitivity_per_subject.csv", index=False)
    summ = summarise(per, ["s_q", "s_d", "s_q_free", "s_d_free"])
    order = (
        ["human"]
        + pf.order([s for s in summ.source if s != "human" and not s.startswith("fake")])
        + sorted(s for s in summ.source if s.startswith("fake"))
    )
    summ = summ.set_index("source").loc[order].reset_index()
    summ.to_csv(OUT / "sensitivity_summary.csv", index=False)

    # ---- markdown table
    def f3(v):
        return "nan" if not np.isfinite(v) else f"{v:.2f}"

    lines = [
        "| source | n | s_q (fit) [95% CI] | diff vs humans | s_d (fit) [95% CI] | diff vs humans "
        "| s_q (model-free) | s_d (model-free) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summ.iterrows():
        lab = pf.spec(r.source)["label"] if not r.source.startswith("fake") else r.source
        dq = (
            ""
            if r.source == "human"
            else f"{r.s_q_diff:+.2f} [{r.s_q_diff_lo:+.2f}, {r.s_q_diff_hi:+.2f}]"
        )
        dd = (
            ""
            if r.source == "human"
            else f"{r.s_d_diff:+.2f} [{r.s_d_diff_lo:+.2f}, {r.s_d_diff_hi:+.2f}]"
        )
        lines.append(
            f"| {lab} | {int(r.n_subjects)} "
            f"| {f3(r.s_q)} [{f3(r.s_q_lo)}, {f3(r.s_q_hi)}] | {dq} "
            f"| {f3(r.s_d)} [{f3(r.s_d_lo)}, {f3(r.s_d_hi)}] | {dd} "
            f"| {f3(r.s_q_free)} [{f3(r.s_q_free_lo)}, {f3(r.s_q_free_hi)}] "
            f"| {f3(r.s_d_free)} [{f3(r.s_d_free_lo)}, {f3(r.s_d_free_hi)}] |"
        )
    md = "\n".join(lines)
    (OUT / "SENSITIVITY.md").write_text(
        "# System-sensitivity pair\n\n" + __doc__ + "\n\n" + md + "\n"
    )
    print(md)

    # ---- figure
    import matplotlib.pyplot as plt

    with plt.rc_context(pf.rc()):
        fig, ax = plt.subplots(figsize=(pf.TEXT_WIDTH_IN, 4.6))
        ax.axhline(1, color="#dddddd", lw=0.6)
        ax.axvline(1, color="#dddddd", lw=0.6)
        ax.plot([1], [1], marker="*", color="#7a7a7a", markersize=9, linestyle="none", zorder=3)
        ax.annotate(
            "Bayes (1, 1)",
            (1, 1),
            xytext=(5, 4),
            textcoords="offset points",
            color="#7a7a7a",
            fontsize=pf.FONT_PT - 1,
        )
        ax.plot([0], [0], marker="x", color="#7a7a7a", markersize=6, linestyle="none", zorder=3)
        ax.annotate(
            "full neglect (0, 0)",
            (0, 0),
            xytext=(5, -9),
            textcoords="offset points",
            color="#7a7a7a",
            fontsize=pf.FONT_PT - 1,
        )
        labels = []
        for _, r in summ.iterrows():
            if not (np.isfinite(r.s_q) and np.isfinite(r.s_d)):
                continue
            if r.source.startswith("fake"):
                ax.plot(
                    r.s_q,
                    r.s_d,
                    marker="o",
                    color="#9a9a9a",
                    markersize=3,
                    linestyle="none",
                    zorder=2,
                )
                labels.append(
                    (
                        r.s_q,
                        r.s_d,
                        "fake: " + r.source.split(" [")[0].replace("fake-", ""),
                        "#8a8a8a",
                        pf.FONT_PT - 2.5,
                    )
                )
                continue
            st = pf.spec(r.source)
            if r.source == "human":
                ax.errorbar(
                    r.s_q,
                    r.s_d,
                    xerr=[[r.s_q - r.s_q_lo], [r.s_q_hi - r.s_q]],
                    yerr=[[r.s_d - r.s_d_lo], [r.s_d_hi - r.s_d]],
                    fmt="s",
                    color="#000000",
                    markersize=6,
                    capsize=2,
                    elinewidth=1.0,
                    zorder=5,
                )
                labels.append((r.s_q, r.s_d, "Humans", "#000000", pf.FONT_PT))
                continue
            hollow = st["mode"] != "main" or st["cond"] == "reasoning" or bool(st["partial"])
            ax.errorbar(
                r.s_q,
                r.s_d,
                xerr=[[r.s_q - r.s_q_lo], [r.s_q_hi - r.s_q]],
                yerr=[[r.s_d - r.s_d_lo], [r.s_d_hi - r.s_d]],
                fmt=st["marker"] or "o",
                color=st["color"],
                markerfacecolor="white" if hollow else st["color"],
                markeredgecolor=st["color"],
                markersize=4.5,
                capsize=1.5,
                elinewidth=0.6,
                zorder=4,
            )
            labels.append((r.s_q, r.s_d, st["label"], pf.text_color(st["color"]), pf.FONT_PT - 1.5))
        ax.set_xlim(-0.3, 1.22)
        ax.set_ylim(-0.3, 1.45)
        # place labels, then push overlapping label boxes apart vertically (display space)
        fig.canvas.draw()
        arts = []
        for x, y, text, colour, fs in labels:
            arts.append(
                ax.annotate(
                    text,
                    (x, y),
                    xytext=(5, 0),
                    textcoords="offset points",
                    color=colour,
                    fontsize=fs,
                    va="center",
                    zorder=10,
                )
            )
        for _ in range(60):
            fig.canvas.draw()
            boxes = [a.get_window_extent() for a in arts]
            moved = False
            for a_i in range(len(arts)):
                for b_i in range(a_i + 1, len(arts)):
                    ba, bb = boxes[a_i], boxes[b_i]
                    if ba.overlaps(bb):
                        dy = min(ba.y1 - bb.y0, bb.y1 - ba.y0) / 2 + 1
                        lo_i, hi_i = (a_i, b_i) if ba.y0 <= bb.y0 else (b_i, a_i)
                        for idx, sign in ((lo_i, -1), (hi_i, 1)):
                            ox, oy = arts[idx].xyann
                            arts[idx].xyann = (ox, oy + sign * dy * 72 / fig.dpi)
                        moved = True
            if not moved:
                break
        for a in arts:  # leader from point to label when the label was moved
            ox, oy = a.xyann
            if abs(oy) > 4:
                ax.annotate(
                    "",
                    xy=a.xy,
                    xytext=(ox - 1, oy),
                    textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "color": "#bbbbbb", "lw": 0.4},
                    zorder=1,
                )
        ax.set_xlabel(
            r"$s_q$: sensitivity to the stated switch probability "
            r"(slope of $\log \alpha q$ on $\log q$)"
        )
        ax.set_ylabel(
            r"$s_d$: sensitivity to the stated diagnosticity (slope of $\beta \log d$ on $\log d$)"
        )
        fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.12)
        preview = ROOT / "paper/figures/preview"
        preview.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(preview / f"fig_sensitivity.{ext}", dpi=220)
        plt.close(fig)
    print("\nfigure: paper/figures/preview/fig_sensitivity.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

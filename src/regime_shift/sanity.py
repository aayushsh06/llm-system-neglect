"""Model-free checks of the raw answers.

Applied with identical code to humans, to every model run and to the Bayesian
posterior (as a reference "source"):

- effective parameters alpha*q and d**beta per level and per condition;
- mean change in the reported probability after a blue versus a red ball, by d;
- mean reported probability by period in each of the nine conditions;
- a descriptive regression of the answer on period, blue count so far and the
  previous answer (how much variance each explains);
- raw-answer histograms, a concentration flag (few values making up most answers)
  and the share of answers that copy a number printed in the prompt;
- calibration against the Bayesian posterior and accuracy by posterior state;
- a multi-turn versus stateless comparison for the same model and temperature;
- rule-based one-word verdicts per source.

Everything is computed per subject first and averaged with the SEM across subjects.
:func:`run_sanity_checks` runs every check and writes one CSV per table.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .analysis import TRIAL_KEY, _sem, add_bayes  # noqa: E402
from .fitting import PARAM_NAMES, fit_all_subjects  # noqa: E402
from .human_data import HUMAN_SOURCE  # noqa: E402
from .task import BLUE, D_LEVELS, N_PERIODS, Q_LEVELS, RED  # noqa: E402

BAYES_SOURCE = "Bayes"
BLUE_C, RED_C = "#2a78d6", "#e34948"  # bar colours for the two ball colours


def bayes_as_source(human_sub: pd.DataFrame) -> pd.DataFrame:
    """The Bayesian posterior on the human sequences, as a reference source."""
    b = human_sub.copy()
    b["source"] = BAYES_SOURCE
    b["estimate"] = b["p_bayes"]
    return b


# ---------------------------------------------------------------- effective parameters


def effective_parameters(fits: pd.DataFrame) -> pd.DataFrame:
    """Per level: effective switch rate alpha*q and effective diagnosticity d**beta.

    Per subject first, then mean and SEM across subjects. ``bayes`` is the value a
    Bayesian would have (q itself, d itself).
    """
    rows = []
    ok = fits[fits.success.astype(bool)]
    for src, g in ok.groupby("source", sort=True):
        for k, q in enumerate(Q_LEVELS):
            eff = g[PARAM_NAMES[k]] * q
            rows.append(
                {
                    "source": src,
                    "quantity": "effective switch rate alpha*q",
                    "level": q,
                    "bayes": q,
                    "mean": eff.mean(),
                    "sem": _sem(eff),
                    "n": len(eff),
                }
            )
        for j, d in enumerate(D_LEVELS):
            eff = d ** g[PARAM_NAMES[3 + j]]
            rows.append(
                {
                    "source": src,
                    "quantity": "effective diagnosticity d^beta",
                    "level": d,
                    "bayes": d,
                    "mean": eff.mean(),
                    "sem": _sem(eff),
                    "n": len(eff),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- signal response


def _with_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add the previous answer, the change from it, the blue count so far and the ball colour."""
    d = df.sort_values([*TRIAL_KEY, "period"]).copy()
    d["prev_estimate"] = d.groupby(TRIAL_KEY).estimate.shift(1)
    d["delta"] = d.estimate - d.prev_estimate
    d["n_blue_so_far"] = (d.signal == BLUE).groupby([d[c] for c in TRIAL_KEY]).cumsum()
    d["colour"] = np.where(d.signal == RED, "red", "blue")
    return d


def signal_response(df: pd.DataFrame) -> pd.DataFrame:
    """Mean change in the reported probability after a blue vs a red ball, by d and overall.

    One row per source, d level (or "all"), colour; plus a ``red_minus_blue`` row per
    source and d with a one-sample t-test across subjects (does the answer move in
    opposite directions for the two colours?).
    """
    d = _with_deltas(df)
    d = d[(d.period > 1)].dropna(subset=["delta"])
    rows = []
    for d_level, g in [*d.groupby("d"), ("all", d)]:
        per = g.groupby(["source", "subject", "colour"]).delta.mean().unstack("colour")
        for src, gs in per.groupby(level="source"):
            for colour in ("blue", "red"):
                if colour in gs:
                    rows.append(
                        {
                            "source": src,
                            "d": d_level,
                            "after": colour,
                            "mean_delta": gs[colour].mean(),
                            "sem": _sem(gs[colour]),
                            "n_subjects": int(gs[colour].notna().sum()),
                            "t": np.nan,
                            "p": np.nan,
                        }
                    )
            if "red" in gs and "blue" in gs:
                sep = (gs["red"] - gs["blue"]).dropna()
                t, p = stats.ttest_1samp(sep, 0.0) if len(sep) > 2 else (np.nan, np.nan)
                rows.append(
                    {
                        "source": src,
                        "d": d_level,
                        "after": "red_minus_blue",
                        "mean_delta": sep.mean(),
                        "sem": _sem(sep),
                        "n_subjects": len(sep),
                        "t": float(t),
                        "p": float(p),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- by-period and marginal means


def period_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean reported probability by source, q, d and period (per subject first)."""
    per = df.groupby(["source", "subject", "q", "d", "period"]).estimate.mean().reset_index()
    return (
        per.groupby(["source", "q", "d", "period"])
        .estimate.agg(mean="mean", sem=_sem, n_subjects="count")
        .reset_index()
    )


def marginal_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean reported probability by q and by d (all periods), per source: does the answer
    move with the system parameters at all? Includes a one-way ANOVA across subjects."""
    rows = []
    for factor in ("q", "d"):
        per = df.groupby(["source", "subject", factor]).estimate.mean().reset_index()
        for src, g in per.groupby("source"):
            groups = [x.estimate.dropna().to_numpy() for _, x in g.groupby(factor)]
            f, p = stats.f_oneway(*groups) if all(len(x) > 1 for x in groups) else (np.nan, np.nan)
            for lvl, x in g.groupby(factor):
                rows.append(
                    {
                        "source": src,
                        "factor": factor,
                        "level": lvl,
                        "mean": x.estimate.mean(),
                        "sem": _sem(x.estimate),
                        "anova_F": float(f),
                        "anova_p": float(p),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- regression


# ---------------------------------------------------------------- raw answers


def raw_answer_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of the integer answer (0..99) per source and condition ("all" included)."""
    d = df.dropna(subset=["estimate"]).copy()
    d["answer"] = np.rint(d.estimate * 100).astype(int)
    parts = [
        d.groupby(["source", "answer"]).size().reset_index(name="count").assign(condition="all")
    ]
    parts.append(d.groupby(["source", "condition", "answer"]).size().reset_index(name="count"))
    out = pd.concat(parts, ignore_index=True)
    out["share"] = out["count"] / out.groupby(["source", "condition"])["count"].transform("sum")
    return out[["source", "condition", "answer", "count", "share"]]


# ---------------------------------------------------------------- prompt-number anchoring


def ratio_anchor_share(df: pd.DataFrame, tolerance: int = 2) -> pd.DataFrame:
    """Share of answers equal to a number printed in the prompt, per source and condition.

    Two numbers appear in every prompt: the minority colour proportion (d = 1.5 -> 40,
    d = 3 -> 25, d = 9 -> 10) and the switch probability (q = 0.01 -> 1, 0.05 -> 5,
    0.10 -> 10). In the q = 0.10, d = 9 cell both are 10, so that cell is flagged
    ``ambiguous`` and its share is reported under ``both_numbers``. Rows for all
    periods and for period 1 alone. Hits are counted only on answers where the printed
    number differs from the rounded Bayesian answer by more than ``tolerance`` points
    (default 2; ``detectable_*`` = share of such answers), so Bayes scores 0 by construction
    and the human row is the behavioural baseline (``excess_*`` columns).
    A model that copies a printed number looks "d-sensitive" (or "q-sensitive") without
    weighing any ball.
    """
    d = df.dropna(subset=["estimate"]).copy()
    if "p_bayes" not in d:
        d = add_bayes(d)
    d["answer"] = np.rint(d.estimate * 100).astype(int)
    d["bayes_answer"] = np.rint(d.p_bayes * 100).astype(int)
    d["minority"] = np.rint(100 / (1 + d.d)).astype(int)
    d["switch"] = np.rint(d.q * 100).astype(int)
    amb = d.minority == d.switch
    # A hit counts only where the printed number differs from the rounded Bayesian answer by
    # more than ``tolerance`` points; otherwise a copy and a correct answer are the same digits
    # (e.g. "1" after any first ball at q = 0.01, d = 1.5).
    det_min = (d.bayes_answer - d.minority).abs() > tolerance
    det_sw = (d.bayes_answer - d.switch).abs() > tolerance
    d["eq_minority"] = ((d.answer == d.minority) & ~amb).where(det_min & ~amb)
    d["eq_switch"] = ((d.answer == d.switch) & ~amb).where(det_sw & ~amb)
    d["eq_both"] = ((d.answer == d.minority) & amb).where(det_min & amb)
    d["eq_majority"] = d.answer == 100 - d.minority
    d["eq_neither"] = ~(
        d.eq_minority.fillna(False) | d.eq_switch.fillna(False) | d.eq_both.fillna(False)
    )
    d["detectable_ratio_echo"] = det_min & ~amb
    d["detectable_prior_anchoring"] = det_sw & ~amb
    cols = ["eq_minority", "eq_switch", "eq_both", "eq_majority", "eq_neither"]
    dcols = ["detectable_ratio_echo", "detectable_prior_anchoring"]
    rows = []
    for label, part in (("all periods", d), ("period 1", d[d.period == 1])):
        per = part.groupby(["source", "subject", "condition", "q", "d"])[cols + dcols].mean()
        for (src, cond, q, d_level), g in per.groupby(["source", "condition", "q", "d"]):
            minority = int(round(100 / (1 + d_level)))
            switch = int(round(q * 100))
            rows.append(
                {
                    "source": src,
                    "periods": label,
                    "condition": int(cond),
                    "q": q,
                    "d": d_level,
                    "ratio_number": minority,
                    "prior_number": switch,
                    "ambiguous": minority == switch,
                    "ratio_echo": g.eq_minority.mean(),
                    "ratio_echo_sem": _sem(g.eq_minority),
                    "prior_anchoring": g.eq_switch.mean(),
                    "prior_anchoring_sem": _sem(g.eq_switch),
                    "both_numbers": g.eq_both.mean(),
                    "majority_number_share": g.eq_majority.mean(),
                    "neither": g.eq_neither.mean(),
                    "detectable_ratio_echo": g.detectable_ratio_echo.mean(),
                    "detectable_prior_anchoring": g.detectable_prior_anchoring.mean(),
                    "n_subjects": len(g),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) and (out.source == HUMAN_SOURCE).any():
        # humans are the reference: copying = more hits on a printed number than the human
        # subjects produce in the same cell (a correct period-1 answer is itself close to q,
        # so Bayes is listed in the table as a second baseline but not subtracted)
        cols = ["ratio_echo", "prior_anchoring", "both_numbers"]
        base = out[out.source == HUMAN_SOURCE][["periods", "condition", *cols]].rename(
            columns={c: "human_" + c for c in cols}
        )
        out = out.merge(base, on=["periods", "condition"], how="left")
        for c in cols:
            out["excess_" + c] = out[c] - out["human_" + c]
    return out


# ---------------------------------------------------------------- copying, calibration, state


def copying_tolerance_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Period-1 copying shares under two detectability rules, side by side, per source:
    "old rule" counts an answer when the printed number differs from the rounded Bayesian
    answer at all (tolerance 0), "new rule" when it differs by more than 2 points.
    Two behaviours: *ratio echo* (the answer is the printed colour-ratio number)
    and *prior anchoring* (the answer is the printed switch probability). For each: the
    maximum and the mean over the eight unambiguous cells, plus the q = 0.10, d = 9 cell
    where both printed numbers are 10."""
    rows = []
    tables = {
        "old rule": ratio_anchor_share(df, tolerance=0),
        "new rule": ratio_anchor_share(df, tolerance=2),
    }
    for src in df.source.unique():
        r = {"source": src}
        for rule, a in tables.items():
            p1 = a[(a.source == src) & (a.periods == "period 1")]
            una, amb = p1[~p1.ambiguous.astype(bool)], p1[p1.ambiguous.astype(bool)]
            r[f"ratio echo max | {rule}"] = una.ratio_echo.max()
            r[f"ratio echo mean | {rule}"] = una.ratio_echo.mean()
            r[f"prior anchoring max | {rule}"] = una.prior_anchoring.max()
            r[f"prior anchoring mean | {rule}"] = una.prior_anchoring.mean()
            r[f"both numbers q=.10,d=9 | {rule}"] = amb.both_numbers.iloc[0] if len(amb) else np.nan
            r[f"detectable prior anchoring mean | {rule}"] = una.detectable_prior_anchoring.mean()
        rows.append(r)
    return pd.DataFrame(rows)


CALIBRATION_BINS = (0.0, 0.05, 0.2, 0.5, 0.8, 1.0)


def calibration(df: pd.DataFrame, bins: Sequence[float] = CALIBRATION_BINS) -> pd.DataFrame:
    """Mean stated probability per bin of the Bayesian posterior, per source
    (every (sequence, period) is one observation; bins are [lo, hi), the last one closed)."""
    d = df.dropna(subset=["estimate"]).copy()
    edges = list(bins)
    idx = np.clip(np.searchsorted(edges, d.p_bayes.to_numpy(), side="right") - 1, 0, len(edges) - 2)
    d["bin"] = [f"[{edges[k]:g}, {edges[k + 1]:g})" for k in idx]
    out = (
        d.groupby(["source", "bin"])
        .agg(
            n=("estimate", "size"),
            mean_bayes=("p_bayes", "mean"),
            mean_stated=("estimate", "mean"),
            sem_stated=("estimate", _sem),
        )
        .reset_index()
    )
    out["gap"] = out.mean_stated - out.mean_bayes
    return out.sort_values(["source", "mean_bayes"]).reset_index(drop=True)


def accuracy_by_posterior_state(
    df: pd.DataFrame, threshold: float = 0.2
) -> dict[str, pd.DataFrame]:
    """Mean |answer - Bayes| separately for periods where the Bayesian posterior is below
    ``threshold`` (nothing has visibly happened yet) and at or above it, with the share of
    periods in each group; per condition and overall per source, plus the decomposition of the
    overall distance into the two groups' contributions."""
    d = df.dropna(subset=["estimate"]).copy()
    d["dev"] = (d.estimate - d.p_bayes).abs()
    d["state"] = np.where(
        d.p_bayes < threshold, f"posterior < {threshold:g}", f"posterior >= {threshold:g}"
    )
    per_cond = (
        d.groupby(["source", "condition", "q", "d", "state"])
        .agg(n=("dev", "size"), mean_abs_dev=("dev", "mean"))
        .reset_index()
    )
    totals = d.groupby(["source", "condition"]).size().rename("n_cond").reset_index()
    per_cond = per_cond.merge(totals, on=["source", "condition"])
    per_cond["share_of_periods"] = per_cond.n / per_cond.n_cond
    overall = (
        d.groupby(["source", "state"])
        .agg(n=("dev", "size"), mean_abs_dev=("dev", "mean"))
        .reset_index()
    )
    tot = d.groupby("source").size().rename("n_all").reset_index()
    overall = overall.merge(tot, on="source")
    overall["share_of_periods"] = overall.n / overall.n_all
    overall["contribution_to_overall"] = overall.share_of_periods * overall.mean_abs_dev
    return {"per_condition": per_cond, "overall": overall}


# ---------------------------------------------------------------- last-period jump


def last_period_jump(df: pd.DataFrame) -> pd.DataFrame:
    """Mean step at period 10 minus the mean step over periods 2 to 9, per subject then
    averaged, with a t-test across subjects. A large positive value is an end-of-trial
    effect ("Period 10 of 10") rather than a response to the last ball."""
    d = _with_deltas(df)
    d = d[d.period > 1].dropna(subset=["delta"])
    d["last"] = d.period == N_PERIODS
    per = d.groupby(["source", "subject", "last"]).delta.mean().unstack("last")
    rows = []
    for src, g in per.groupby(level="source"):
        if True not in g or False not in g:
            continue
        jump = (g[True] - g[False]).dropna()
        t, p = stats.ttest_1samp(jump, 0.0) if len(jump) > 2 else (np.nan, np.nan)
        rows.append(
            {
                "source": src,
                "step_period_10": g[True].mean(),
                "step_periods_2_to_9": g[False].mean(),
                "jump": jump.mean(),
                "sem": _sem(jump),
                "t": float(t),
                "p": float(p),
                "n_subjects": len(jump),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- mode comparison


# ---------------------------------------------------------------- one-sentence verdicts


def verdict_tables(human_sub: pd.DataFrame, long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The four tables :func:`verdict_words` needs (``eff``, ``signal``, ``anchor``,
    ``marginal``), for a long table of complete trials with ``p_bayes``."""
    fits = fit_all_subjects(long) if len(long) else pd.DataFrame(columns=["source", "success"])
    both = pd.concat([human_sub, bayes_as_source(human_sub), long], ignore_index=True, sort=False)
    return {
        "eff": effective_parameters(fits),
        "signal": signal_response(both),
        "anchor": ratio_anchor_share(both),
        "marginal": marginal_means(both),
    }


def verdict_words(
    source: str,
    eff: pd.DataFrame,
    signal: pd.DataFrame,
    anchor: pd.DataFrame,
    marginal: pd.DataFrame,
) -> dict:
    """Rule-based one-word answers (plus the numbers behind them) for one source:
    copies the printed ratio number / responds to q / responds to d / red lowers the answer.
    Thresholds: copying = the period-1 share of answers equal to the printed ratio number
    or to the printed switch probability, minus the human share in the same cell (the
    q = 0.10, d = 9 cell where both numbers are 10 is left out); >= 20 points = yes,
    >= 8 points = little; effective switch
    rate ratio q=0.10 vs 0.01 >= 1.5 = yes (>= 1.2 weakly; humans 1.7, Bayes 10);
    separation ratio d=9 vs 1.5 >= 1.25 = yes, <= 0.8 inverted, else flat (humans 1.25,
    Bayes 2.1); red lowers = mean change after red < 0 with p < 0.05 across subjects."""
    a = anchor[(anchor.source == source) & (anchor.periods == "period 1")]
    a = a[~a.ambiguous.astype(bool)] if "ambiguous" in a else a
    copy_info = {}
    excesses = []
    for kind in ("ratio_echo", "prior_anchoring"):
        if not len(a) or kind not in a:
            continue
        exc_col = "excess_" + kind
        key = exc_col if exc_col in a else kind
        row = a.loc[a[key].idxmax()]
        copy_info[f"{kind}_max"] = float(row[kind])
        copy_info[f"{kind}_mean"] = float(a[kind].mean())
        copy_info[f"human_{kind}"] = float(row.get("human_" + kind, float("nan")))
        copy_info[f"excess_{kind}"] = float(row[key]) if exc_col in a else float(row[kind])
        copy_info[f"{kind}_cell"] = f"q={row.q:g}, d={row.d:g}"
        excesses.append(copy_info[f"excess_{kind}"])
    pk = max(excesses) if excesses else float("nan")
    e = eff[(eff.source == source) & eff.quantity.str.startswith("effective switch")]["mean"]
    e = e.tolist()
    ratio = e[2] / e[0] if len(e) == 3 and e[0] > 0 else float("nan")
    sd = signal.d.astype(str)
    sep = {}
    for d in D_LEVELS:
        r = signal[(signal.source == source) & (signal.after == "red_minus_blue") & (sd == str(d))]
        sep[d] = float(r.mean_delta.iloc[0]) if len(r) else float("nan")
    tiny = abs(sep.get(1.5, 0.0)) < 0.005 and abs(sep.get(9.0, 0.0)) < 0.005
    rd = float("nan") if tiny or not sep.get(1.5) else sep[9.0] / sep[1.5]
    red = signal[(signal.source == source) & (signal.after == "red") & (sd == "all")]
    if len(red):
        red = red.iloc[0]
        t = red.mean_delta / red["sem"] if red["sem"] else float("nan")
        p = 2 * stats.t.sf(abs(t), red.n_subjects - 1) if np.isfinite(t) else float("nan")
        red_mean, red_sem = float(red.mean_delta), float(red["sem"])
    else:
        p = red_mean = red_sem = float("nan")
    mq = marginal[(marginal.source == source) & (marginal.factor == "q")]
    return {
        "source": source,
        "copy": "yes" if pk >= 0.2 else "little" if pk >= 0.08 else "no",
        "copy_share_p1_max": pk,
        **copy_info,
        "q": "yes" if ratio >= 1.5 else "weakly" if ratio >= 1.2 else "no",
        "q_ratio": ratio,
        "alpha_q": e,
        "q_anova_p": float(mq.anova_p.iloc[0]) if len(mq) else float("nan"),
        "d": "flat" if tiny else "yes" if rd >= 1.25 else "inverted" if rd <= 0.8 else "flat",
        "d_ratio": rd,
        "sep_d1.5": -sep.get(1.5, float("nan")),
        "sep_d9": -sep.get(9.0, float("nan")),
        "red": "yes" if (red_mean < 0 and p < 0.05) else "no",
        "red_delta": red_mean,
        "red_sem": red_sem,
        "red_p": p,
    }


# ---------------------------------------------------------------- figures


# ---------------------------------------------------------------- driver


def run_sanity_checks(
    human_sub: pd.DataFrame,
    model_long: pd.DataFrame,
    fits: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """All check tables for humans, Bayes and every model source, written as CSVs to
    ``out_dir``. ``model_long`` must already be restricted to complete trials and carry
    ``p_bayes``; ``fits`` is the fit table from ``analyze``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    both = pd.concat(
        [human_sub, bayes_as_source(human_sub), model_long], ignore_index=True, sort=False
    )

    t: dict[str, pd.DataFrame] = {}
    t["effective_parameters"] = effective_parameters(fits)
    t["signal_response"] = signal_response(both)
    t["period_means"] = period_means(both)
    t["marginal_means"] = marginal_means(both)
    t["raw_answer_counts"] = raw_answer_counts(both[both.source != BAYES_SOURCE])
    t["ratio_anchor_share"] = ratio_anchor_share(both)
    t["last_period_jump"] = last_period_jump(both)
    t["copying_tolerance_comparison"] = copying_tolerance_comparison(both)
    t["calibration"] = calibration(both)
    state = accuracy_by_posterior_state(both)
    t["accuracy_by_state"] = state["overall"]
    t["accuracy_by_state_condition"] = state["per_condition"]
    for name, table in t.items():
        table.to_csv(out / f"sanity_{name}.csv", index=False)

    return t

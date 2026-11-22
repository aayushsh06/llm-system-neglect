"""Behavioural analyses, applied with identical code to humans and models.

Every function takes the long table (one row per subject, trial, period) and
computes per-subject statistics first; summaries average across subjects
with the standard error across subjects.

- Bayesian reference: :func:`add_bayes`.
- Deviation from Bayes: :func:`index_of_overreaction`, :func:`distance_from_bayes`,
  :func:`grid_summary`, :func:`accuracy_overall`.
- System-neglect parameters: :func:`parameter_table`, :func:`neglect_slopes`,
  :func:`slope_summary`, :func:`fit_quality`.
- Human-model agreement and data quality: :func:`deviation_correlation`,
  :func:`parse_failure_table`, :func:`missing_rate`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .bayes import forward_filter
from .fitting import PARAM_NAMES, design_from_long
from .parsing import ALL_STATUSES, OK_STATUSES
from .task import D_LEVELS, N_PERIODS, Q_LEVELS

TRIAL_KEY = ["source", "subject", "trial"]


def add_bayes(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``p_bayes`` (forward filter, p_0 = 0) to a long table."""
    df = df.sort_values([*TRIAL_KEY, "period"]).reset_index(drop=True)
    p = np.empty(len(df))
    starts = np.arange(0, len(df), N_PERIODS)
    q = df.q.to_numpy(dtype=float)
    d = df.d.to_numpy(dtype=float)
    sig = df.signal.to_numpy(dtype=int)
    for s in starts:
        p[s : s + N_PERIODS] = forward_filter(q[s], d[s], sig[s : s + N_PERIODS])
    df["p_bayes"] = p
    return df


def _sem(x: pd.Series) -> float:
    """Standard error of the mean (ddof = 1) over non-missing values; NaN below two values."""
    x = x.dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def _ttest_zero(x: pd.Series) -> tuple[float, float]:
    """One-sample t-test of ``x`` against 0 as ``(t, p)``; NaN below 3 values or with no spread."""
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(x, 0.0)
    return float(t), float(p)


def index_of_overreaction(df: pd.DataFrame, include_first_period: bool = True) -> pd.DataFrame:
    """IO per subject and condition.

    IO_t = (P_t - P_{t-1}) - (P^B_t - P^B_{t-1}). With ``include_first_period``
    (the archived code's convention) period 1 contributes 0 to the mean;
    otherwise it is excluded.
    """
    if "p_bayes" not in df:
        df = add_bayes(df)
    df = df.sort_values([*TRIAL_KEY, "period"]).reset_index(drop=True)
    est = df.estimate.to_numpy(dtype=float)
    pb = df.p_bayes.to_numpy(dtype=float)
    first = df.period.to_numpy() == 1
    d_est = np.empty_like(est)
    d_pb = np.empty_like(pb)
    d_est[1:] = est[1:] - est[:-1]
    d_pb[1:] = pb[1:] - pb[:-1]
    d_est[first] = 0.0 if include_first_period else np.nan
    d_pb[first] = 0.0 if include_first_period else np.nan
    io = pd.Series(d_est - d_pb, index=df.index, name="io")
    return (
        df.assign(io=io)
        .groupby(["source", "subject", "condition", "q", "d"], sort=True)["io"]
        .mean()
        .reset_index()
    )


def distance_from_bayes(df: pd.DataFrame) -> pd.DataFrame:
    """Mean |estimate - p_bayes| and mean (estimate - p_bayes) per subject and condition."""
    if "p_bayes" not in df:
        df = add_bayes(df)
    dev = df.estimate - df.p_bayes
    g = df.assign(abs_dev=dev.abs(), signed_dev=dev).groupby(
        ["source", "subject", "condition", "q", "d"], sort=True
    )
    return g[["abs_dev", "signed_dev"]].mean().reset_index()


def grid_summary(per_subject: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean, SEM, n and a one-sample t-test against 0 of ``value`` per source and condition."""
    rows = []
    keys = ["source", "condition", "q", "d"]
    for (source, cond, q, d), g in per_subject.groupby(keys, sort=True):
        t, p = _ttest_zero(g[value])
        rows.append(
            {
                "source": source,
                "condition": cond,
                "q": q,
                "d": d,
                "mean": g[value].mean(),
                "sem": _sem(g[value]),
                "n": int(g[value].notna().sum()),
                "t": t,
                "p": p,
            }
        )
    return pd.DataFrame(rows)


def grid_matrix(summary: pd.DataFrame, source: str, column: str = "mean") -> pd.DataFrame:
    """3 x 3 matrix of ``column`` for one source, d on rows and q on columns."""
    sub = summary[summary.source == source]
    return sub.pivot(index="d", columns="q", values=column).reindex(
        index=list(D_LEVELS), columns=list(Q_LEVELS)
    )


def parameter_table(fits: pd.DataFrame) -> pd.DataFrame:
    """Long summary of the six system-neglect parameters per source (successful fits only)."""
    rows = []
    for source, g in fits.groupby("source", sort=True):
        ok = g[g.success.astype(bool)]
        for name in PARAM_NAMES:
            kind, lv = name.split("_")
            level_value = Q_LEVELS[int(lv) - 1] if kind == "alpha" else D_LEVELS[int(lv) - 1]
            rows.append(
                {
                    "source": source,
                    "parameter": kind,
                    "level": level_value,
                    "name": name,
                    "mean": ok[name].mean(),
                    "sem": _sem(ok[name]),
                    "median": ok[name].median(),
                    "n_fits": int(ok[name].notna().sum()),
                    "n_failed": int(len(g) - len(ok)),
                }
            )
    return pd.DataFrame(rows)


def neglect_slopes(fits: pd.DataFrame) -> pd.DataFrame:
    """Per-subject slopes of alpha against q and beta against log d, plus paired contrasts."""
    xq = np.array(Q_LEVELS)
    xd = np.log(np.array(D_LEVELS))
    rows = []
    for _, r in fits[fits.success.astype(bool)].iterrows():
        a = r[["alpha_1", "alpha_2", "alpha_3"]].to_numpy(dtype=float)
        b = r[["beta_1", "beta_2", "beta_3"]].to_numpy(dtype=float)
        rows.append(
            {
                "source": r.source,
                "subject": r.subject,
                "alpha_slope_q": np.polyfit(xq, a, 1)[0],
                "beta_slope_logd": np.polyfit(xd, b, 1)[0],
                "alpha_low_minus_high": a[0] - a[2],
                "beta_low_minus_high": b[0] - b[2],
            }
        )
    return pd.DataFrame(rows)


def slope_summary(slopes: pd.DataFrame) -> pd.DataFrame:
    """Mean, SEM, t-test against 0 and share below 0 of each slope statistic, per source."""
    rows = []
    for source, g in slopes.groupby("source", sort=True):
        cols = ["alpha_slope_q", "beta_slope_logd", "alpha_low_minus_high", "beta_low_minus_high"]
        for col in cols:
            t, p = _ttest_zero(g[col])
            rows.append(
                {
                    "source": source,
                    "statistic": col,
                    "mean": g[col].mean(),
                    "sem": _sem(g[col]),
                    "n": int(g[col].notna().sum()),
                    "t_vs_0": t,
                    "p": p,
                    "share_negative": float((g[col] < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Pearson r over finite pairs and the number of pairs; r is NaN below 3 pairs or no spread."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), n
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), n


def deviation_correlation(human: pd.DataFrame, model: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Correlation of (estimate - p_bayes) between humans and a model on shared sequences.

    Returns tables ``overall``, ``by_condition``, ``by_period``, ``by_subject``
    (within-subject r across that subject's periods) and ``by_subject_summary``.
    """
    if "p_bayes" not in human:
        human = add_bayes(human)
    if "p_bayes" not in model:
        model = add_bayes(model)
    key = ["subject", "trial", "period"]
    h = human[[*key, "condition", "q", "d", "signal", "estimate", "p_bayes"]].rename(
        columns={"estimate": "est_h"}
    )
    m = model[[*key, "signal", "estimate", "source"]].rename(columns={"estimate": "est_m"})
    merged = h.merge(m, on=key, how="inner", suffixes=("", "_m"))
    if len(merged) and not (merged.signal == merged.signal_m).all():
        raise ValueError("human and model rows disagree on the signal: sequences are not shared")
    merged["dev_h"] = merged.est_h - merged.p_bayes
    merged["dev_m"] = merged.est_m - merged.p_bayes
    source = merged.source.iloc[0] if len(merged) else ""
    r, n = _pearson(merged.dev_h.to_numpy(), merged.dev_m.to_numpy())
    overall = pd.DataFrame(
        [
            {
                "source": source,
                "r": r,
                "n_pairs": n,
                "n_matched_rows": len(merged),
                "n_excluded_nan": len(merged) - n,
            }
        ]
    )

    def by(cols: list[str]) -> pd.DataFrame:
        rows = []
        for keys, g in merged.groupby(cols, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rr, nn = _pearson(g.dev_h.to_numpy(), g.dev_m.to_numpy())
            rows.append({"source": source, **dict(zip(cols, keys, strict=True)), "r": rr, "n": nn})
        return pd.DataFrame(rows)

    by_subject = by(["subject"])
    return {
        "overall": overall,
        "by_condition": by(["condition", "q", "d"]),
        "by_period": by(["period"]),
        "by_subject": by_subject,
        "by_subject_summary": pd.DataFrame(
            [
                {
                    "source": source,
                    "mean_r": by_subject.r.mean(),
                    "sem_r": _sem(by_subject.r),
                    "n_subjects": int(by_subject.r.notna().sum()),
                }
            ]
        ),
    }


def parse_failure_table(
    model_long: pd.DataFrame, by: tuple[str, ...] = ("source", "mode")
) -> pd.DataFrame:
    """Rate of every parse status per group, plus the usable-response rate."""
    rows = []
    cols = [c for c in by if c in model_long]
    for keys, g in model_long.groupby(cols, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        counts = g.parse_status.value_counts()
        n = len(g)
        row = dict(zip(cols, keys, strict=True))
        row["n_calls"] = n
        for s in ALL_STATUSES:
            row[s] = counts.get(s, 0) / n if n else float("nan")
        row["usable"] = sum(counts.get(s, 0) for s in OK_STATUSES) / n if n else float("nan")
        row["failed"] = 1.0 - row["usable"] if n else float("nan")
        subj_fail = g.groupby("subject").parse_status.apply(lambda s: (~s.isin(OK_STATUSES)).mean())
        row["share_subjects_over_10pct_failed"] = float((subj_fail > 0.10).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def missing_rate(long_df: pd.DataFrame) -> float:
    """Share of rows without a usable estimate."""
    return float(long_df.estimate.isna().mean())


def accuracy_overall(
    per_subject: pd.DataFrame,
    human_source: str = "human",
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Overall accuracy: mean |estimate - Bayes| averaged over the nine conditions.

    One number per subject (the mean of that subject's nine condition means), then the
    across-subject mean with a percentile bootstrap interval over subjects. For every
    non-human source the paired difference from the humans with the same subject index
    (same trial sequences) is reported with its bootstrap interval and a paired t-test;
    ``distinguishable`` is True when the interval excludes zero.
    """
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    per = per_subject.groupby(["source", "subject"]).abs_dev.mean().unstack("source")
    human = per.get(human_source)
    rows = []
    for src in per.columns:
        x = per[src].dropna()
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        boot = x.to_numpy()[idx].mean(axis=1)
        row = {
            "source": src,
            "n_subjects": int(len(x)),
            "mean_abs_dev": float(x.mean()),
            "ci_low": float(np.quantile(boot, lo_q)),
            "ci_high": float(np.quantile(boot, hi_q)),
        }
        if human is not None and src != human_source:
            pair = pd.concat([x, human], axis=1, join="inner").dropna()
            diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy()
            idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
            dboot = diff[idx].mean(axis=1)
            t, p = stats.ttest_1samp(diff, 0.0) if len(diff) > 2 else (np.nan, np.nan)
            d_lo, d_hi = float(np.quantile(dboot, lo_q)), float(np.quantile(dboot, hi_q))
            row.update(
                n_pairs=int(len(diff)),
                diff_from_human=float(diff.mean()),
                diff_ci_low=d_lo,
                diff_ci_high=d_hi,
                t_paired=float(t),
                p_paired=float(p),
                distinguishable=bool(d_lo > 0 or d_hi < 0),
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    order = [human_source] + sorted(s for s in out.source if s != human_source)
    return out.set_index("source").reindex(order).reset_index()


def fit_quality(long: pd.DataFrame, fits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """R² and RMSE of the fitted system-neglect model and of pure Bayes (all alphas and betas
    = 1), per subject and summarised per source.

    R² is 1 - SSE / SST with SST around the subject's mean answer; it can be negative when a
    model does worse than that constant. A source is flagged ``do_not_interpret`` when the
    neglect fit explains less than half the variance on average (mean R² < 0.5) or when fewer
    than 90% of its subject fits converged: its alphas and betas then describe a curve that
    does not describe the answers.
    """
    rows = []
    fits_idx = fits.set_index(["source", "subject"]) if len(fits) else None
    for (src, subj), g in long.groupby(["source", "subject"], sort=True):
        try:
            design, y = design_from_long(g)
        except ValueError:
            continue
        mask = np.isfinite(y)
        if mask.sum() < 7:
            continue
        yy = y[mask]
        sst = float(np.sum((yy - yy.mean()) ** 2))
        out = {"source": src, "subject": subj, "n_obs": int(mask.sum())}
        bayes = design.predict(np.ones(3), np.ones(3))[mask]
        out["rmse_bayes"] = float(np.sqrt(np.mean((bayes - yy) ** 2)))
        out["r2_bayes"] = 1.0 - float(np.sum((bayes - yy) ** 2)) / sst if sst > 0 else np.nan
        ok = fits_idx is not None and (src, subj) in fits_idx.index
        if ok:
            f = fits_idx.loc[(src, subj)]
            theta = np.array([f[p] for p in PARAM_NAMES], dtype=float)
            ok = bool(f.get("success", False)) and np.all(np.isfinite(theta))
        if ok:
            pred = design.predict(theta[:3], theta[3:])[mask]
            out["rmse_neglect"] = float(np.sqrt(np.mean((pred - yy) ** 2)))
            out["r2_neglect"] = 1.0 - float(np.sum((pred - yy) ** 2)) / sst if sst > 0 else np.nan
            out["fit_success"] = True
        else:
            out["rmse_neglect"] = out["r2_neglect"] = np.nan
            out["fit_success"] = False
        rows.append(out)
    per_subject = pd.DataFrame(rows)
    if not len(per_subject):
        return {"per_subject": per_subject, "summary": pd.DataFrame()}
    summ = (
        per_subject.groupby("source")
        .agg(
            n_subjects=("subject", "size"),
            fit_success_share=("fit_success", "mean"),
            r2_neglect_mean=("r2_neglect", "mean"),
            r2_neglect_min=("r2_neglect", "min"),
            rmse_neglect_mean=("rmse_neglect", "mean"),
            r2_bayes_mean=("r2_bayes", "mean"),
            rmse_bayes_mean=("rmse_bayes", "mean"),
        )
        .reset_index()
    )
    summ["r2_gain_over_bayes"] = summ.r2_neglect_mean - summ.r2_bayes_mean
    summ["do_not_interpret"] = (summ.r2_neglect_mean < 0.5) | (summ.fit_success_share < 0.9)
    summ["reason"] = np.select(
        [summ.fit_success_share < 0.9, summ.r2_neglect_mean < 0.5],
        ["fewer than 90% of subject fits converged", "neglect fit explains < 50% of the variance"],
        "",
    )
    return {"per_subject": per_subject, "summary": summ}

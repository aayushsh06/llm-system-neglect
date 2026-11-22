"""Generate every number, table and figure of the paper from the result files.

    .venv/bin/python scripts/build_paper_numbers.py

Reads the human data (data/human), the raw model results (results/*.jsonl, results/partial/,
results/fake/) and the run logs (results/api_runs.csv, results/model_info.csv), recomputes every
per-subject fit and bootstrap, and writes into paper/:

- numbers.tex    one \\newcommand per number (letters-only macro names)
- numbers.json   the same values as JSON
- tables/*.tex   booktabs tables
- figures/*      the figures, as PDF and PNG
- data/          the intermediate analysis CSVs

Nothing is typed by hand.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_sensitivity import fitted_pairs, free_pairs, summarise  # noqa: E402
from regime_shift import analysis as an  # noqa: E402
from regime_shift import paper_figures as pf  # noqa: E402
from regime_shift import pipeline, prompts, sanity  # noqa: E402
from regime_shift.collection.backends import ANSWER_SCHEMA  # noqa: E402
from regime_shift.collection.runner import read_jsonl  # noqa: E402
from regime_shift.human_data import PUBLISHED_ALPHA_MEANS, PUBLISHED_BETA_MEANS  # noqa: E402
from regime_shift.paper_figures import order_sources, plain_name  # noqa: E402
from regime_shift.parsing import OK_STATUSES  # noqa: E402
from regime_shift.results import complete_trials_only, load_results  # noqa: E402
from regime_shift.task import BLUE, D_LEVELS, N_PERIODS, Q_LEVELS  # noqa: E402

RES = ROOT / "results"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figures"
DATA = PAPER / "data"

ROWS_PER_SUBJECT = 720  # 72 trials x 10 periods
N_SUBJECTS_FULL = 30
ROWS_PER_FULL_RUN = N_SUBJECTS_FULL * ROWS_PER_SUBJECT

SONNET_NO = "claude-sonnet-5-5 [main, T1, number only]"
HAIKU_NO = "claude-haiku-5-5 [main, T1, number only]"
CELLS = [(q, d) for q in Q_LEVELS for d in D_LEVELS]  # the nine conditions, q-major

# ------------------------------------------------------------------ source keys

# model tag -> the word that starts every macro name of that model
TAG_WORDS = {
    "qwen2.5:3b": "QwenThreeB",
    "qwen2.5:7b": "QwenSevenB",
    "qwen2.5:14b": "QwenFourteenB",
    "qwen2.5:32b": "QwenThirtyTwoB",
    "llama3.1:8b": "Llama",
    "gemma2:9b": "Gemma",
    "granite4.2:8b": "Granite",
    "claude-haiku-5-5": "Haiku",
    "claude-sonnet-5-5": "Sonnet",
    "claude-opus-5-5": "Opus",
    "claude-haiku-4-5": "HaikuFourFive",
    "claude-sonnet-4-6": "SonnetFourSix",
    "claude-sonnet-5": "SonnetFive",
    "human": "Hum",
    "Bayes": "Bayes",
}
# words inside the source's bracket -> macro suffix
WORDS = {
    "stateless": "Stateless",
    "greedy": "Greedy",
    "number only": "NO",
    "reasoning allowed": "RA",
    "ratio-words": "RatioWords",
    "no-horizon": "NoHorizon",
}

NUM = ("One", "Two", "Three")


def source_key(source: str) -> str:
    """Letters-only macro prefix for a source label.

    'qwen2.5:14b [stateless, T0.7, PARTIAL s1-10]' -> 'QwenFourteenBStatelessP'. The temperature
    and the word 'main' are dropped; a partial run gets a trailing 'P' (subjects 1-10) or
    'PThree', 'PFive', 'PFifteen' for the other subject ranges.
    """
    m = re.match(r"^(?P<tag>[^\[]+?)\s*(?:\[(?P<inside>[^\]]*)\])?$", source)
    tag = m.group("tag").strip()
    inside = m.group("inside") or ""
    key = TAG_WORDS.get(tag, re.sub(r"[^A-Za-z]", "", tag.title()))
    parts = [p.strip() for p in inside.split(",") if p.strip()]
    partial: bool | str = False
    for p in parts:
        if p == "main" or re.match(r"^T[\d.]+$", p):
            continue
        if p.startswith("PARTIAL"):
            partial = True
            m_ = re.search(r"s1-(\d+)", p)
            n_ = int(m_.group(1)) if m_ else 10
            if n_ != 10:  # the standard partial size keeps the bare "P" suffix
                partial = {3: "PThree", 5: "PFive", 15: "PFifteen"}.get(n_, f"P{n_}")
        elif p.startswith("reasoning allowed"):
            key += "RA"
            eff = re.search(r"effort (\w+)", p)
            if eff:
                key += eff.group(1).title()
        elif p in WORDS:
            key += WORDS[p]
        else:
            key += re.sub(r"[^A-Za-z]", "", p.title())
    if "all 99 trials" in source:
        key = "HumAll"
    if not partial:
        return key
    return key + ("P" if partial is True else partial)


def parse_tag(source: str) -> str:
    """'qwen2.5:14b [main, T0.7]' -> 'qwen2.5:14b'."""
    return re.match(r"^([^\[]+?)\s*(?:\[|$)", source).group(1).strip()


def sens_key(src: str) -> str:
    """Macro prefix for a sensitivity row: humans, a rule-based agent or a model source."""
    if src == "human":
        return "Hum"
    if src.startswith("fake"):
        return fake_key(src.split(" [")[0])
    return source_key(src)


def fake_key(tag: str) -> str:
    """'fake-random30' -> 'FakeRandom'."""
    return "Fake" + re.sub(r"[^A-Za-z]", "", tag.replace("fake-", "").title())


# ------------------------------------------------------------------ formatting


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return f"{x:.{nd}f}"


def pct(x, nd=0):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return f"{100 * x:.{nd}f}"


def sci(p):
    """p value: three decimals, or scientific notation below 1e-4."""
    if p is None or not math.isfinite(p):
        return None
    if p < 1e-4:
        e = int(math.floor(math.log10(p)))
        return f"{p / 10**e:.1f} \\times 10^{{{e}}}"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def latex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def short_name(src: str) -> str:
    return plain_name(src)


def row_label(src: str) -> str:
    """Table row label: the plain name without the number-only tag (captions state it)."""
    label = short_name(src).replace(" (number only)", "").replace(", number only", "")
    return re.sub(r"subjects (\d+)-(\d+)", r"s\1--\2", label)


class Macros:
    """The numbers of the paper: one LaTeX macro each, plus the same values as JSON.

    ``set`` stores a formatted value (``None``, a NaN in the analysis, becomes a dash);
    ``write`` emits numbers.tex and numbers.json."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.json: dict[str, object] = {}

    def set(self, name: str, value, raw=None) -> None:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name must be letters only: {name}")
        if value is None:  # not available (NaN in the analysis): typeset as a dash
            self.values[name] = "--"
        else:
            self.values[name] = str(value)
        self.json[name] = raw if raw is not None else value

    def write(self) -> None:
        lines = [
            "% GENERATED by scripts/build_paper_numbers.py -- do not edit by hand",
            f"% generated {time.strftime('%Y-%m-%d %H:%M')}",
        ]
        for k in sorted(self.values):
            lines.append(f"\\newcommand{{\\{k}}}{{{self.values[k]}}}")
        (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
        (PAPER / "numbers.json").write_text(json.dumps(self.json, indent=1, default=str))


# ------------------------------------------------------------------ stopped partial runs

# Partial runs that were stopped on purpose (the run stops at the first refusal). Their complete
# trials are analysed and their macros emitted although the file holds fewer rows than
# subjects x 720.
STOPPED_PARTIALS = {
    "claude-sonnet-5-5 [main, T1, reasoning allowed, PARTIAL s1-3]",
}


# ------------------------------------------------------------------ analysis


def drop_sources(obj, sources):
    """Remove rows of ``sources`` from every DataFrame (with a ``source`` column) in a dict."""
    if isinstance(obj, pd.DataFrame):
        return obj[~obj["source"].isin(sources)] if "source" in obj.columns else obj
    if isinstance(obj, dict):
        return {k: drop_sources(v, sources) for k, v in obj.items()}
    return obj


def analyse_folder(folder: Path, label: str, human: dict, out: Path):
    """Run the full analysis and the sanity checks on every JSONL file in ``folder``.

    Returns ``(long, complete, tables, sane)``: the long per-period frame, the complete trials
    with the Bayesian posterior, the analysis tables and the sanity-check tables; ``None`` when
    the folder holds no results.
    """
    if not folder.exists() or not any(folder.glob("*.jsonl")):
        return None
    long = load_results(folder)
    tables = pipeline.analyze(
        human["human_sub"], long, out / "analysis", human_all=human["human_all"], label=label
    )
    complete, _ = complete_trials_only(long)
    complete = an.add_bayes(complete) if len(complete) else complete
    sane = sanity.run_sanity_checks(human["human_sub"], complete, tables["fits"], out / "explore")
    return long, complete, tables, sane


def emit_source(
    M: Macros, key: str, src: str, tables, sane, long_rows: pd.DataFrame | None
) -> None:
    """Every per-source macro of ``src`` under the prefix ``key``: fitted parameters, effective
    parameters, distance from the posterior, fit quality, state and calibration numbers, copying
    shares, correlation with the humans, signal response, verdict words and row counts."""
    params = tables["parameters"].pivot(index="source", columns="name", values="mean")
    sems = tables["parameters"].pivot(index="source", columns="name", values="sem")
    if src in params.index:
        for i, nm in enumerate(NUM):
            M.set(
                f"{key}Alpha{nm}",
                fmt(params.loc[src, f"alpha_{i + 1}"]),
                params.loc[src, f"alpha_{i + 1}"],
            )
            M.set(
                f"{key}AlphaSem{nm}",
                fmt(sems.loc[src, f"alpha_{i + 1}"]),
                sems.loc[src, f"alpha_{i + 1}"],
            )
            M.set(
                f"{key}Beta{nm}",
                fmt(params.loc[src, f"beta_{i + 1}"]),
                params.loc[src, f"beta_{i + 1}"],
            )
            M.set(
                f"{key}BetaSem{nm}",
                fmt(sems.loc[src, f"beta_{i + 1}"]),
                sems.loc[src, f"beta_{i + 1}"],
            )
    eff = sane["effective_parameters"]
    e = eff[eff.source == src]
    eq = e[e.quantity.str.startswith("effective switch")]["mean"].tolist()
    ed = e[e.quantity.str.startswith("effective diag")]["mean"].tolist()
    for i, nm in enumerate(NUM):
        M.set(
            f"{key}EffQ{nm}",
            fmt(eq[i], 3) if len(eq) == 3 else None,
            eq[i] if len(eq) == 3 else None,
        )
        M.set(
            f"{key}EffD{nm}",
            fmt(ed[i], 1) if len(ed) == 3 else None,
            ed[i] if len(ed) == 3 else None,
        )
    acc = tables["accuracy_overall"].set_index("source")
    if src in acc.index:
        a = acc.loc[src]
        M.set(f"{key}Dev", fmt(a.mean_abs_dev, 3), a.mean_abs_dev)
        M.set(f"{key}DevLo", fmt(a.ci_low, 3), a.ci_low)
        M.set(f"{key}DevHi", fmt(a.ci_high, 3), a.ci_high)
        if "diff_from_human" in a and pd.notna(a.get("diff_from_human", np.nan)):
            M.set(f"{key}Diff", f"{a.diff_from_human:+.3f}", a.diff_from_human)
            M.set(f"{key}DiffLo", f"{a.diff_ci_low:+.3f}", a.diff_ci_low)
            M.set(f"{key}DiffHi", f"{a.diff_ci_high:+.3f}", a.diff_ci_high)
            M.set(f"{key}DiffP", sci(float(a.p_paired)), float(a.p_paired))
            M.set(
                f"{key}Distinguishable",
                "yes" if bool(a.distinguishable) else "no",
                bool(a.distinguishable),
            )
    fq = tables["fit_quality"].set_index("source") if len(tables["fit_quality"]) else pd.DataFrame()
    if src in fq.index:
        f = fq.loc[src]
        M.set(f"{key}RtwoNeglect", fmt(f.r2_neglect_mean), f.r2_neglect_mean)
        M.set(f"{key}RtwoBayes", fmt(f.r2_bayes_mean), f.r2_bayes_mean)
        M.set(f"{key}RmseNeglect", fmt(f.rmse_neglect_mean), f.rmse_neglect_mean)
        M.set(
            f"{key}FitFlag",
            "do not interpret" if bool(f.do_not_interpret) else "interpretable",
            bool(f.do_not_interpret),
        )
    st = sane["accuracy_by_state"]
    s = st[st.source == src]
    if len(s):
        lo = s[s.state.str.contains("<")]
        hi = s[s.state.str.contains(">=")]
        M.set(
            f"{key}Quiet",
            fmt(float(lo.mean_abs_dev.iloc[0]), 3) if len(lo) else None,
            float(lo.mean_abs_dev.iloc[0]) if len(lo) else None,
        )
        M.set(
            f"{key}Active",
            fmt(float(hi.mean_abs_dev.iloc[0]), 3) if len(hi) else None,
            float(hi.mean_abs_dev.iloc[0]) if len(hi) else None,
        )
    cal = sane["calibration"]
    c = cal[cal.source == src].sort_values("mean_bayes")
    for i, nm in enumerate(("One", "Two", "Three", "Four", "Five")):
        M.set(
            f"{key}Cal{nm}",
            fmt(float(c.mean_stated.iloc[i]), 2) if len(c) == 5 else None,
            float(c.mean_stated.iloc[i]) if len(c) == 5 else None,
        )
    cmp_ = sane["copying_tolerance_comparison"].set_index("source")
    if src in cmp_.index:
        r = cmp_.loc[src]
        M.set(
            f"{key}RatioEchoMax",
            pct(r["ratio echo max | new rule"]),
            r["ratio echo max | new rule"],
        )
        M.set(
            f"{key}RatioEchoMean",
            pct(r["ratio echo mean | new rule"]),
            r["ratio echo mean | new rule"],
        )
        M.set(
            f"{key}PriorAnchMax",
            pct(r["prior anchoring max | new rule"]),
            r["prior anchoring max | new rule"],
        )
        M.set(
            f"{key}PriorAnchMean",
            pct(r["prior anchoring mean | new rule"]),
            r["prior anchoring mean | new rule"],
        )
    co = tables["correlation_overall"]
    if len(co) and (co.source == src).any():
        M.set(
            f"{key}Corr",
            fmt(float(co[co.source == src].r.iloc[0])),
            float(co[co.source == src].r.iloc[0]),
        )
    lj = sane["last_period_jump"]
    if (lj.source == src).any():
        M.set(
            f"{key}Jump",
            f"{float(lj[lj.source == src].jump.iloc[0]):+.3f}",
            float(lj[lj.source == src].jump.iloc[0]),
        )
    if src not in ("human", "Bayes"):
        try:
            v = sanity.verdict_words(
                src,
                sane["effective_parameters"],
                sane["signal_response"],
                sane["ratio_anchor_share"],
                sane["marginal_means"],
            )
            M.set(f"{key}RedDelta", f"{v['red_delta']:+.3f}", v["red_delta"])
            M.set(f"{key}SepRatio", fmt(v["d_ratio"]), v["d_ratio"])
            M.set(f"{key}QRatio", fmt(v["q_ratio"]), v["q_ratio"])
            M.set(f"{key}VerdictCopy", v["copy"], v["copy"])
            M.set(f"{key}VerdictQ", v["q"], v["q"])
            M.set(f"{key}VerdictD", v["d"], v["d"])
            M.set(f"{key}VerdictRed", v["red"], v["red"])
        except Exception:  # noqa: BLE001
            pass  # a source without enough rows for a verdict
    sr = sane["signal_response"]
    b = sr[(sr.source == src) & (sr.after == "blue") & (sr.d.astype(str) == "all")]
    if len(b):
        M.set(f"{key}BlueDelta", f"{float(b.mean_delta.iloc[0]):+.3f}", float(b.mean_delta.iloc[0]))
    r_ = sr[(sr.source == src) & (sr.after == "red") & (sr.d.astype(str) == "all")]
    if len(r_) and f"{key}RedDelta" not in M.values:
        M.set(
            f"{key}RedDelta", f"{float(r_.mean_delta.iloc[0]):+.3f}", float(r_.mean_delta.iloc[0])
        )
    if long_rows is not None and len(long_rows):
        M.set(f"{key}Rows", f"{len(long_rows):,}", len(long_rows))
        usable = long_rows.parse_status.isin(OK_STATUSES).mean()
        M.set(f"{key}Usable", pct(usable, 1), usable)
        M.set(f"{key}Subjects", str(long_rows.subject.nunique()), int(long_rows.subject.nunique()))


def eff_cells(df: pd.DataFrame, src: str) -> list[str]:
    """Six table cells for ``src``: effective switch rate at the three q levels (3 decimals) and
    effective diagnosticity at the three d levels (2 decimals); '--' where missing."""
    out = []
    for quantity, levels in (("effective switch", Q_LEVELS), ("effective diag", D_LEVELS)):
        sub = (
            df[(df.source == src) & df.quantity.str.startswith(quantity)]
            .set_index("level")
            .reindex(list(levels))
        )
        out += [
            "--" if pd.isna(v) else (f"{v:.3f}" if quantity == "effective switch" else f"{v:.2f}")
            for v in sub["mean"]
        ]
    return out


# ------------------------------------------------------------------ per-source tables


def table_main_rows(tables, sane, sources: list[str]) -> list[str]:
    """One row per source: fitted alpha and beta, effective parameters, distance with CI, R^2 of
    the neglect model and whether the fit is interpretable."""
    params = tables["parameters"].pivot(index="source", columns="name", values="mean")
    acc = tables["accuracy_overall"].set_index("source")
    fq = tables["fit_quality"].set_index("source")
    eff = sane["effective_parameters"]
    rows = []
    for src in sources:
        if src not in params.index or not np.isfinite(params.loc[src, "alpha_1"]):
            continue
        a = [f"{params.loc[src, f'alpha_{i}']:.2f}" for i in (1, 2, 3)]
        b = [f"{params.loc[src, f'beta_{i}']:.2f}" for i in (1, 2, 3)]
        e = eff[eff.source == src]
        eq = e[e.quantity.str.startswith("effective switch")]["mean"].tolist()
        ed = e[e.quantity.str.startswith("effective diag")]["mean"].tolist()
        dev = (
            f"{acc.loc[src, 'mean_abs_dev']:.3f} [{acc.loc[src, 'ci_low']:.3f}, {acc.loc[src, 'ci_high']:.3f}]"
            if src in acc.index
            else ""
        )
        r2 = f"{fq.loc[src, 'r2_neglect_mean']:.2f}" if src in fq.index else ""
        flag = (
            "\\textbf{no}" if src in fq.index and bool(fq.loc[src, "do_not_interpret"]) else "yes"
        )
        rows.append(
            f"{latex_escape(row_label(src))} & {', '.join(a)} & {', '.join(b)} & "
            f"{', '.join(f'{x:.3f}' for x in eq)} & {', '.join(f'{x:.1f}' for x in ed)} & {dev} & {r2} & {flag} \\\\"
        )
    return rows


def table_main(tables, sane, sources: list[str], path: Path) -> None:
    """Parameters, effective parameters, distance and fit quality, one row per source."""
    body = "\n".join(table_main_rows(tables, sane, sources))
    path.write_text(
        "\\begin{tabular}{@{}lllllllc@{}}\n\\toprule\n"
        "Source & $\\alpha$ ($q$ = .01, .05, .10) & $\\beta$ ($d$ = 1.5, 3, 9) & $\\alpha q$ & $d^{\\beta}$ & "
        "$|\\hat p - p^{B}|$ [95\\% CI] & $R^2$ & Interpret \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def table_state_cal(sane, sources: list[str], path: Path) -> None:
    """Distance from the posterior in quiet and active periods, and the mean stated probability
    in each of the five bins of the Bayesian posterior."""
    st = sane["accuracy_by_state"]
    cal = sane["calibration"]
    rows = []
    for src in sources:
        s = st[st.source == src]
        c = cal[cal.source == src].sort_values("mean_bayes")
        if not len(s) or len(c) != 5:
            continue
        lo = float(s[s.state.str.contains("<")].mean_abs_dev.iloc[0])
        hi = float(s[s.state.str.contains(">=")].mean_abs_dev.iloc[0])
        rows.append(
            f"{latex_escape(row_label(src))} & {lo:.3f} & {hi:.3f} & "
            + " & ".join(f"{v:.2f}" for v in c.mean_stated)
            + " \\\\"
        )
    bins = cal[cal.source == "Bayes"].sort_values("mean_bayes").mean_bayes.tolist()
    header = (
        " & ".join(f"{v:.2f}" for v in bins)
        if len(bins) == 5
        else "bin 1 & bin 2 & bin 3 & bin 4 & bin 5"
    )
    path.write_text(
        "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{$|\\hat p - p^{B}|$ by state} & \\multicolumn{5}{c}{mean stated probability by Bayesian bin} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-8}\n"
        f"Source & quiet ($p^B<0.2$) & active ($p^B\\ge 0.2$) & {header} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_copying(sane, sources: list[str], path: Path) -> None:
    """Share of answers that echo a printed number: the minority ratio, the prior, or both."""
    cmp_ = sane["copying_tolerance_comparison"].set_index("source")
    rows = []
    for src in sources:
        if src not in cmp_.index:
            continue
        r = cmp_.loc[src]
        vals = [
            r["ratio echo max | new rule"],
            r["ratio echo mean | new rule"],
            r["prior anchoring max | new rule"],
            r["prior anchoring mean | new rule"],
            r["both numbers q=.10,d=9 | new rule"],
        ]
        rows.append(
            f"{latex_escape(row_label(src))} & "
            + " & ".join("" if pd.isna(v) else f"{100 * v:.0f}" for v in vals)
            + " \\\\"
        )
    path.write_text(
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{ratio echo (\\%)} & \\multicolumn{2}{c}{prior anchoring (\\%)} & both numbers \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Source & max & mean & max & mean & $q$=.10, $d$=9 (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_history(M: Macros, cp, tp, human: dict, sens: pd.DataFrame, path: Path) -> None:
    """Multi-turn against stateless on subjects 1-15 for every Claude model that has both
    partial files: distance, mean signed error in active periods, s_q and s_d under each
    protocol. Also emits the protocol-shift macros and the human numbers on the same subjects.

    ``cp`` and ``tp`` are the complete trials and the analysis tables of the partial folder.
    """
    acc = tp["accuracy_overall"].set_index("source")
    models = [
        ("claude-haiku-4-5", "Haiku 4.5"),
        ("claude-haiku-5-5", "Haiku 5.5"),
        ("claude-sonnet-4-6", "Sonnet 4.6"),
        ("claude-sonnet-5", "Sonnet 5"),
        ("claude-sonnet-5-5", "Sonnet 5.5"),
    ]
    pairs = []
    for tag, name in models:
        mt = f"{tag} [main, T1, number only, PARTIAL s1-15]"
        sl = f"{tag} [stateless, T1, number only, PARTIAL s1-15]"
        if mt in acc.index and sl in acc.index:
            pairs.append((name, mt, sl))
    if not pairs:
        return

    def signed_active(rows: pd.DataFrame) -> float:
        """Mean over subjects of the mean (stated - posterior) in periods with posterior >= 0.2."""
        r = rows[rows.p_bayes >= 0.2].dropna(subset=["estimate"])
        per = (r.estimate - r.p_bayes).groupby(r.subject).mean()
        return float(per.mean())

    def first_blue_d1(rows: pd.DataFrame) -> float:
        """Mean period-1 answer (in points) after a blue ball at d = 1.5."""
        r = rows[(rows.period == 1) & (rows.signal == BLUE) & (rows.d == 1.5)].dropna(
            subset=["estimate"]
        )
        return float(100 * r.groupby("subject").estimate.mean().mean())

    # humans on the same 15 subjects: distance with a subject bootstrap, quiet/active, signed error
    hs = human["human_sub"]
    h15 = hs[hs.subject <= 15]
    per = h15.dropna(subset=["estimate"]).assign(ad=lambda d_: (d_.estimate - d_.p_bayes).abs())
    per_subj = per.groupby(["subject", "condition"]).ad.mean().groupby("subject").mean()
    rng_ = np.random.default_rng(0)
    boots = [
        per_subj.sample(frac=1, replace=True, random_state=int(rng_.integers(1 << 31))).mean()
        for _ in range(2000)
    ]
    hq = float(per[per.p_bayes < 0.2].ad.mean())
    ha = float(per[per.p_bayes >= 0.2].ad.mean())
    hsa = signed_active(h15)
    hfb = first_blue_d1(h15)
    hlo, hhi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    M.set("HumFifteenDev", fmt(per_subj.mean(), 3), float(per_subj.mean()))
    M.set("HumFifteenDevLo", fmt(hlo, 3), hlo)
    M.set("HumFifteenDevHi", fmt(hhi, 3), hhi)
    M.set("HumFifteenQuiet", fmt(hq, 3), hq)
    M.set("HumFifteenActive", fmt(ha, 3), ha)
    M.set("HumFifteenSignedActive", f"{hsa:+.3f}", hsa)
    M.set("HumFifteenFirstBlueDOneMean", f"{hfb:.0f}", hfb)

    # per model: both protocols, then the shift from multi-turn to stateless
    max_stateless_dev = 0.0
    for _name, mt, sl in pairs:
        vals = {}
        for label, src in (("multi-turn", mt), ("stateless", sl)):
            a_ = acc.loc[src]
            rows_ = cp[cp.source == src]
            sa_ = signed_active(rows_)
            fb_ = first_blue_d1(rows_)
            key_ = source_key(src)
            M.set(f"{key_}SignedActive", f"{sa_:+.3f}", sa_)
            M.set(f"{key_}FirstBlueDOneMean", f"{fb_:.0f}", fb_)
            vals[label] = {"dev": float(a_.mean_abs_dev), "sa": sa_, "rows": rows_, "key": key_}
        k = vals["stateless"]["key"]
        shift = vals["stateless"]["sa"] - vals["multi-turn"]["sa"]
        M.set(f"{k}SignedActiveShift", f"{shift:+.2f}", shift)
        ddev = vals["stateless"]["dev"] - vals["multi-turn"]["dev"]
        M.set(f"{k}DevShift", f"{ddev:+.3f}", ddev)
        # period-1 agreement between the protocols (identical first prompt)
        m1 = vals["multi-turn"]["rows"]
        s1 = vals["stateless"]["rows"]
        m1 = m1[m1.period == 1].set_index(["subject", "trial"]).estimate
        s1 = s1[s1.period == 1].set_index(["subject", "trial"]).estimate
        jj = m1.to_frame("m").join(s1.to_frame("s")).dropna()
        M.set(
            f"{k}PeriodOneMeanDiff",
            f"{100 * float((jj.s - jj.m).mean()):+.1f}",
            float((jj.s - jj.m).mean()),
        )
        M.set(
            f"{k}PeriodOneSameShare",
            pct(float((jj.m == jj.s).mean()), 0),
            float((jj.m == jj.s).mean()),
        )
        max_stateless_dev = max(
            max_stateless_dev, vals["stateless"]["dev"], vals["multi-turn"]["dev"]
        )
    M.set("StatelessMaxDev", fmt(max_stateless_dev, 3), max_stateless_dev)
    M.set("NStatelessPairs", str(len(pairs)), len(pairs))

    # the table: distance, signed error in active periods, s_q, s_d; both protocols
    si = sens.set_index("source")
    hl = [
        "\\begin{tabular}{@{}lcccccccc@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{distance} & \\multicolumn{2}{c}{signed error, active} & \\multicolumn{2}{c}{$s_q$} & \\multicolumn{2}{c}{$s_d$} \\\\",
        "model & multi-turn & stateless & multi-turn & stateless & multi-turn & stateless & multi-turn & stateless \\\\",
        "\\midrule",
    ]
    for name, mt, sl in pairs:
        cells = [f"{acc.loc[src].mean_abs_dev:.3f}" for src in (mt, sl)]
        cells += [f"{signed_active(cp[cp.source == src]):+.3f}" for src in (mt, sl)]
        for col in ("s_q", "s_d"):
            cells += [f"{si.loc[src, col]:.2f}" if src in si.index else "--" for src in (mt, sl)]
        hl.append(f"{latex_escape(name)} & " + " & ".join(cells) + " \\\\")
    hl += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(hl) + "\n")


def compact_columns(path: Path, drop: tuple[int, ...] = (3, 4)) -> None:
    """Remove the given cell indices from every row of a generated tabular (the robustness
    table drops the alpha q and d^beta groups, which the main table already reports)."""
    lines = path.read_text().split("\n")
    out = []
    for line in lines:
        if "&" in line and not line.lstrip().startswith("\\"):
            cells = line.split("&")
            if len(cells) > max(drop):
                cells = [c for k, c in enumerate(cells) if k not in drop]
                line = "&".join(cells)
        elif line.startswith("\\begin{tabular}"):
            line = re.sub(
                r"\{@\{\}l([lc]+)@\{\}\}",
                lambda m: "{@{}l" + m.group(1)[: len(m.group(1)) - len(drop)] + "@{}}",
                line,
            )
        out.append(line)
    path.write_text("\n".join(out))


# ------------------------------------------------------------------ design and run facts


def emit_task_facts(M: Macros, human: dict) -> None:
    """Constants of the design and facts of the human data: the validation gate against the
    published per-subject means, the published parameter means, trial, period and condition
    counts, the q and d levels, missing-answer rates."""
    gate = pipeline.run_gate(human["human_all"], human["published"])
    M.set(
        "GateMaxSubject",
        f"{gate.max_abs_subject_diff:.1e}".replace("e-0", "e-"),
        gate.max_abs_subject_diff,
    )
    M.set("GateMaxMean", fmt(gate.max_abs_mean_diff, 4), gate.max_abs_mean_diff)
    mm = re.search(r"\(tol ([\d.e-]+)\).*\(tol ([\d.e-]+)\)", gate.message)
    M.set("GateTolSubject", mm.group(1) if mm else None, float(mm.group(1)) if mm else None)
    M.set("GateTolMean", mm.group(2) if mm else None, float(mm.group(2)) if mm else None)
    M.set("GatePassed", "passed" if gate.passed else "FAILED", bool(gate.passed))
    for i, nm in enumerate(NUM):
        M.set(f"PubAlpha{nm}", fmt(PUBLISHED_ALPHA_MEANS[i]), PUBLISHED_ALPHA_MEANS[i])
        M.set(f"PubBeta{nm}", fmt(PUBLISHED_BETA_MEANS[i]), PUBLISHED_BETA_MEANS[i])
    hs = human["human_sub"]
    ha = human["human_all"]
    M.set("NHumanSubjects", str(hs.subject.nunique()), int(hs.subject.nunique()))
    M.set("NTrialsBehavioural", str(int(hs.groupby("subject").trial.nunique().iloc[0])), 72)
    M.set("HumanMissingRate", pct(hs.estimate.isna().mean(), 2), float(hs.estimate.isna().mean()))
    M.set("NCallsPerFullRun", f"{ROWS_PER_FULL_RUN:,}", ROWS_PER_FULL_RUN)
    M.set("NPeriods", str(N_PERIODS), N_PERIODS)
    M.set("NConditions", str(len(Q_LEVELS) * len(D_LEVELS)), len(Q_LEVELS) * len(D_LEVELS))
    M.set("QLevels", ", ".join(f"{q:g}" for q in Q_LEVELS), list(Q_LEVELS))
    M.set("DLevels", ", ".join(f"{d:g}" for d in D_LEVELS), list(D_LEVELS))
    M.set("QLevelLow", f"{Q_LEVELS[0]:g}", Q_LEVELS[0])
    M.set("QLevelMid", f"{Q_LEVELS[1]:g}", Q_LEVELS[1])
    M.set("QLevelHigh", f"{Q_LEVELS[2]:g}", Q_LEVELS[2])
    M.set("DLevelLow", f"{D_LEVELS[0]:g}", D_LEVELS[0])
    M.set("DLevelMid", f"{D_LEVELS[1]:g}", D_LEVELS[1])
    M.set("DLevelHigh", f"{D_LEVELS[2]:g}", D_LEVELS[2])
    n_all = int(ha.groupby("subject").trial.nunique().iloc[0])
    n_beh = int(hs.groupby("subject").trial.nunique().iloc[0])
    M.set("NTrialsAll", str(n_all), n_all)
    M.set("NTrialsFmri", str(n_all - n_beh), n_all - n_beh)
    M.set(
        "TrialsPerCondition",
        str(int(hs.groupby(["subject", "condition"]).trial.nunique().iloc[0])),
        None,
    )
    M.set(
        "HumanMissingRateAll", pct(ha.estimate.isna().mean(), 2), float(ha.estimate.isna().mean())
    )
    M.set("NHumanPeriodsAll", f"{len(ha):,}", len(ha))
    M.set("NHumanPeriods", f"{len(hs):,}", len(hs))


def emit_model_info(M: Macros) -> None:
    """Quantisation, parameter count and size of every open-weight model (results/model_info.csv)."""
    info = RES / "model_info.csv"
    if not info.exists():
        return
    for _, r in pd.read_csv(info).iterrows():
        key = TAG_WORDS.get(r.tag)
        if key:
            M.set(f"{key}Quant", latex_escape(str(r.quantization)), str(r.quantization))
            M.set(f"{key}Params", latex_escape(str(r.parameter_size)), str(r.parameter_size))
            M.set(f"{key}SizeGb", fmt(int(r.size_bytes) / 2**30, 1), int(r.size_bytes) / 2**30)


def emit_main_sources(M: Macros, human: dict):
    """Analyse results/ (the 30-subject runs): per-source macros for the humans, the Bayesian
    observer and every complete run; an incomplete run is dropped from every frame so that it
    cannot reach a table or figure.

    Returns ``(long, complete, tables, sane, complete_full, in_progress)``.
    """
    long_main, complete_main, tables_main, sane_main = analyse_folder(RES, "", human, DATA / "main")
    full_sources = order_sources(complete_main.source.unique())
    emit_source(M, "Hum", "human", tables_main, sane_main, None)
    emit_source(M, "HumAll", "human (all 99 trials)", tables_main, sane_main, None)
    emit_source(M, "Bayes", "Bayes", tables_main, sane_main, None)
    complete_full = []
    for src in full_sources:
        g = long_main[long_main.source == src]
        if len(g) >= ROWS_PER_FULL_RUN:
            emit_source(M, source_key(src), src, tables_main, sane_main, g)
            complete_full.append(src)
    in_progress = [s_ for s_ in full_sources if s_ not in complete_full]
    if in_progress:
        print(f"in progress, excluded from every main table and figure: {in_progress}")
        long_main = long_main[~long_main.source.isin(in_progress)]
        complete_main = complete_main[~complete_main.source.isin(in_progress)]
        tables_main = drop_sources(tables_main, in_progress)
        sane_main = drop_sources(sane_main, in_progress)
    return long_main, complete_main, tables_main, sane_main, complete_full, in_progress


def emit_partial_sources(M: Macros, human: dict):
    """Analyse results/partial (runs on a subset of subjects): per-source macros for every file
    that is complete for its subject range or was stopped on purpose.

    Returns ``(partial, incomplete)``: the ``analyse_folder`` tuple (``None`` when the folder is
    empty) and the set of in-progress sources, which get no table rows.
    """
    partial = analyse_folder(RES / "partial", "partial", human, DATA / "partial")
    incomplete: set[str] = set()
    if partial is not None:
        long_p, complete_p, tables_p, sane_p = partial
        for src in complete_p.source.unique():
            g = long_p[long_p.source == src]
            n_sub = (
                int(re.search(r"s1-(\d+)", src).group(1)) if "PARTIAL" in src else N_SUBJECTS_FULL
            )
            if len(g) >= n_sub * ROWS_PER_SUBJECT or src in STOPPED_PARTIALS:
                emit_source(M, source_key(src), src, tables_p, sane_p, g)
            else:
                incomplete.add(src)
    return partial, incomplete


def emit_calibration_bins(M: Macros, sane_main: dict) -> None:
    """Share of quiet periods for the humans and the five bins of the Bayesian posterior."""
    st = sane_main["accuracy_by_state"]
    quiet_share = float(
        st[(st.source == "human") & st.state.str.contains("<")].share_of_periods.iloc[0]
    )
    M.set("QuietShare", pct(quiet_share), quiet_share)
    cal = sane_main["calibration"]
    bins = cal[cal.source == "Bayes"].sort_values("mean_bayes")
    for i, nm in enumerate(("One", "Two", "Three", "Four", "Five")):
        M.set(
            f"BayesBin{nm}", fmt(float(bins.mean_bayes.iloc[i]), 3), float(bins.mean_bayes.iloc[i])
        )
        M.set(f"BayesBinLabel{nm}", latex_escape(str(bins.bin.iloc[i])), str(bins.bin.iloc[i]))


def emit_api_determinism(M: Macros, complete_main: pd.DataFrame, complete_full: list[str]) -> None:
    """How many period-1 cells (q, d, first ball) got the identical raw answer in every trial of
    the Sonnet and Haiku number-only runs (identical first prompts)."""
    for src, key in ((SONNET_NO, "Sonnet"), (HAIKU_NO, "Haiku")):
        if src in complete_full:
            p1 = complete_main[(complete_main.source == src) & (complete_main.period == 1)].copy()
            p1["cell"] = list(zip(p1.q, p1.d, p1.signal, strict=False))
            nun = p1.groupby("cell").raw_text.nunique()
            M.set(f"{key}NODeterministicCells", str(int((nun == 1).sum())), int((nun == 1).sum()))
            M.set(f"{key}NOCells", str(len(nun)), len(nun))


def emit_api_spend(M: Macros) -> None:
    """Total API spend and the spend of the Haiku and Sonnet number-only runs (results/api_runs.csv)."""
    api = RES / "api_runs.csv"
    if not api.exists():
        return
    a = pd.read_csv(api)
    M.set(
        "ApiSpendUsd",
        fmt(float(a.cost_usd.astype(float).sum()), 2),
        float(a.cost_usd.astype(float).sum()),
    )
    for model, key in (("claude-haiku-5-5", "Haiku"), ("claude-sonnet-5-5", "Sonnet")):
        sel = a[(a.model == model) & (a.condition.astype(str) == "number-only")]
        M.set(
            f"{key}NOCostUsd",
            fmt(float(sel.cost_usd.astype(float).sum()), 2),
            float(sel.cost_usd.astype(float).sum()),
        )


def emit_parser_audit(
    M: Macros, long_main: pd.DataFrame, long_partial: pd.DataFrame | None
) -> None:
    """Parser facts for the open-weight runs: how many of the Qwen 14B answers came wrapped in
    text, and how many local answers overall were not a bare integer."""
    local = long_main[~long_main.source.str.startswith("claude")]
    if long_partial is not None:
        local = pd.concat(
            [local, long_partial[~long_partial.source.str.startswith("claude")]],
            ignore_index=True,
        )
    # the prompt-variant and greedy runs are audited too, although no table uses them
    for sub in ("variants", "greedy"):
        if any((RES / sub).glob("*.jsonl")):
            extra = load_results(RES / sub)
            local = pd.concat(
                [local, extra[~extra.source.str.startswith("claude")]], ignore_index=True
            )
    bare = local.raw_text.astype(str).str.match(r"^\s*\d{1,3}\s*$")
    q14 = long_main[long_main.source == "qwen2.5:14b [main, T0.7]"]
    n_wr = int((q14.parse_status == "ok_wrapped").sum()) if len(q14) else 0
    M.set("QwenFourteenBWrapped", f"{n_wr:,}", n_wr)
    M.set(
        "QwenFourteenBWrappedPct",
        pct(n_wr / len(q14), 1) if len(q14) else "0",
        n_wr / max(1, len(q14)),
    )
    M.set("ParserAuditRows", f"{len(local):,}", len(local))
    M.set("ParserAuditNonBare", f"{int((~bare).sum()):,}", int((~bare).sum()))
    M.set("ParserAuditNonBarePct", pct(float((~bare).mean()), 2), float((~bare).mean()))


def emit_api_usage(M: Macros, complete_full: list[str]) -> None:
    """Token and cost facts of the hosted runs, read from the raw JSONL rows: cache reads,
    output and thinking tokens, refusals and clamped answers."""
    for src, key, fname in (
        (HAIKU_NO, "HaikuNO", "claude-haiku-5-5__main__T1__behavioral__number-only.jsonl"),
        (SONNET_NO, "SonnetNO", "claude-sonnet-5-5__main__T1__behavioral__number-only.jsonl"),
    ):
        if src in complete_full:
            rows = pd.DataFrame(read_jsonl(RES / fname))
            M.set(
                f"{key}CacheReadPerCall",
                f"{rows.cache_read_input_tokens.mean():.0f}",
                float(rows.cache_read_input_tokens.mean()),
            )
            M.set(
                f"{key}OutputPerCall",
                fmt(float(rows.output_tokens.mean()), 1),
                float(rows.output_tokens.mean()),
            )
            M.set(
                f"{key}ThinkingTotal",
                f"{int(rows.thinking_tokens.sum()):,}",
                int(rows.thinking_tokens.sum()),
            )
            M.set(
                f"{key}Refusals",
                str(int((rows.stop_reason == "refusal").sum())),
                int((rows.stop_reason == "refusal").sum()),
            )
    # every complete hosted run: thinking tokens, cost, clamped answers
    for f_ in sorted(RES.glob("claude-*.jsonl")):
        rows_ = read_jsonl(f_)
        if len(rows_) < ROWS_PER_FULL_RUN:
            continue
        key_ = source_key(rows_[0]["source"])
        th = pd.Series([int(r.get("thinking_tokens") or 0) for r in rows_])
        cost_ = sum(float(r.get("cost_usd") or 0) for r in rows_)
        M.set(f"{key_}ThinkMean", f"{th.mean():.0f}", float(th.mean()))
        M.set(f"{key_}ThinkMedian", f"{th.median():.0f}", float(th.median()))
        M.set(f"{key_}ThinkMax", f"{th.max():,}", int(th.max()))
        M.set(f"{key_}ThinkZeroPct", pct(float((th == 0).mean()), 0), float((th == 0).mean()))
        M.set(f"{key_}CostTotalUsd", f"{cost_:.2f}", cost_)
        clamped = sum(1 for r in rows_ if str(r.get("parse_status", "")).startswith("clamped"))
        M.set(f"{key_}ClampedRows", str(clamped), clamped)


def emit_pilot_calls(M: Macros) -> None:
    """Number of calls of the Sonnet pilot with up-front thinking off, which stopped at a
    refusal. Its rows are not distributed; the count comes from the API run log."""
    api = RES / "api_runs.csv"
    if not api.exists():
        return
    a_ = pd.read_csv(api)
    a_ = a_[a_.spec_label.astype(str).str.endswith("think-off")]
    if len(a_):
        M.set("FailedPilotRows", str(int(a_.n_calls.iloc[0])), int(a_.n_calls.iloc[0]))


def emit_run_counts(M: Macros, long_main: pd.DataFrame, complete_full: list[str]) -> None:
    """How many models and runs are complete, the lowest usable-answer share among them, and
    the total number of model calls in results/."""
    M.set(
        "NLocalModels",
        str(len({parse_tag(s_) for s_ in complete_full if not s_.startswith("claude")})),
        None,
    )
    M.set(
        "NApiModels",
        str(len({parse_tag(s_) for s_ in complete_full if s_.startswith("claude")})),
        None,
    )
    M.set("NFullRuns", str(len(complete_full)), len(complete_full))
    n_models_any = len({parse_tag(s_) for s_ in complete_full})
    M.set("NModelsAny", str(n_models_any), n_models_any)
    usable_min = min(
        long_main[long_main.source == s_].parse_status.isin(OK_STATUSES).mean()
        for s_ in complete_full
    )
    M.set("CompletedMinUsable", pct(float(usable_min), 1), float(usable_min))
    local_tags = {
        parse_tag(s_)
        for s_ in long_main.source.unique()
        if not s_.startswith("claude") and not s_.startswith("human")
    }
    M.set("NLocalModelsAny", str(len(local_tags)), None)
    M.set("TotalModelCalls", f"{int(len(long_main)):,}", int(len(long_main)))


def emit_stopped_reasoning_run(M: Macros) -> None:
    """Facts of the Sonnet 5.5 reasoning-allowed run, which stopped at its first refusal:
    calls made, complete trials, subjects covered, thinking tokens and cost."""
    ra_part = (
        RES / "partial" / "claude-sonnet-5-5__main__T1__behavioral__reasoning__PARTIAL_s1-3.jsonl"
    )
    if not ra_part.exists():
        return
    rows_ = read_jsonl(ra_part)
    per_trial: dict = {}
    for r in rows_:
        per_trial[(r["subject"], r["trial"])] = per_trial.get((r["subject"], r["trial"]), 0) + 1
    n_complete = sum(1 for v in per_trial.values() if v == 10)
    subs_ = sorted({r["subject"] for r in rows_})
    th_ = pd.Series([int(r.get("thinking_tokens") or 0) for r in rows_])
    M.set("SonnetRAPCallsMade", f"{len(rows_) + 1:,}", len(rows_) + 1)  # + the refused call
    M.set("SonnetRAPRows", f"{len(rows_):,}", len(rows_))
    M.set("SonnetRAPCompleteTrials", str(n_complete), n_complete)
    M.set("SonnetRAPSubjectsCovered", f"{subs_[0]} to {subs_[-1]}", len(subs_))
    M.set("SonnetRAPThinkMean", f"{th_.mean():.0f}", float(th_.mean()))
    M.set("SonnetRAPCostUsd", f"{sum(float(r.get('cost_usd') or 0) for r in rows_):.2f}", 0)
    M.set("SonnetRAPRefusalCategory", "reasoning\\_extraction", 0)


# ------------------------------------------------------------------ condition tables


def cell_table_head() -> list[str]:
    """Header of a nine-column table with one cell per (q, d) condition."""
    return [
        "\\begin{tabular}{@{}l" + "r" * 9 + "@{}}",
        "\\toprule",
        " & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{$q$ = {q:g}}}" for q in Q_LEVELS)
        + " \\\\ \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        "source & " + " & ".join(f"$d$ = {d:g}" for _q, d in CELLS) + " \\\\",
        "\\midrule",
    ]


def table_first_evidence(
    M: Macros, complete_main: pd.DataFrame, human: dict, complete_full: list[str]
) -> pd.DataFrame:
    """Mean period-1 answer after a blue ball in every condition, humans and every complete run,
    against the Bayesian posterior. Returns the pooled per-period frame (model rows plus the
    humans) that the signed-error table reuses."""
    cols_ = ["source", "subject", "q", "d", "period", "signal", "estimate", "p_bayes"]
    hum_ = human["human_sub"].copy()
    hum_["source"] = "human"
    pooled = pd.concat([complete_main[cols_], hum_[cols_]], ignore_index=True)
    fe = (
        pooled[(pooled.period == 1) & (pooled.signal == BLUE)]
        .groupby(["source", "q", "d"])
        .agg(stated=("estimate", "mean"), bayes=("p_bayes", "mean"), n=("estimate", "size"))
        .reset_index()
    )
    fe.to_csv(DATA / "main" / "explore" / "first_evidence_period1_blue.csv", index=False)
    lines_ = cell_table_head()
    bayes_row = fe[fe.source == "human"].set_index(["q", "d"]).reindex(CELLS)
    lines_.append(
        "Bayesian posterior & " + " & ".join(f"{100 * v:.1f}" for v in bayes_row.bayes) + " \\\\"
    )
    for i_, (_q, _d) in enumerate(CELLS):
        M.set(
            f"BayesFirstBlueQ{NUM[i_ // 3]}D{NUM[i_ % 3]}",
            f"{100 * bayes_row.bayes.iloc[i_]:.1f}",
            float(bayes_row.bayes.iloc[i_]),
        )
    for src in ["human", *complete_full]:
        sub_ = fe[fe.source == src].set_index(["q", "d"]).reindex(CELLS)
        key_ = source_key(src)
        lines_.append(
            f"{latex_escape(row_label(src))} & "
            + " & ".join("--" if pd.isna(v) else f"{100 * v:.0f}" for v in sub_.stated)
            + " \\\\"
        )
        for i_, (_q, _d) in enumerate(CELLS):
            v = sub_.stated.iloc[i_]
            M.set(
                f"{key_}FirstBlueQ{NUM[i_ // 3]}D{NUM[i_ % 3]}",
                "--" if pd.isna(v) else f"{100 * v:.0f}",
                None if pd.isna(v) else float(v),
            )
        d1 = sub_.stated.xs(1.5, level="d")
        M.set(f"{key_}FirstBlueDOneMin", f"{100 * d1.min():.0f}", float(d1.min()))
        M.set(f"{key_}FirstBlueDOneMax", f"{100 * d1.max():.0f}", float(d1.max()))
    bd1 = bayes_row.bayes.xs(1.5, level="d")
    M.set("BayesFirstBlueDOneMin", f"{100 * bd1.min():.1f}", float(bd1.min()))
    M.set("BayesFirstBlueDOneMax", f"{100 * bd1.max():.1f}", float(bd1.max()))
    lines_ += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "first_evidence.tex").write_text("\n".join(lines_) + "\n")
    return pooled


def table_signed_error(pooled: pd.DataFrame, complete_full: list[str]) -> None:
    """Mean signed error (stated minus posterior, all periods) by condition for the humans,
    Sonnet 5 and Haiku 4.5."""
    se_df = pooled.assign(signed=pooled.estimate - pooled.p_bayes).dropna(subset=["signed"])
    se = se_df.groupby(["source", "q", "d"]).signed.mean()
    se_sources = ["human"] + [
        s_ for s_ in complete_full if s_.startswith(("claude-sonnet-5 [", "claude-haiku-4-5 ["))
    ]
    lines_ = cell_table_head()
    for src in se_sources:
        vals = [se.get((src, q_, d_), float("nan")) for q_, d_ in CELLS]
        lines_.append(
            f"{latex_escape(row_label(src))} & "
            + " & ".join("--" if pd.isna(v) else f"{v:+.3f}" for v in vals)
            + " \\\\"
        )
    lines_ += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "signed_error.tex").write_text("\n".join(lines_) + "\n")


def table_paired_distance(M: Macros, tables_main: dict, complete_full: list[str]) -> None:
    """Distance from the posterior with CI, the paired difference from the humans, the number
    of simulated subjects closer than their matched human, and the correlation with the humans."""
    acc_ = tables_main["accuracy_overall"].set_index("source")
    per_ = (
        tables_main["distance_per_subject"]
        .groupby(["source", "subject"])
        .abs_dev.mean()
        .unstack("source")
    )
    lines_ = [
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "source & distance [95\\% CI] & difference from humans [95\\% CI] & closer & $r$ \\\\",
        "\\midrule",
    ]
    hum_acc = acc_.loc["human"]
    co_r = tables_main["correlation_overall"].set_index("source")["r"]
    lines_.append(
        f"Humans & {hum_acc.mean_abs_dev:.3f} [{hum_acc.ci_low:.3f}, {hum_acc.ci_high:.3f}] & -- & -- & -- \\\\"
    )
    for src in complete_full:
        if src not in acc_.index:
            continue
        a_ = acc_.loc[src]
        n_closer = int((per_[src] < per_["human"]).sum()) if src in per_.columns else 0
        key_ = source_key(src)
        M.set(f"{key_}NCloser", str(n_closer), n_closer)
        lines_.append(
            f"{latex_escape(row_label(src))} & {a_.mean_abs_dev:.3f} [{a_.ci_low:.3f}, {a_.ci_high:.3f}] & {a_.diff_from_human:+.3f} [{a_.diff_ci_low:+.3f}, {a_.diff_ci_high:+.3f}] & {n_closer} & {(f'{co_r[src]:.2f}' if src in co_r.index else '--')} \\\\"
        )
    lines_ += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "paired_distance.tex").write_text("\n".join(lines_) + "\n")


# ------------------------------------------------------------------ rule-based agents

FAKE_DIR = RES / "fake"

# column heads of the fake-subject table
AGENT_COL = {
    "fake-minority": "minority",
    "fake-switchprob": "switch prob.",
    "fake-random30": "random 30",
    "fake-countup": "count-up",
    "fake-fixedstep": "fixed step",
    "fake-wrongbayes": "wrong $q$, $d$",
    "fake-bayesnoise": "Bayes + noise",
}
# row labels of the fake agents in the full sensitivity table
FAKE_ROW_LABELS = {
    "fake-minority": "minority share",
    "fake-switchprob": "switch probability",
    "fake-random30": "random around 30",
    "fake-countup": "count-up",
    "fake-fixedstep": "fixed step",
    "fake-wrongbayes": "Bayes with the wrong $q$, $d$",
    "fake-bayesnoise": "Bayes plus noise",
}


def table_fake_subjects(M: Macros, eff_all: pd.DataFrame) -> None:
    """One column per rule-based agent (and the humans): fitted and effective parameters, fit
    quality, distance from the posterior and signal response, from the summary that
    scripts/explore_fake_subjects.py writes."""
    ft = FAKE_DIR / "fake_subjects_table.csv"
    if not ft.exists():
        return
    fake_eff_path = FAKE_DIR / "fake_effective_parameters.csv"
    fake_eff = pd.read_csv(fake_eff_path) if fake_eff_path.exists() else pd.DataFrame()
    fdf = pd.read_csv(ft)
    agents = [r_ for _, r_ in fdf.iterrows() if r_["source"] != "human"]
    hum_rows = fdf[fdf.source == "human"]
    columns = ([hum_rows.iloc[0]] if len(hum_rows) else []) + agents
    head_ = [
        "Humans"
        if c["source"] == "human"
        else AGENT_COL.get(c["source"], latex_escape(c["source"]))
        for c in columns
    ]

    def eff_triplet(src_, which):
        df_ = eff_all if src_ == "human" else fake_eff
        if not len(df_):
            return "--"
        cand = [x for x in df_.source.unique() if x.split(" [")[0] == src_]
        if not cand:
            return "--"
        vals_ = eff_cells(df_, cand[0])
        return " / ".join(vals_[:3] if which == "q" else vals_[3:])

    def trip(src_, which):
        v = eff_triplet(src_, which)
        return v.split(" / ") if v != "--" else ["--", "--", "--"]

    rows_ = []
    for k_, lab in enumerate(
        ("$\\alpha$ at $q$ = 0.01", "$\\alpha$ at $q$ = 0.05", "$\\alpha$ at $q$ = 0.10")
    ):
        rows_.append(
            (lab, [f"{c[['alpha q=.01', 'alpha q=.05', 'alpha q=.10'][k_]]:.2f}" for c in columns])
        )
    for k_, lab in enumerate(
        ("$\\beta$ at $d$ = 1.5", "$\\beta$ at $d$ = 3", "$\\beta$ at $d$ = 9")
    ):
        rows_.append(
            (lab, [f"{c[['beta d=1.5', 'beta d=3', 'beta d=9'][k_]]:.2f}" for c in columns])
        )
    for k_, lab in enumerate(
        ("$\\alpha q$ at $q$ = 0.01", "$\\alpha q$ at $q$ = 0.05", "$\\alpha q$ at $q$ = 0.10")
    ):
        rows_.append((lab, [trip(c["source"], "q")[k_] for c in columns]))
    for k_, lab in enumerate(
        ("$d^{\\beta}$ at $d$ = 1.5", "$d^{\\beta}$ at $d$ = 3", "$d^{\\beta}$ at $d$ = 9")
    ):
        rows_.append((lab, [trip(c["source"], "d")[k_] for c in columns]))
    rows_ += [
        ("$R^2$, neglect model", [f"{c['R2 neglect']:.2f}" for c in columns]),
        ("$R^2$, Bayes", [str(c["R2 Bayes"]) for c in columns]),
        ("flagged", [str(c["flagged"]) for c in columns]),
        ("$\\alpha$ falls in $q$", [str(c["alpha falls in q"]) for c in columns]),
        ("$\\beta$ falls in $d$", [str(c["beta falls in d"]) for c in columns]),
        ("distance from the posterior", [f"{c['mean |answer - Bayes|']:.3f}" for c in columns]),
        (
            "signal response (blue $-$ red)",
            [f"{c['signal response (blue - red move)']:+.3f}" for c in columns],
        ),
    ]
    lines_ = [
        "\\begin{tabular}{@{}l" + "c" * len(columns) + "@{}}",
        "\\toprule",
        " & " + " & ".join(head_) + " \\\\",
        "\\midrule",
    ]
    for lab, vals_ in rows_:
        lines_.append(lab + " & " + " & ".join(vals_) + " \\\\")
    lines_ += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "fake_subjects.tex").write_text("\n".join(lines_) + "\n")
    for r_ in agents:
        key_ = fake_key(r_["source"])
        M.set(key_ + "RtwoNeglect", f"{r_['R2 neglect']:.2f}", float(r_["R2 neglect"]))
        M.set(key_ + "RtwoBayes", str(r_["R2 Bayes"]), None)
        M.set(
            key_ + "Dev",
            f"{r_['mean |answer - Bayes|']:.3f}",
            float(r_["mean |answer - Bayes|"]),
        )
        M.set(
            key_ + "Signal",
            f"{r_['signal response (blue - red move)']:+.3f}",
            float(r_["signal response (blue - red move)"]),
        )
        M.set(key_ + "Flagged", str(r_["flagged"]), None)
        M.set(key_ + "AlphaFalls", str(r_["alpha falls in q"]), None)
        M.set(key_ + "BetaFalls", str(r_["beta falls in d"]), None)
        for nm, c_ in zip(
            ("AlphaOne", "AlphaTwo", "AlphaThree", "BetaOne", "BetaTwo", "BetaThree"),
            ("alpha q=.01", "alpha q=.05", "alpha q=.10", "beta d=1.5", "beta d=3", "beta d=9"),
            strict=True,
        ):
            M.set(key_ + nm, f"{r_[c_]:.2f}", float(r_[c_]))
    M.set(
        "NFakeAgents",
        str(int((fdf.source != "human").sum())),
        int((fdf.source != "human").sum()),
    )


# ------------------------------------------------------------------ summary numbers


def emit_signal_response(M: Macros, sane_main: dict, complete_full: list[str]) -> None:
    """Model-free signal response (mean move after a blue ball minus after a red ball), humans
    and every complete run."""
    sr_ = sane_main["signal_response"].copy()
    sr_ = sr_[sr_["d"].astype(str) == "all"] if (sr_["d"].astype(str) == "all").any() else sr_
    for src in ["human", *complete_full]:
        b_ = sr_[(sr_.source == src) & (sr_.after == "blue")]["mean_delta"]
        r_ = sr_[(sr_.source == src) & (sr_.after == "red")]["mean_delta"]
        if len(b_) and len(r_):
            v = float(b_.mean() - r_.mean())
            M.set(("Hum" if src == "human" else source_key(src)) + "SignalResponse", f"{v:+.3f}", v)


def emit_parameter_notes(M: Macros, tables_main: dict) -> None:
    """Numbers quoted in the text: the range of beta at d = 9 across the three mid-size
    open-weight models, and the mean alpha of the Sonnet 5.5 number-only run."""
    pm_ = tables_main["parameters"].pivot(index="source", columns="name", values="mean")
    mid3 = [
        s_
        for s_ in ("qwen2.5:14b [main, T0.7]", "llama3.1:8b [main, T0.7]", "gemma2:9b [main, T0.7]")
        if s_ in pm_.index
    ]
    if mid3:
        b3 = {s_: float(pm_.loc[s_, "beta_3"]) for s_ in mid3}
        lo_s, hi_s = min(b3, key=b3.get), max(b3, key=b3.get)
        M.set("MidBetaThreeMin", fmt(b3[lo_s]), b3[lo_s])
        M.set("MidBetaThreeMax", fmt(b3[hi_s]), b3[hi_s])
        M.set("MidBetaThreeMinModel", short_name(lo_s).replace(" (number only)", ""), None)
        M.set("MidBetaThreeMaxModel", short_name(hi_s).replace(" (number only)", ""), None)
    if SONNET_NO in pm_.index:
        am = float(np.mean([pm_.loc[SONNET_NO, f"alpha_{i}"] for i in (1, 2, 3)]))
        M.set("SonnetNOAlphaMeanPct", f"{100 * am:.0f}", am)


# ------------------------------------------------------------------ system sensitivity


def compute_sensitivity(
    tables_main: dict, complete_full: list[str], partial, partial_incomplete: set[str]
):
    """Per-subject (s_q, s_d) from the fits of the humans, every complete run, the stateless and
    reasoning-allowed partial runs, the 15-subject multi-turn subsets and the rule-based agents,
    summarised with a subject bootstrap and paired differences from the humans.

    Returns ``(sens, variant_sources, fake_sources)``: the summary table, the partial sources
    that go into the full sensitivity table and the agent sources.
    """
    fits_main = tables_main["fits"]
    fits_all = [fits_main[fits_main.source.isin(set(complete_full) | {"human"})]]
    variant_sources: list[str] = []
    if partial is not None:
        _, cp_, tp_, _ = partial
        fp_ = tp_["fits"]
        variant_sources = [
            s_
            for s_ in cp_.source.unique()
            if s_ not in partial_incomplete and ("stateless" in s_ or "reasoning allowed" in s_)
        ]
        subset15_ = [
            s_
            for s_ in cp_.source.unique()
            if s_ not in partial_incomplete and "[main, T1, number only, PARTIAL s1-15]" in s_
        ]
        fits_all.append(fp_[fp_.source.isin(variant_sources + subset15_)])
    fake_fits_path = FAKE_DIR / "analysis" / "fake_fits.csv"
    fake_sources: list[str] = []
    if fake_fits_path.exists():
        ff_ = pd.read_csv(fake_fits_path)
        ff_ = ff_[~ff_.source.str.startswith("human")]
        fake_sources = sorted(ff_.source.unique())
        fits_all.append(ff_)
    sens_per = fitted_pairs(pd.concat(fits_all, ignore_index=True))
    sens = summarise(sens_per, ["s_q", "s_d"])
    sens.to_csv(DATA / "main" / "explore" / "sensitivity_summary.csv", index=False)
    sens_per.to_csv(DATA / "main" / "explore" / "sensitivity_per_subject.csv", index=False)
    return sens, variant_sources, fake_sources


def emit_sensitivity(M: Macros, sens: pd.DataFrame) -> None:
    """s_q and s_d with CI for every row of the summary, plus the paired difference from the
    humans; then the model-free version for the rule-based agents."""
    for _, r_ in sens.iterrows():
        k_ = sens_key(r_.source)
        for col, nm in (("s_q", "Sq"), ("s_d", "Sd")):
            M.set(f"{k_}{nm}", fmt(r_[col]), float(r_[col]))
            M.set(f"{k_}{nm}Lo", fmt(r_[f"{col}_lo"]), float(r_[f"{col}_lo"]))
            M.set(f"{k_}{nm}Hi", fmt(r_[f"{col}_hi"]), float(r_[f"{col}_hi"]))
            if r_.source != "human" and f"{col}_diff" in r_ and pd.notna(r_[f"{col}_diff"]):
                M.set(f"{k_}{nm}Diff", f"{r_[f'{col}_diff']:+.2f}", float(r_[f"{col}_diff"]))
                M.set(
                    f"{k_}{nm}DiffLo", f"{r_[f'{col}_diff_lo']:+.2f}", float(r_[f"{col}_diff_lo"])
                )
                M.set(
                    f"{k_}{nm}DiffHi", f"{r_[f'{col}_diff_hi']:+.2f}", float(r_[f"{col}_diff_hi"])
                )
    if FAKE_DIR.exists() and any(FAKE_DIR.glob("*.jsonl")):
        fr_ = an.add_bayes(load_results(FAKE_DIR))
        free_ = free_pairs(
            fr_[["source", "subject", "trial", "period", "signal", "q", "d", "estimate", "p_bayes"]]
        )
        free_s = free_.groupby("source")[["s_q_free", "s_d_free"]].mean()
        for src_, r_ in free_s.iterrows():
            k_ = sens_key(src_)
            M.set(f"{k_}SqFree", fmt(r_.s_q_free), float(r_.s_q_free))
            M.set(f"{k_}SdFree", fmt(r_.s_d_free), float(r_.s_d_free))


def table_sensitivity(
    M: Macros,
    sens: pd.DataFrame,
    tables_main: dict,
    complete_full: list[str],
    variant_sources: list[str],
    fake_sources: list[str],
) -> None:
    """The two sensitivity tables (main runs; agents and protocol variants), the Spearman
    correlations between distance and sensitivity, and the two sensitivity-map figures."""

    def sens_row(src: str, label: str) -> str:
        r_ = sens.set_index("source").loc[src]
        cells = [f"{r_.s_q:.2f} [{r_.s_q_lo:.2f}, {r_.s_q_hi:.2f}]"]
        cells.append(
            "--"
            if src == "human"
            else f"{r_.s_q_diff:+.2f} [{r_.s_q_diff_lo:+.2f}, {r_.s_q_diff_hi:+.2f}]"
        )
        cells.append(f"{r_.s_d:.2f} [{r_.s_d_lo:.2f}, {r_.s_d_hi:.2f}]")
        cells.append(
            "--"
            if src == "human"
            else f"{r_.s_d_diff:+.2f} [{r_.s_d_diff_lo:+.2f}, {r_.s_d_diff_hi:+.2f}]"
        )
        return f"{latex_escape(label)} & " + " & ".join(cells) + " \\\\"

    head = [
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "source & $s_q$ [95\\% CI] & difference from humans & $s_d$ [95\\% CI] & difference from humans \\\\",
        "\\midrule",
    ]
    in_sens = set(sens.source)
    main_srcs = [
        "human",
        *[s_ for s_ in complete_full if "reasoning allowed" not in s_ and "stateless" not in s_],
    ]
    lines_ = (
        head
        + [
            sens_row(s_, "Humans" if s_ == "human" else row_label(s_))
            for s_ in main_srcs
            if s_ in in_sens
        ]
        + ["\\bottomrule", "\\end{tabular}"]
    )
    (TABLES / "sensitivity.tex").write_text("\n".join(lines_) + "\n")
    extra_srcs = (
        [s_ for s_ in complete_full if "reasoning allowed" in s_]
        + [s_ for s_ in complete_full if "stateless" in s_]
        + order_sources(variant_sources)
    )
    lines_ = (
        head
        + [
            sens_row(s_, FAKE_ROW_LABELS.get(s_.split(" [")[0], s_))
            for s_ in fake_sources
            if s_ in in_sens
        ]
        + ["\\bottomrule", "\\end{tabular}"]
    )
    (TABLES / "sensitivity_agents.tex").write_text("\n".join(lines_) + "\n")
    lines_ = (
        head
        + [sens_row(s_, row_label(s_)) for s_ in extra_srcs if s_ in in_sens]
        + ["\\bottomrule", "\\end{tabular}"]
    )
    (TABLES / "sensitivity_variants.tex").write_text("\n".join(lines_) + "\n")

    acc_s = tables_main["accuracy_overall"].set_index("source")
    sens_i = sens.set_index("source")
    corr_srcs = [s_ for s_ in main_srcs if s_ in acc_s.index and s_ in sens_i.index]
    dev_v = np.array([acc_s.loc[s_, "mean_abs_dev"] for s_ in corr_srcs])
    sq_v = np.array([sens_i.loc[s_, "s_q"] for s_ in corr_srcs])
    sd_v = np.array([sens_i.loc[s_, "s_d"] for s_ in corr_srcs])
    for nm, v in (("Sq", sq_v), ("Sd", sd_v), ("Mean", (sq_v + sd_v) / 2)):
        rho = float(spearmanr(dev_v, v).correlation)
        M.set(f"SpearmanDev{nm}", f"{rho:+.2f}", rho)
    M.set("NSpearman", str(len(corr_srcs)), len(corr_srcs))
    pf.fig_sensitivity_map(
        sens, FIGS / "fig_sensitivity_map", main_srcs, inset=(-0.08, 0.42, -0.22, 0.26)
    )
    pf.fig_sensitivity_map(
        sens,
        FIGS / "fig_sensitivity_full",
        main_srcs + extra_srcs,
        all_fakes=True,
        height=4.8,
        inset=(-0.08, 0.42, -0.22, 0.26),
    )
    M.set("NFakeAgentsSens", str(len(fake_sources)), len(fake_sources))


# ------------------------------------------------------------------ robustness, appendix


def write_robustness_tables(
    M: Macros,
    partial,
    partial_incomplete: set[str],
    tables_main: dict,
    sane_main: dict,
    robustness_full: list[str],
    human: dict,
    sens: pd.DataFrame,
) -> None:
    """partial_results.tex (every complete partial run plus the reasoning-allowed full runs,
    without the effective-parameter columns) and history_15.tex."""
    _, cp, tp, sp = partial
    if partial_incomplete:
        print(f"in progress, excluded from the robustness tables: {sorted(partial_incomplete)}")
    cp = cp[~cp.source.isin(partial_incomplete)]
    tp = drop_sources(tp, partial_incomplete)
    sp = drop_sources(sp, partial_incomplete)
    srcs = ["human", *order_sources(cp.source.unique())]
    pr = TABLES / "partial_results.tex"
    table_main(tp, sp, srcs, pr)
    table_history(M, cp, tp, human, sens, TABLES / "history_15.tex")
    if robustness_full:
        # the reasoning-allowed full run(s) join the robustness table (own fits, same columns)
        body = "\n".join(table_main_rows(tables_main, sane_main, robustness_full))
        txt = pr.read_text().replace("\n\\bottomrule", "\n\\midrule\n" + body + "\n\\bottomrule", 1)
        pr.write_text(txt)
    compact_columns(pr)


def table_prompts(human: dict) -> None:
    """Verbatim prompts: system message, the multi-turn turns of an example trial, the stateless
    prompt, the two prompt variants and the JSON answer schema."""
    trial = human["trials_by_subject"][1][4]
    lines = ["% GENERATED: verbatim prompts (subject 1, trial 5 as the example)"]
    lines.append("\\paragraph{System message (all conditions).}")
    lines.append("\\begin{lstlisting}\n" + prompts.system_prompt("v1") + "\n\\end{lstlisting}")
    msgs = prompts.build_main_messages(trial, 2, ["50"], "v1")
    lines.append(
        "\\paragraph{Multi-turn protocol, example trial ($q$ = "
        + f"{trial.q:g}"
        + ", $d$ = "
        + f"{trial.d:g}"
        + "): period 1 user turn, the replayed assistant turn, period 2 user turn.}"
    )
    for m in msgs[1:]:
        lines.append("\\noindent\\texttt{" + m["role"] + "}:")
        lines.append("\\begin{lstlisting}\n" + m["content"] + "\n\\end{lstlisting}")
    st = prompts.build_stateless_messages(trial, 3, "v1")
    lines.append("\\paragraph{Stateless protocol (period 3 shown).}")
    lines.append("\\begin{lstlisting}\n" + st[-1]["content"] + "\n\\end{lstlisting}")
    for variant in ("ratio-words", "no-horizon"):
        lines.append("\\paragraph{Prompt variant \\texttt{" + variant + "}: first user turn.}")
        vm = prompts.build_main_messages(trial, 1, [], variant)
        lines.append("\\begin{lstlisting}\n" + vm[-1]["content"] + "\n\\end{lstlisting}")
    lines.append(
        "\\paragraph{JSON answer schema (Claude runs and the Qwen2.5 14B number-only run).}"
    )
    lines.append(
        "\\begin{lstlisting}\n" + json.dumps(ANSWER_SCHEMA, indent=1) + "\n\\end{lstlisting}"
    )
    text = "\n".join(lines) + "\n"
    # run-in \paragraph headings collide with the listing frame: use a standalone bold line
    text = re.sub(
        r"\\paragraph\{(.+)\}$",
        r"\\medskip\\noindent\\textbf{\1}\\par\\nopagebreak",
        text,
        flags=re.M,
    )
    (TABLES / "prompts.tex").write_text(text)


def table_models(long_main: pd.DataFrame) -> None:
    """One row per model: how it was served, the Ollama digest or the API model id, the
    quantisation and the parameter count."""
    rows = []
    info = RES / "model_info.csv"
    mi = pd.read_csv(info) if info.exists() else pd.DataFrame()
    seen = set()
    for src in order_sources(long_main.source.unique()):
        tag = parse_tag(src)
        if tag in seen or tag == "human":
            continue
        seen.add(tag)
        g = long_main[long_main.source == src]
        dig = "; ".join(sorted({str(x)[:12] for x in g.model_digest.dropna().unique()}))
        r = mi[mi.tag == tag].iloc[0] if len(mi) and (mi.tag == tag).any() else None
        if tag.startswith("claude"):
            fjson = next(iter(RES.glob(f"{tag}__main__*.jsonl")), None)
            api_id = read_jsonl(fjson)[0].get("api_model", "") if fjson else ""
            rows.append(
                f"\\texttt{{{latex_escape(tag)}}} & API & id as returned: "
                f"\\texttt{{{latex_escape(api_id)}}} & -- & -- \\\\"
            )
        else:
            rows.append(
                f"\\texttt{{{latex_escape(tag)}}} & Ollama & \\texttt{{{dig}}} & {latex_escape(r.quantization) if r is not None else '--'} & {latex_escape(r.parameter_size) if r is not None else '--'} \\\\"
            )
    (TABLES / "models.tex").write_text(
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\nmodel tag & served by & digest / API id & quantisation & parameters \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def check_no_leak(in_progress: list[str]) -> None:
    """Stop if an in-progress source appears in any generated table."""
    for tex in TABLES.glob("*.tex"):
        body = tex.read_text()
        for s_ in in_progress:
            if latex_escape(s_) in body or s_ in body:
                raise SystemExit(f"in-progress source {s_!r} leaked into {tex.name}")


def make_figures(
    M: Macros, tables_main: dict, sane_main: dict, eff_all: pd.DataFrame, complete_full: list[str]
) -> int:
    """The paper figures (regime_shift.paper_figures), PDF and PNG, from the same frames as the
    tables: the complete 30-subject runs plus the humans and the Bayesian observer. Returns the
    number of figures made."""
    fig_data = {
        "accuracy": tables_main["accuracy_overall"],
        "eff": eff_all,
        "parameters": tables_main["parameters"],
        "io": tables_main["io_summary"],
        "dist": tables_main["distance_summary"],
        "calibration": sane_main["calibration"],
        "period_means": sane_main["period_means"],
        "signal": sane_main["signal_response"],
    }
    made = pf.make_all(fig_data, FIGS, complete_full)
    M.set("IOVmax", f"{made['io'][2]:g}", float(made["io"][2]))
    # one panel per d level of the 3B raw-answer histogram (q = 0.05)
    counts = sane_main["raw_answer_counts"]
    src3 = "qwen2.5:3b [main, T0.7]"
    if src3 in complete_full:
        cdf = counts[
            (counts.source == src3) & counts.condition.astype(str).isin(["4", "5", "6"])
        ].copy()
        cdf["q"] = 0.05
        cdf["d"] = cdf.condition.astype(int).map({4: 1.5, 5: 3.0, 6: 9.0})
        cdf["n"] = cdf["count"]
        pf.fig_hist_3b_by_d(cdf, src3, FIGS / "fig_hist_3b_by_d")
    return len(made)


# ------------------------------------------------------------------ main


def write_summary_csv(sens: pd.DataFrame, table_sets: list[dict], path: Path) -> None:
    """One row per source: subjects, the two sensitivities and distance with their intervals,
    alpha and beta by level, and the fit's R^2. The quickest view of the results."""
    acc = pd.concat([t["accuracy_overall"] for t in table_sets if "accuracy_overall" in t])
    par = pd.concat([t["parameters"] for t in table_sets if "parameters" in t])
    fq = pd.concat([t["fit_quality"] for t in table_sets if "fit_quality" in t])
    acc = acc.drop_duplicates("source").set_index("source")
    fq = fq.drop_duplicates("source").set_index("source")
    pm = par.drop_duplicates(["source", "name"]).pivot(
        index="source", columns="name", values="mean"
    )
    rows = []
    for r in sens.itertuples():
        src = r.source
        row = {
            "source": src,
            "n_subjects": int(r.n_subjects),
            "s_q": r.s_q,
            "s_q_lo": r.s_q_lo,
            "s_q_hi": r.s_q_hi,
            "s_d": r.s_d,
            "s_d_lo": r.s_d_lo,
            "s_d_hi": r.s_d_hi,
            "distance": acc.mean_abs_dev.get(src),
            "distance_lo": acc.ci_low.get(src),
            "distance_hi": acc.ci_high.get(src),
        }
        for name, col in zip(
            ("alpha_1", "alpha_2", "alpha_3"),
            ("alpha_q0.01", "alpha_q0.05", "alpha_q0.10"),
            strict=True,
        ):
            row[col] = pm[name].get(src) if name in pm else None
        for name, col in zip(
            ("beta_1", "beta_2", "beta_3"), ("beta_d1.5", "beta_d3", "beta_d9"), strict=True
        ):
            row[col] = pm[name].get(src) if name in pm else None
        row["r2"] = fq.r2_neglect_mean.get(src) if "r2_neglect_mean" in fq else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")


def main() -> int:
    for d in (PAPER, TABLES, FIGS, DATA):
        d.mkdir(parents=True, exist_ok=True)
    M = Macros()
    human = pipeline.prepare_human("data/human", "behavioral")
    emit_task_facts(M, human)

    # ---- the result folders: per-source macros
    long_main, complete_main, tables_main, sane_main, complete_full, in_progress = (
        emit_main_sources(M, human)
    )
    partial, partial_incomplete = emit_partial_sources(M, human)
    long_partial = partial[0] if partial is not None else None

    # ---- facts about the runs
    emit_calibration_bins(M, sane_main)
    emit_api_determinism(M, complete_main, complete_full)
    emit_api_spend(M)
    emit_model_info(M)
    emit_parser_audit(M, long_main, long_partial)
    emit_api_usage(M, complete_full)
    emit_pilot_calls(M)
    emit_run_counts(M, long_main, complete_full)

    # ---- condition tables, rule-based agents, summary numbers
    pooled = table_first_evidence(M, complete_main, human, complete_full)
    table_signed_error(pooled, complete_full)
    table_paired_distance(M, tables_main, complete_full)
    eff_all = sanity.effective_parameters(tables_main["fits"])
    table_fake_subjects(M, eff_all)
    emit_signal_response(M, sane_main, complete_full)
    emit_parameter_notes(M, tables_main)

    # ---- system sensitivity (s_q, s_d)
    sens, variant_sources, fake_sources = compute_sensitivity(
        tables_main, complete_full, partial, partial_incomplete
    )
    emit_sensitivity(M, sens)
    fake_tables = {}
    for name in ("accuracy_overall", "parameters", "fit_quality"):
        f_ = FAKE_DIR / "analysis" / f"fake_{name}.csv"
        if f_.exists():
            fake_tables[name] = pd.read_csv(f_)
    write_summary_csv(
        sens,
        [tables_main, *([partial[2]] if partial is not None else []), fake_tables],
        RES / "summary.csv",
    )
    table_sensitivity(M, sens, tables_main, complete_full, variant_sources, fake_sources)
    emit_stopped_reasoning_run(M)

    # ---- main tables: the number-only runs; the reasoning-allowed run goes to the robustness table
    robustness_full = [s_ for s_ in complete_full if "reasoning allowed" in s_]
    complete_full = [s_ for s_ in complete_full if "reasoning allowed" not in s_]
    table_main(tables_main, sane_main, ["human", *complete_full], TABLES / "main_results.tex")
    compact_columns(TABLES / "main_results.tex")
    table_state_cal(sane_main, ["human", *complete_full], TABLES / "state_calibration.tex")
    table_copying(sane_main, ["human", *complete_full], TABLES / "copying.tex")
    if partial is not None:
        write_robustness_tables(
            M, partial, partial_incomplete, tables_main, sane_main, robustness_full, human, sens
        )
    table_prompts(human)
    table_models(long_main)
    check_no_leak(in_progress)

    # ---- figures
    n_figures = make_figures(M, tables_main, sane_main, eff_all, complete_full)
    M.set("FigureCount", str(n_figures), n_figures)
    M.set("GeneratedAt", time.strftime("%Y-%m-%d %H:%M"), time.strftime("%Y-%m-%d %H:%M"))
    M.write()
    print(f"wrote {len(M.values)} macros to paper/numbers.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())

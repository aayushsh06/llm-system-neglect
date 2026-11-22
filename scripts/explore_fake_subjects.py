"""Rule-based agents that plainly do not do the task, run through the same system-neglect fit on
the same 30 subjects' behavioural sequences as every model.

    .venv/bin/python scripts/explore_fake_subjects.py

Writes one results/fake/synthetic__<agent>__main__T0.7__behavioral.jsonl per agent (seeded; a
rerun with the files in place resumes and changes nothing), the analysis CSVs under
results/fake/analysis/, the summary results/fake/fake_subjects_table.csv and
results/fake/fake_effective_parameters.csv that the paper builder reads, and
results/fake/FAKE_SUBJECTS.md with the same summary as a Markdown table. No model calls.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_shift import pipeline  # noqa: E402
from regime_shift.bayes import forward_filter  # noqa: E402
from regime_shift.collection.backends import ChatResponse  # noqa: E402
from regime_shift.collection.runner import run_spec  # noqa: E402
from regime_shift.config import RunSpec  # noqa: E402
from regime_shift.prompts import parse_prompt_state  # noqa: E402
from regime_shift.results import load_results  # noqa: E402
from regime_shift.sanity import effective_parameters  # noqa: E402
from regime_shift.task import BLUE, RED  # noqa: E402

OUT = ROOT / "results" / "fake"


class Fake:
    """A chat backend that answers with ``rule(q, d, signals, rng)``, rounded and clipped to
    0..99. The rule reads the task state back from the prompt; ``rng`` is seeded per call, so
    every agent is deterministic."""

    def __init__(self, name: str, rule) -> None:
        self.name = name
        self.rule = rule

    def chat(
        self, messages: Sequence[dict], *, temperature: float, seed: int, num_predict: int
    ) -> ChatResponse:
        q, d, signals = parse_prompt_state(messages)
        rng = np.random.default_rng(seed)
        v = int(np.clip(int(round(self.rule(q, d, signals, rng))), 0, 99))
        return ChatResponse(str(v), self.name, "synthetic", 0.0, 1)

    def digest(self) -> str | None:
        return "synthetic"


# agent tag -> (description, rule); the description is the "what it does" column of the summary
AGENTS = {
    "fake-minority": (
        "always the printed minority share (40/25/10)",
        lambda q, d, s, r: 100.0 / (1.0 + d),
    ),
    "fake-switchprob": (
        "always the printed switch probability (1/5/10)",
        lambda q, d, s, r: 100.0 * q,
    ),
    "fake-random30": (
        "random around 30, sd 10, clipped to 0-99, no use of the balls",
        lambda q, d, s, r: r.normal(30.0, 10.0),
    ),
    "fake-countup": (
        "counts up by 9 per period regardless of the balls (9, 18, ... 90)",
        lambda q, d, s, r: 9.0 * len(s),
    ),
    "fake-fixedstep": (
        "starts at 5, +12 per blue ball, -6 per red ball, clipped, regardless of q and d",
        lambda q, d, s, r: (
            5.0 + 12.0 * sum(1 for x in s if x == BLUE) - 6.0 * sum(1 for x in s if x == RED)
        ),
    ),
    "fake-bayesnoise": (
        "exact Bayesian filter with the true q and d, plus N(0, 3) points of noise, rounded and "
        "clipped",
        lambda q, d, s, r: 100.0 * forward_filter(q, d, s)[-1] + r.normal(0.0, 3.0),
    ),
    "fake-wrongbayes": (
        "exact Bayesian filter but always with q = 0.05, d = 3",
        lambda q, d, s, r: 100.0 * forward_filter(0.05, 3.0, s)[-1],
    ),
}


def md_table(df: pd.DataFrame) -> str:
    """A DataFrame as a Markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def signal_response(long: pd.DataFrame, human_sub: pd.DataFrame) -> dict[str, float]:
    """Mean change in the stated probability after a blue ball minus after a red ball
    (periods 2-10), per source; the human row is computed the same way."""
    out: dict[str, float] = {}
    frames = [long[["source", "subject", "trial", "period", "signal", "estimate"]]]
    h = human_sub.copy()
    if "estimate" not in h.columns and "response" in h.columns:
        h["estimate"] = h["response"]
    h["source"] = "human"
    frames.append(h[["source", "subject", "trial", "period", "signal", "estimate"]])
    df = pd.concat(frames, ignore_index=True).dropna(subset=["estimate"])
    df = df.sort_values(["source", "subject", "trial", "period"])
    df["prev"] = df.groupby(["source", "subject", "trial"])["estimate"].shift(1)
    df = df[df.period > 1].dropna(subset=["prev"])
    df["move"] = df["estimate"] - df["prev"]
    for src, g in df.groupby("source"):
        blue = g[g.signal == BLUE]["move"].mean()
        red = g[g.signal == RED]["move"].mean()
        out[src] = float(blue - red)
    return out


def main() -> int:
    human = pipeline.prepare_human("data/human", "behavioral")
    # one run per agent, same seed, subjects and trial set as the model runs
    tbs = human["trials_by_subject"]
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, (_desc, rule) in AGENTS.items():
        spec = RunSpec(
            model=tag,
            mode="main",
            temperature=0.7,
            seed=20251120,
            n_subjects=30,
            trial_set="behavioral",
            num_predict=8,
            synthetic=True,
        )
        run_spec(spec, Fake(tag, rule), tbs, OUT, parallel=1, progress=None)
    # the standard analysis, then one summary row per agent (and the humans)
    long = load_results(OUT)
    tables = pipeline.analyze(
        human["human_sub"], long, OUT / "analysis", human_all=human["human_all"], label="fake"
    )
    params = tables["parameters"].pivot(index="source", columns="name", values="mean")
    fq = tables["fit_quality"].set_index("source")
    acc = tables["accuracy_overall"].set_index("source")
    sig = signal_response(long, human["human_sub"])
    rows = []
    for src in params.index:
        if src not in fq.index:
            continue
        a = [params.loc[src, f"alpha_{i}"] for i in (1, 2, 3)]
        b = [params.loc[src, f"beta_{i}"] for i in (1, 2, 3)]
        r2n = (
            fq.loc[src, "r2_neglect_mean"]
            if "r2_neglect_mean" in fq.columns
            else fq.loc[src].filter(like="r2").iloc[0]
        )
        r2b = fq.loc[src, "r2_bayes_mean"] if "r2_bayes_mean" in fq.columns else float("nan")
        conv = fq.loc[src, "fit_success_share"]
        flag = bool(fq.loc[src, "do_not_interpret"])
        dev = acc.loc[src, "mean_abs_dev"] if src in acc.index else float("nan")
        tag = src.split(" [")[0]
        desc = "humans" if tag == "human" else AGENTS.get(tag, ("", None))[0]
        rows.append(
            {
                "source": tag,
                "what it does": desc,
                "alpha q=.01": round(a[0], 2),
                "alpha q=.05": round(a[1], 2),
                "alpha q=.10": round(a[2], 2),
                "beta d=1.5": round(b[0], 2),
                "beta d=3": round(b[1], 2),
                "beta d=9": round(b[2], 2),
                "R2 neglect": round(float(r2n), 2),
                "R2 Bayes": round(float(r2b), 2) if r2b == r2b else "",
                "converged": f"{100 * conv:.0f}%",
                "flagged": "yes" if flag else "no",
                "alpha falls in q": "yes" if a[0] > a[1] > a[2] else "no",
                "beta falls in d": "yes" if b[0] > b[1] > b[2] else "no",
                "mean |answer - Bayes|": round(float(dev), 3),
                "signal response (blue - red move)": round(sig.get(src, float("nan")), 3),
            }
        )
    df = pd.DataFrame(rows)
    md = [
        "# Rule-based agents",
        "",
        "Agents that do not do the task, on the 30 subjects' behavioural sequences (72 trials x 10 "
        "periods each), fitted with the same system-neglect code as every other source. Human "
        "ordering: alpha falling in q, beta falling in d.",
        "",
        md_table(df),
    ]
    (OUT / "FAKE_SUBJECTS.md").write_text("\n".join(md) + "\n")
    df.to_csv(OUT / "fake_subjects_table.csv", index=False)
    effective_parameters(tables["fits"]).to_csv(OUT / "fake_effective_parameters.csv", index=False)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

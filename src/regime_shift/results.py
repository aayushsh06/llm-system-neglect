"""Load JSONL run outputs into the long analysis table (one row per subject, trial, period)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .collection.runner import read_jsonl
from .parsing import parse_estimate
from .task import N_PERIODS

MODEL_COLUMNS = [
    "source",
    "subject",
    "session",
    "block",
    "trial",
    "trial_in_block",
    "condition",
    "q",
    "d",
    "period",
    "signal",
    "estimate",
    "raw_text",
    "parse_status",
    "mode",
    "model_tag",
    "model_digest",
    "temperature",
    "seed",
    "run_label",
    "prompt_version",
    "synthetic",
]


def _one_source_per_file(rows: list[dict]) -> list[dict]:
    """Give every row of one file the file's majority ``source`` label.

    A file holds exactly one run, so a second label can only be a bookkeeping slip in
    the writer (the rows themselves are correct); only the label is harmonised.
    """
    labels = [r.get("source") for r in rows]
    distinct = set(labels)
    if len(distinct) <= 1:
        return rows
    majority = max(distinct, key=labels.count)
    return [r if r.get("source") == majority else {**r, "source": majority} for r in rows]


def load_results(paths: str | Path | list[Path]) -> pd.DataFrame:
    """Read one or more JSONL files (or a directory of them) and re-parse the raw text.

    Duplicate keys keep the last record. ``estimate`` is the parsed probability in [0, 1]
    (NaN when unparsable).
    """
    if isinstance(paths, str | Path):
        p = Path(paths)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    else:
        files = [Path(x) for x in paths]
    rows: list[dict] = []
    for f in files:
        rows.extend(_one_source_per_file(read_jsonl(f)))
    if not rows:
        return pd.DataFrame(columns=MODEL_COLUMNS)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates("key", keep="last")
    est = np.full(len(df), np.nan)
    status = []
    for i, text in enumerate(df["raw_text"]):
        pr = parse_estimate(text)
        status.append(pr.status)
        est[i] = pr.estimate
    df["estimate"] = est
    df["parse_status"] = status
    for c in MODEL_COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[MODEL_COLUMNS].sort_values(["source", "subject", "trial", "period"])
    return df.reset_index(drop=True)


def complete_trials_only(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into trials with all 10 periods present and a report of incomplete ones.

    The fitting code needs whole trials. Incomplete trials (a conversation that
    was interrupted and not resumed) are returned separately so they can be
    reported rather than silently dropped.
    """
    key = ["source", "subject", "trial"]
    counts = df.groupby(key).period.nunique().rename("n_periods").reset_index()
    incomplete = counts[counts.n_periods < N_PERIODS]
    if len(incomplete):
        bad = set(map(tuple, incomplete[key].to_numpy()))
        mask = [tuple(x) not in bad for x in df[key].to_numpy()]
        df = df[mask]
    return df.reset_index(drop=True), incomplete

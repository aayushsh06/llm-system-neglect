"""End-to-end steps shared by the run scripts and the paper build.

prepare_human -> run_gate -> simulate -> analyze
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import analysis as an
from .collection.backends import Backend
from .collection.runner import RunStats, run_spec
from .config import ExperimentConfig, RunSpec
from .fitting import fit_all_subjects
from .human_data import (
    HUMAN_SOURCE,
    download_human_data,
    human_trials,
    load_human_long,
    load_published_parameters,
)
from .results import complete_trials_only
from .task import Trial
from .validation import GateResult, run_validation_gate


def prepare_human(data_dir: str | Path, trial_set: str) -> dict[str, object]:
    """Download (if needed), decode and return the human tables.

    ``trial_set`` is ``"all"`` (99 trials) or ``"behavioral"`` (the 72 behavioural trials,
    session 0). ``human_all`` always holds all trials; ``human_sub`` the requested subset.
    """
    files = download_human_data(data_dir)
    human_all = an.add_bayes(load_human_long(files["data_allsubj_expt1.mat"], "all"))
    human_sub = human_all if trial_set == "all" else human_all[human_all.session == 0]
    return {
        "files": files,
        "human_all": human_all,
        "human_sub": human_sub.reset_index(drop=True),
        "published": load_published_parameters(files["BETA_neglect_final.mat"]),
        "trials_by_subject": human_trials(human_sub),
    }


def run_gate(human_all: pd.DataFrame, published: pd.DataFrame) -> GateResult:
    """Run the validation gate and raise if the published human fits are not reproduced."""
    gate = run_validation_gate(human_all, published)
    if not gate.passed:
        raise RuntimeError("VALIDATION GATE FAILED; stop. " + gate.message)
    return gate


def simulate(
    config: ExperimentConfig,
    trials_by_subject: dict[int, list[Trial]],
    backend_factory: Callable[[str], Backend],
    progress: Callable[[str], None] | None = print,
    specs: list[RunSpec] | None = None,
) -> list[RunStats]:
    """Run every spec of ``config`` with resume; returns throughput statistics."""
    cfg = config.effective()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), default=str, indent=2))
    stats = []
    for spec in specs if specs is not None else cfg.runs():
        backend = backend_factory(spec.model)
        stats.append(run_spec(spec, backend, trials_by_subject, out_dir, cfg.parallel, progress))
    pd.DataFrame(
        [vars(s) | {"calls_per_second": s.calls_per_second, "complete": s.complete} for s in stats]
    ).to_csv(out_dir / "run_stats.csv", index=False)
    return stats


def analyze(
    human_sub: pd.DataFrame,
    model_long: pd.DataFrame,
    out_dir: str | Path,
    human_all: pd.DataFrame | None = None,
    label: str = "",
) -> dict[str, pd.DataFrame]:
    """All human-model comparisons; writes one CSV per table to ``out_dir`` and returns them.

    Incomplete trials are dropped first (the latter reported in
    ``incomplete_trials``). With ``human_all`` the 99-trial human fits are added as the
    source ``"human (all 99 trials)"``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pre = f"{label}_" if label else ""
    model_long, incomplete = (
        complete_trials_only(model_long) if len(model_long) else (model_long, pd.DataFrame())
    )
    if len(model_long):
        model_long = an.add_bayes(model_long)
    both = pd.concat([human_sub, model_long], ignore_index=True, sort=False)

    tables: dict[str, pd.DataFrame] = {}
    fits = fit_all_subjects(both)
    if human_all is not None:
        fits_all = fit_all_subjects(human_all)
        fits_all["source"] = f"{HUMAN_SOURCE} (all 99 trials)"
        fits = pd.concat([fits, fits_all], ignore_index=True)
    tables["fits"] = fits
    tables["parameters"] = an.parameter_table(fits)
    slopes = an.neglect_slopes(fits)
    tables["slopes"] = slopes
    tables["slope_summary"] = an.slope_summary(slopes)
    io = an.index_of_overreaction(both)
    tables["io_per_subject"] = io
    tables["io_summary"] = an.grid_summary(io, "io")
    dist = an.distance_from_bayes(both)
    tables["distance_per_subject"] = dist
    tables["distance_summary"] = an.grid_summary(dist, "abs_dev")
    tables["signed_summary"] = an.grid_summary(dist, "signed_dev")
    tables["accuracy_overall"] = an.accuracy_overall(dist)
    fq = an.fit_quality(both, fits)
    tables["fit_quality"] = fq["summary"]
    tables["fit_quality_per_subject"] = fq["per_subject"]
    corr_rows, corr_cond, corr_period, corr_subj = [], [], [], []
    for src, g in model_long.groupby("source", sort=True) if len(model_long) else []:
        c = an.deviation_correlation(human_sub, g)
        corr_rows.append(c["overall"].assign(source=src))
        corr_cond.append(c["by_condition"].assign(source=src))
        corr_period.append(c["by_period"].assign(source=src))
        corr_subj.append(c["by_subject_summary"].assign(source=src))
    tables["correlation_overall"] = (
        pd.concat(corr_rows, ignore_index=True) if corr_rows else pd.DataFrame()
    )
    tables["correlation_by_condition"] = (
        pd.concat(corr_cond, ignore_index=True) if corr_cond else pd.DataFrame()
    )
    tables["correlation_by_period"] = (
        pd.concat(corr_period, ignore_index=True) if corr_period else pd.DataFrame()
    )
    tables["correlation_by_subject"] = (
        pd.concat(corr_subj, ignore_index=True) if corr_subj else pd.DataFrame()
    )
    if len(model_long):
        pf = an.parse_failure_table(model_long)
        pf = pd.concat(
            [
                pf,
                pd.DataFrame(
                    [
                        {
                            "source": HUMAN_SOURCE,
                            "mode": "keypad, 4 s deadline",
                            "n_calls": len(human_sub),
                            "failed": an.missing_rate(human_sub),
                            "usable": 1 - an.missing_rate(human_sub),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        pf = pd.DataFrame()
    tables["parse_failures"] = pf
    tables["incomplete_trials"] = incomplete
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out / f"{pre}{name}.csv", index=False)
    return tables

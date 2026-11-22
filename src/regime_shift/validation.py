"""Validation gate: reproduce the published human parameter estimates before any model run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fitting import PARAM_NAMES, fit_all_subjects
from .human_data import PUBLISHED_ALPHA_MEANS, PUBLISHED_BETA_MEANS


@dataclass
class GateResult:
    """Outcome of :func:`run_validation_gate`: per-subject fits, the comparison table and means."""

    passed: bool
    max_abs_subject_diff: float
    max_abs_mean_diff: float
    our_means: np.ndarray
    archive_means: np.ndarray
    paper_means: np.ndarray
    fits: pd.DataFrame
    comparison: pd.DataFrame
    message: str

    def summary_table(self) -> pd.DataFrame:
        """One row per parameter: our mean, the archive-file mean and the paper's rounded mean."""
        return pd.DataFrame(
            {
                "parameter": PARAM_NAMES,
                "ours": np.round(self.our_means, 4),
                "archive_file": np.round(self.archive_means, 4),
                "paper_text": self.paper_means,
            }
        )


def run_validation_gate(
    human_all: pd.DataFrame,
    published: pd.DataFrame,
    tol_subject: float = 1e-3,
    tol_mean: float = 0.005,
) -> GateResult:
    """Fit all subjects on the 99-trial data and compare with ``BETA_neglect_final.mat``.

    Passes when every per-subject estimate is within ``tol_subject`` of the
    archived value and every mean is within ``tol_mean`` of the paper's
    rounded means (0.005: the mean rounds to the published two decimals).
    """
    if human_all.session.nunique() < 2:
        raise ValueError("the gate needs the full data (both sessions); use trial_set='all'")
    fits = fit_all_subjects(human_all)
    ours = fits.set_index("subject")[list(PARAM_NAMES)]
    arch = published.set_index("subject")[list(PARAM_NAMES)].loc[ours.index]
    diff = (ours - arch).abs()
    comparison = pd.concat(
        {"ours": ours.round(4), "archive": arch.round(4), "abs_diff": diff}, axis=1
    )
    our_means = ours.mean().to_numpy()
    archive_means = arch.mean().to_numpy()
    paper_means = np.array([*PUBLISHED_ALPHA_MEANS, *PUBLISHED_BETA_MEANS])
    max_subject = float(diff.to_numpy().max())
    max_mean = float(np.abs(our_means - paper_means).max())
    passed = bool(fits.success.all() and max_subject <= tol_subject and max_mean <= tol_mean)
    msg = (
        f"{'PASSED' if passed else 'FAILED'}: {len(fits)} subjects fitted, "
        f"max |ours - archive| per subject = {max_subject:.2e} (tol {tol_subject}), "
        f"max |mean - paper| = {max_mean:.4f} (tol {tol_mean})"
    )
    return GateResult(
        passed, max_subject, max_mean, our_means, archive_means, paper_means, fits, comparison, msg
    )

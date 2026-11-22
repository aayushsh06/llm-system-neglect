"""Per-subject nonlinear least-squares fit of the system-neglect model.

Mirrors ``get_individual_parameter.m`` from the OSF archive: unweighted least
squares on the probability scale, all six parameters started at 1,
Levenberg-Marquardt, missing responses dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .bayes import NeglectDesign
from .task import N_PERIODS, d_level, q_level

PARAM_NAMES: tuple[str, ...] = ("alpha_1", "alpha_2", "alpha_3", "beta_1", "beta_2", "beta_3")


@dataclass
class NeglectFit:
    """Result of one subject's fit; ``params`` is NaN and ``success`` False when the fit failed."""

    subject: int
    params: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    n_obs: int = 0
    sse: float = float("nan")
    residual_sd: float = float("nan")
    success: bool = False
    message: str = ""

    @property
    def alphas(self) -> np.ndarray:
        return self.params[:3]

    @property
    def betas(self) -> np.ndarray:
        return self.params[3:]

    def as_row(self) -> dict[str, object]:
        """The fit as one flat table row."""
        row: dict[str, object] = {"subject": self.subject}
        row.update(dict(zip(PARAM_NAMES, self.params, strict=True)))
        row.update(
            n_obs=self.n_obs,
            sse=self.sse,
            residual_sd=self.residual_sd,
            success=self.success,
            message=self.message,
        )
        return row


def design_from_long(df: pd.DataFrame) -> tuple[NeglectDesign, np.ndarray]:
    """Build the vectorized design and the ``(n_trials, 10)`` response matrix for one subject."""
    df = df.sort_values(["trial", "period"])
    n_trials = len(df) // N_PERIODS
    if n_trials * N_PERIODS != len(df):
        raise ValueError("subject data length is not a multiple of 10")
    periods = df.period.to_numpy().reshape(n_trials, N_PERIODS)
    if not (periods == np.arange(1, N_PERIODS + 1)).all():
        raise ValueError("each trial must have periods 1..10 exactly once")
    q = df.q.to_numpy(dtype=float)[::N_PERIODS]
    d = df.d.to_numpy(dtype=float)[::N_PERIODS]
    signals = df.signal.to_numpy(dtype=float).reshape(n_trials, N_PERIODS)
    design = NeglectDesign(
        q, d, np.array([q_level(x) for x in q]), np.array([d_level(x) for x in d]), signals
    )
    y = df.estimate.to_numpy(dtype=float).reshape(n_trials, N_PERIODS)
    return design, y


def fit_system_neglect(
    df_subject: pd.DataFrame,
    start: np.ndarray | None = None,
    max_nfev: int = 2000,
) -> NeglectFit:
    """Fit alpha_1..3 and beta_1..3 for one subject (long-format rows of that subject)."""
    subject = int(df_subject.subject.iloc[0]) if len(df_subject) else -1
    fit = NeglectFit(subject=subject)
    try:
        design, y = design_from_long(df_subject)
    except ValueError as exc:
        fit.message = f"bad design: {exc}"
        return fit
    mask = np.isfinite(y)
    fit.n_obs = int(mask.sum())
    # Identifiability: every level needs at least one usable observation.
    for lv in range(3):
        if not mask[design.q_idx == lv].any() or not mask[design.d_idx == lv].any():
            fit.message = "a q or d level has no usable responses"
            return fit
    if fit.n_obs <= 6:
        fit.message = "fewer usable responses than parameters"
        return fit
    x0 = np.ones(6) if start is None else np.asarray(start, dtype=float)

    def residuals(theta: np.ndarray) -> np.ndarray:
        pred = design.predict(theta[:3], theta[3:])
        r = (pred - y)[mask]
        return np.where(np.isfinite(r), r, 1e3)

    try:
        res = least_squares(residuals, x0, method="lm", max_nfev=max_nfev)
    except Exception as exc:  # pragma: no cover - defensive
        fit.message = f"least_squares raised: {exc}"
        return fit
    fit.params = res.x
    fit.sse = float(np.sum(res.fun**2))
    dof = max(fit.n_obs - 6, 1)
    fit.residual_sd = float(np.sqrt(fit.sse / dof))
    fit.success = bool(res.success and np.all(np.isfinite(res.x)))
    fit.message = res.message
    return fit


def fit_all_subjects(df: pd.DataFrame, source_col: str = "source") -> pd.DataFrame:
    """Fit every (source, subject) in a long table; one row per fit, failures kept with NaNs."""
    rows = []
    for (source, _subject), g in df.groupby([source_col, "subject"], sort=True):
        row = fit_system_neglect(g).as_row()
        row["source"] = source
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = ["source", "subject", *PARAM_NAMES, "n_obs", "sse", "residual_sd", "success", "message"]
    return out[cols] if len(out) else pd.DataFrame(columns=cols)

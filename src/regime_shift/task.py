"""The regime-shift detection task (Massey & Wu 2005; Wang, Wu & Wu, eLife).

Conventions follow the OSF archive: a signal is ``1`` for a red ball and ``0``
for a blue ball; conditions are indexed 1..9 with q varying slowest.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

RED = 1
BLUE = 0
N_PERIODS = 10
Q_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)
D_LEVELS: tuple[float, ...] = (1.5, 3.0, 9.0)
N_BLOCKS_BEHAVIORAL = 8
N_BLOCKS_FMRI = 3


def q_level(q: float) -> int:
    """0-based index of ``q`` in :data:`Q_LEVELS`."""
    return _level(q, Q_LEVELS, "q")


def d_level(d: float) -> int:
    """0-based index of ``d`` in :data:`D_LEVELS`."""
    return _level(d, D_LEVELS, "d")


def _level(value: float, levels: tuple[float, ...], name: str) -> int:
    for i, lv in enumerate(levels):
        if abs(value - lv) < 1e-9:
            return i
    raise ValueError(f"{name}={value!r} is not one of {levels}")


def condition_index(q: float, d: float) -> int:
    """Archive condition index 1..9: (q=.01,d=1.5)=1, (q=.01,d=3)=2, ..., (q=.1,d=9)=9."""
    return 3 * q_level(q) + d_level(d) + 1


def condition_from_index(index: int) -> tuple[float, float]:
    """Inverse of :func:`condition_index`."""
    if not 1 <= index <= 9:
        raise ValueError(f"condition index must be 1..9, got {index}")
    return Q_LEVELS[(index - 1) // 3], D_LEVELS[(index - 1) % 3]


ALL_CONDITIONS: tuple[tuple[float, float], ...] = tuple(product(Q_LEVELS, D_LEVELS))


def red_fraction(d: float) -> float:
    """Proportion of red balls in the red urn, as shown to subjects (0.6, 0.75, 0.9)."""
    return d / (1.0 + d)


@dataclass(frozen=True)
class Trial:
    """One trial: a condition and its 10-signal sequence.

    ``change_period`` is the 1-based period from which signals came from the
    blue urn, ``N_PERIODS + 1`` if the regime never shifted, and ``None`` when
    unknown (human data; the archive does not store it).
    """

    subject: int
    trial: int
    session: int
    block: int
    trial_in_block: int
    q: float
    d: float
    signals: tuple[int, ...]
    change_period: int | None = None

    def __post_init__(self) -> None:
        if len(self.signals) != N_PERIODS:
            raise ValueError(f"expected {N_PERIODS} signals, got {len(self.signals)}")
        if any(s not in (RED, BLUE) for s in self.signals):
            raise ValueError("signals must be 0 (blue) or 1 (red)")
        q_level(self.q)
        d_level(self.d)

    @property
    def condition(self) -> int:
        return condition_index(self.q, self.d)


def sample_change_period(q: float, rng: np.random.Generator, n_periods: int = N_PERIODS) -> int:
    """Period before which the regime shifts, mirroring ``createInputs_shift.m``.

    Before every period (including the first) the regime shifts with
    probability ``q``; the first shift is permanent. Returns ``n_periods + 1``
    when no shift happens within the trial.
    """
    dice = rng.random(n_periods)
    hits = np.flatnonzero(dice < q)
    return int(hits[0]) + 1 if hits.size else n_periods + 1


def sample_signals(
    d: float, change_period: int, rng: np.random.Generator, n_periods: int = N_PERIODS
) -> tuple[int, ...]:
    """Draw one ball per period with replacement from the current urn."""
    p_red_red_urn = red_fraction(d)
    dice = rng.random(n_periods)
    out = []
    for period in range(1, n_periods + 1):
        p_red = p_red_red_urn if period < change_period else 1.0 - p_red_red_urn
        out.append(RED if dice[period - 1] < p_red else BLUE)
    return tuple(out)


def generate_trial(
    q: float, d: float, rng: np.random.Generator, n_periods: int = N_PERIODS
) -> tuple[tuple[int, ...], int]:
    """Return ``(signals, change_period)`` for one trial."""
    change = sample_change_period(q, rng, n_periods)
    return sample_signals(d, change, rng, n_periods), change


def generate_subject_trials(
    subject: int,
    seed: int,
    n_blocks: int = N_BLOCKS_BEHAVIORAL,
    session: int = 0,
) -> list[Trial]:
    """Generate a subject's trials: ``n_blocks`` blocks of the 9 conditions in random order.

    Deterministic given ``(subject, seed)``.
    """
    rng = np.random.default_rng([seed, subject, session])
    trials: list[Trial] = []
    t = 0
    for block in range(1, n_blocks + 1):
        order = rng.permutation(len(ALL_CONDITIONS))
        for k, idx in enumerate(order, start=1):
            q, d = ALL_CONDITIONS[idx]
            signals, change = generate_trial(q, d, rng)
            t += 1
            trials.append(
                Trial(
                    subject=subject,
                    trial=t,
                    session=session,
                    block=block,
                    trial_in_block=k,
                    q=q,
                    d=d,
                    signals=signals,
                    change_period=change,
                )
            )
    return trials


def regime_at(change_period: int, period: int) -> int:
    """1 if the regime is blue at ``period`` (1-based), else 0."""
    return int(period >= change_period)

"""Bayesian posterior (Equation 1) and system-neglect model (Equation 2).

All functions are pure. Signals use the archive convention ``1 = red``,
``0 = blue``; probabilities are P(regime is blue).

- :func:`forward_filter`: recursive Bayesian filter, one trial.
- :func:`closed_form_odds`: the paper's closed form with free ``alpha`` and ``beta``.
- :class:`NeglectDesign`: precomputed arrays that evaluate Equation 2 for many
  trials at once (used by the fitting code).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .task import N_PERIODS


def likelihoods(d: float, signal: int) -> tuple[float, float]:
    """``(L_blue, L_red)``: probability of ``signal`` under the blue and red urns."""
    dominant = d / (1.0 + d)
    minority = 1.0 / (1.0 + d)
    if signal == 0:  # blue ball
        return dominant, minority
    return minority, dominant


def forward_filter(q: float, d: float, signals: Sequence[int], p0: float = 0.0) -> np.ndarray:
    """Posterior P(blue) after each signal, by recursive filtering.

    prior_t = p_{t-1} + (1 - p_{t-1}) q
    p_t = prior_t L_blue / (prior_t L_blue + (1 - prior_t) L_red)
    """
    p = p0
    out = np.empty(len(signals))
    for t, s in enumerate(signals):
        prior = p + (1.0 - p) * q
        l_blue, l_red = likelihoods(d, s)
        p = prior * l_blue / (prior * l_blue + (1.0 - prior) * l_red)
        out[t] = p
    return out


def red_count_matrix(signals: Sequence[int]) -> np.ndarray:
    """``R[j, t]`` = number of red signals in periods ``j..t`` (0-based, inclusive); 0 if j > t."""
    r = np.asarray(signals, dtype=float)
    n = r.size
    cum = np.concatenate([[0.0], np.cumsum(r)])
    j = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    counts = cum[t + 1] - cum[j]
    return np.where(j <= t, counts, 0.0)


def _mixture_weights(q: float, n: int) -> np.ndarray:
    """``W[j, t]`` = q(1-q)^j / (1-(1-q)^(t+1)) for 0-based j <= t, else 0."""
    j = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    w = q * (1.0 - q) ** j / (1.0 - (1.0 - q) ** (t + 1))
    return np.where(j <= t, w, 0.0)


def closed_form_odds(
    q: float,
    d: float,
    signals: Sequence[int],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Posterior odds of blue after each period, Equation 1 (alpha=beta=1) or Equation 2.

    odds_t = [(1-(1-alpha q)^t)/(1-alpha q)^t] * sum_{j<=t} W[j,t] d^(beta (t+1-j-2 R_{j,t}))
    with the 1-based exponent ``t + 1 - j - 2 R`` and the mixture weights using the true ``q``.
    """
    n = len(signals)
    t1 = np.arange(1, n + 1)
    aq = alpha * q
    base = (1.0 - (1.0 - aq) ** t1) / (1.0 - aq) ** t1
    j = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    exponent = (t + 1) + 1 - (j + 1) - 2.0 * red_count_matrix(signals)
    with np.errstate(over="ignore", invalid="ignore"):
        terms = _mixture_weights(q, n) * np.power(d, beta * exponent)
        terms = np.where(j <= t, terms, 0.0)
        return base * terms.sum(axis=0)


def odds_to_prob(odds: np.ndarray) -> np.ndarray:
    """Map odds to probability, with inf -> 1 and 0 -> 0."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 1.0 / (1.0 + 1.0 / odds)
    p = np.where(odds == 0.0, 0.0, p)
    return np.where(np.isposinf(odds), 1.0, p)


def closed_form_posterior(q: float, d: float, signals: Sequence[int]) -> np.ndarray:
    """Equation 1 as probabilities."""
    return odds_to_prob(closed_form_odds(q, d, signals))


def system_neglect_posterior(
    q: float, d: float, signals: Sequence[int], alpha: float, beta: float
) -> np.ndarray:
    """Equation 2 as probabilities for one trial with scalar ``alpha`` and ``beta``."""
    return odds_to_prob(closed_form_odds(q, d, signals, alpha, beta))


class NeglectDesign:
    """Precomputed arrays for fast evaluation of Equation 2 over many trials.

    Parameters are per-level: ``alphas[q_level]`` and ``betas[d_level]``.
    """

    def __init__(
        self,
        q: np.ndarray,
        d: np.ndarray,
        q_idx: np.ndarray,
        d_idx: np.ndarray,
        signals: np.ndarray,
    ) -> None:
        signals = np.asarray(signals, dtype=float)
        if signals.ndim != 2 or signals.shape[1] != N_PERIODS:
            raise ValueError("signals must have shape (n_trials, 10)")
        self.n_trials = signals.shape[0]
        self.q = np.asarray(q, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.q_idx = np.asarray(q_idx, dtype=int)
        self.d_idx = np.asarray(d_idx, dtype=int)
        n = N_PERIODS
        self.t1 = np.arange(1, n + 1)[None, :]  # (1, T)
        j = np.arange(n)[None, :, None]
        t = np.arange(n)[None, None, :]
        self.mask = j <= t  # (1, J, T)
        # R[trial, j, t]
        cum = np.concatenate([np.zeros((self.n_trials, 1)), np.cumsum(signals, axis=1)], axis=1)
        counts = cum[:, None, 1:] - cum[:, :n, None]  # cum[t+1] - cum[j]
        self.exponent = np.where(self.mask, (t + 1) - j - 2.0 * counts, 0.0)  # (N, J, T)
        self.weights = np.stack([_mixture_weights(qq, n) for qq in self.q])  # (N, J, T)
        self.log_d = np.log(self.d)[:, None, None]

    def predict(self, alphas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        """P(blue) for every (trial, period), shape ``(n_trials, 10)``."""
        a = np.asarray(alphas, dtype=float)[self.q_idx][:, None]
        b = np.asarray(betas, dtype=float)[self.d_idx][:, None, None]
        aq = a * self.q[:, None]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            base = (1.0 - (1.0 - aq) ** self.t1) / (1.0 - aq) ** self.t1
            terms = self.weights * np.exp(b * self.exponent * self.log_d)
            terms = np.where(self.mask, terms, 0.0)
            odds = base * terms.sum(axis=1)
        return odds_to_prob(odds)

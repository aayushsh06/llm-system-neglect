import numpy as np
import pytest

from regime_shift.bayes import (
    NeglectDesign,
    closed_form_odds,
    closed_form_posterior,
    forward_filter,
    odds_to_prob,
    red_count_matrix,
    system_neglect_posterior,
)
from regime_shift.task import ALL_CONDITIONS, d_level, q_level


@pytest.mark.parametrize("q,d", ALL_CONDITIONS)
def test_filter_equals_closed_form(q, d):
    rng = np.random.default_rng(1)
    for _ in range(100):
        s = rng.integers(0, 2, 10)
        np.testing.assert_allclose(
            forward_filter(q, d, s), closed_form_posterior(q, d, s), atol=1e-10
        )


def test_hand_computed_first_period():
    # one blue ball, q=0.1, d=9: prior 0.1, odds = (0.1/0.9) * 9 = 1 -> p = 0.5
    assert forward_filter(0.1, 9.0, [0])[0] == pytest.approx(0.5)
    # one red ball: odds = (0.1/0.9) / 9 -> p = 1/82
    assert forward_filter(0.1, 9.0, [1])[0] == pytest.approx(1 / 82)


def test_red_count_matrix():
    r = red_count_matrix([1, 0, 1])
    assert r[0, 2] == 2 and r[1, 2] == 1 and r[2, 2] == 1 and r[2, 0] == 0


def test_system_neglect_reduces_to_bayes_at_one():
    s = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    np.testing.assert_allclose(
        system_neglect_posterior(0.05, 3.0, s, 1.0, 1.0), forward_filter(0.05, 3.0, s), atol=1e-12
    )


def test_system_neglect_direction():
    s = [0] * 10
    base = system_neglect_posterior(0.01, 1.5, s, 1.0, 1.0)
    assert (system_neglect_posterior(0.01, 1.5, s, 4.0, 1.0) > base).all()
    assert (system_neglect_posterior(0.01, 1.5, s, 1.0, 2.0) > base).all()


def test_odds_to_prob_edges():
    np.testing.assert_array_equal(odds_to_prob(np.array([0.0, np.inf, 1.0])), [0.0, 1.0, 0.5])


def test_design_matches_per_trial():
    rng = np.random.default_rng(3)
    conds = [ALL_CONDITIONS[i] for i in rng.integers(0, 9, 40)]
    q = np.array([c[0] for c in conds])
    d = np.array([c[1] for c in conds])
    sig = rng.integers(0, 2, (40, 10))
    des = NeglectDesign(q, d, [q_level(x) for x in q], [d_level(x) for x in d], sig)
    al, be = (2.5, 1.1, 0.4), (1.7, 0.9, 0.5)
    pred = des.predict(al, be)
    for i in range(40):
        ref = system_neglect_posterior(q[i], d[i], sig[i], al[q_level(q[i])], be[d_level(d[i])])
        np.testing.assert_allclose(pred[i], ref, atol=1e-12)
    np.testing.assert_allclose(
        des.predict([1, 1, 1], [1, 1, 1])[0], forward_filter(q[0], d[0], sig[0])
    )


def test_closed_form_odds_no_overflow_crash():
    odds = closed_form_odds(0.1, 9.0, [0] * 10, alpha=1.0, beta=50.0)
    assert np.all(np.isfinite(odds_to_prob(odds)))

from collections import Counter

import numpy as np
import pytest

from regime_shift.task import (
    BLUE,
    RED,
    Trial,
    condition_from_index,
    condition_index,
    generate_subject_trials,
    regime_at,
    sample_change_period,
    sample_signals,
)


class FixedDice:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def random(self, n):
        assert n == len(self.values)
        return self.values


def test_condition_index_roundtrip():
    seen = set()
    for q in (0.01, 0.05, 0.1):
        for d in (1.5, 3.0, 9.0):
            i = condition_index(q, d)
            assert condition_from_index(i) == (q, d)
            seen.add(i)
    assert seen == set(range(1, 10))
    assert condition_index(0.01, 1.5) == 1 and condition_index(0.1, 9.0) == 9


def test_change_period_first_hit_and_no_change():
    assert (
        sample_change_period(0.5, FixedDice([0.9, 0.9, 0.1, 0.05, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]))
        == 3
    )
    assert sample_change_period(0.5, FixedDice([0.9] * 10)) == 11
    assert sample_change_period(0.5, FixedDice([0.0] + [0.9] * 9)) == 1  # shift before period 1


def test_absorbing_regime_switches_urn_permanently():
    # d=9: red urn draws red when dice < 0.9; blue urn draws red only when dice < 0.1
    dice = [0.5] * 10
    assert sample_signals(9.0, 4, FixedDice(dice)) == (
        RED,
        RED,
        RED,
        BLUE,
        BLUE,
        BLUE,
        BLUE,
        BLUE,
        BLUE,
        BLUE,
    )
    assert sample_signals(9.0, 11, FixedDice(dice)) == (RED,) * 10
    assert sample_signals(9.0, 1, FixedDice(dice)) == (BLUE,) * 10


def test_absorbing_regime_statistically():
    rng = np.random.default_rng(0)
    after = []
    for _ in range(2000):
        change = sample_change_period(0.3, rng)
        sig = sample_signals(9.0, change, rng)
        after += [s for p, s in enumerate(sig, 1) if p >= change]
    assert np.mean(np.array(after) == BLUE) == pytest.approx(0.9, abs=0.02)


def test_deterministic_seeding_and_balance():
    a = generate_subject_trials(3, 42)
    b = generate_subject_trials(3, 42)
    c = generate_subject_trials(3, 43)
    d = generate_subject_trials(4, 42)
    assert a == b
    assert a != c and a != d
    assert len(a) == 72
    assert Counter(t.condition for t in a) == dict.fromkeys(range(1, 10), 8)
    assert [t.trial for t in a] == list(range(1, 73))


def test_trial_validation():
    with pytest.raises(ValueError):
        Trial(1, 1, 0, 1, 1, 0.01, 1.5, (1, 0))
    with pytest.raises(ValueError):
        Trial(1, 1, 0, 1, 1, 0.02, 1.5, (1,) * 10)


def test_regime_at():
    assert regime_at(4, 3) == 0 and regime_at(4, 4) == 1 and regime_at(11, 10) == 0

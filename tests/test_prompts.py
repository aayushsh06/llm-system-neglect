from dataclasses import replace

import numpy as np
import pytest

from regime_shift.bayes import forward_filter, red_count_matrix
from regime_shift.collection.backends import BayesianMock
from regime_shift.prompts import (
    COLOR,
    build_main_messages,
    build_stateless_messages,
    parse_prompt_state,
    period_message,
    trial_information,
)
from regime_shift.task import BLUE, RED, Trial, generate_subject_trials

TRIAL = Trial(1, 5, 0, 1, 5, 0.05, 3.0, (1, 0, 0, 1, 0, 0, 0, 1, 0, 0), 4)


def test_trial_information_uses_task_wording():
    s = trial_information(0.01, 1.5)
    assert "Switch Probability= 1%" in s and "Color Ratio= [ 0.6  0.4 ]" in s


@pytest.mark.parametrize("period", [1, 4, 10])
def test_main_messages_structure_and_roundtrip(period):
    prev = [str(10 * t) for t in range(1, period)]
    msgs = build_main_messages(TRIAL, period, prev)
    assert msgs[0]["role"] == "system"
    assert len(msgs) == 1 + 2 * period - 1
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == prev
    assert "Trial information" in msgs[1]["content"]
    q, d, signals = parse_prompt_state(msgs)
    assert (q, d) == (0.05, 3.0)
    assert signals == TRIAL.signals[:period]
    # only signals up to the current period are mentioned
    assert f"periods 1 to {period})" in msgs[-1]["content"]


def test_stateless_messages_roundtrip():
    msgs = build_stateless_messages(TRIAL, 7)
    assert len(msgs) == 2
    assert parse_prompt_state(msgs) == (0.05, 3.0, TRIAL.signals[:7])


def test_previous_answer_count_enforced():
    with pytest.raises(ValueError):
        build_main_messages(TRIAL, 3, ["1"])


def test_colour_words_match_signal_coding_everywhere(human_all):
    """The word shown to the model, the integer in the filter and the fits, the Bayesian
    mock that reads the prompt, and the archive's own coding all agree: 1 = RED, 0 = BLUE,
    and the reported probability is P(BLUE regime)."""
    assert (RED, BLUE) == (1, 0) and COLOR[RED] == "RED" and COLOR[BLUE] == "BLUE"
    # 1. prompt word <-> integer, period by period, on real sequences
    for t in generate_subject_trials(1, 11)[:9]:
        for p in range(1, 11):
            word = "RED" if t.signals[p - 1] == RED else "BLUE"
            assert f"Ball drawn: {word}." in period_message(p, t.signals)
    # 2. filter: red evidence lowers P(blue), blue evidence raises it; R counts the 1s
    low = forward_filter(0.1, 9.0, [RED] * 10)
    high = forward_filter(0.1, 9.0, [BLUE] * 10)
    assert low[-1] < 0.05 < 0.95 < high[-1]
    assert red_count_matrix([RED, BLUE, RED])[0].tolist() == [1, 1, 2]
    # 3. the mock answers the same text the model sees, through the same parser
    mk = replace(TRIAL, q=0.1, d=9.0, signals=tuple([BLUE] * 10), change_period=1)
    red_trial = replace(TRIAL, q=0.1, d=9.0, signals=tuple([RED] * 10), change_period=None)
    mock = BayesianMock()
    ans_blue = int(
        mock.chat(build_main_messages(mk, 10, ["0"] * 9), temperature=0, seed=0, num_predict=8).text
    )
    ans_red = int(
        mock.chat(
            build_main_messages(red_trial, 10, ["0"] * 9), temperature=0, seed=0, num_predict=8
        ).text
    )
    assert ans_red < 5 < 95 <= ans_blue
    # 4. the archive codes signals the same way: its posterior falls after a 1, rises after a 0
    h = human_all.sort_values(["subject", "trial", "period"])
    delta = h.groupby(["subject", "trial"]).p_bayes_archive.diff()
    after_red = delta[(h.period > 1) & (h.signal == RED)].mean()
    after_blue = delta[(h.period > 1) & (h.signal == BLUE)].mean()
    assert after_red < 0 < after_blue
    # 5. our filter on the archive's integers reproduces the archive's posterior
    sub = h[(h.subject == 1) & (h.trial <= 9)]
    for _, g in sub.groupby("trial"):
        np.testing.assert_allclose(
            forward_filter(g.q.iloc[0], g.d.iloc[0], g.signal.to_numpy()),
            g.p_bayes_archive.to_numpy(),
            atol=1e-10,
        )

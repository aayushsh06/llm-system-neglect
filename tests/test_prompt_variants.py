import pytest

from regime_shift import prompts
from regime_shift.collection.backends import BayesianMock
from regime_shift.config import RunSpec
from regime_shift.task import D_LEVELS, Q_LEVELS, generate_subject_trials

TRIAL = generate_subject_trials(1, 7)[0]


def test_v1_is_the_default_and_unchanged():
    for q in Q_LEVELS:
        for d in D_LEVELS:
            assert prompts.trial_information(q, d) == prompts.trial_information(q, d, "v1")
    assert prompts.period_message(3, TRIAL.signals) == prompts.period_message(
        3, TRIAL.signals, "v1"
    )
    assert prompts.system_prompt() == prompts.SYSTEM_PROMPT
    assert "Period 3 of 10." in prompts.period_message(3, TRIAL.signals)
    assert prompts.build_main_messages(TRIAL, 2, ["5"]) == prompts.build_main_messages(
        TRIAL, 2, ["5"], "v1"
    )


@pytest.mark.parametrize("d, odds", [(1.5, (3, 2)), (3.0, (3, 1)), (9.0, (9, 1))])
def test_ratio_words_hides_the_minority_share(d, odds):
    text = prompts.trial_information(0.05, d, "ratio-words")
    a, b = odds
    assert f"{a} red balls for every {b} blue" in text
    for forbidden in ("Color Ratio=", "%", "0.4", "0.25", "0.1 ", "(d ="):
        assert forbidden not in text.replace("Switch Probability= 5%", "")
    assert prompts.odds_integers(d) == odds


def test_ratio_words_roundtrips_through_the_prompt_parser():
    for d in D_LEVELS:
        t = next(x for x in generate_subject_trials(2, 3) if x.d == d)
        msgs = prompts.build_stateless_messages(t, 4, variant="ratio-words")
        q, dd, sig = prompts.parse_prompt_state(msgs)
        assert (q, dd, sig) == (t.q, t.d, t.signals[:4])


def test_no_horizon_removes_the_trial_length():
    msgs = prompts.build_main_messages(TRIAL, 10, ["1"] * 9, variant="no-horizon")
    assert "of 10" not in msgs[0]["content"] and "10 periods" not in msgs[0]["content"]
    for m in msgs[1:]:
        assert "of 10" not in m["content"]
    assert msgs[-1]["content"].startswith("Period 10. Ball drawn")
    assert prompts.parse_prompt_state(msgs)[2] == TRIAL.signals


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        prompts.period_message(1, TRIAL.signals, variant="v2")
    with pytest.raises(ValueError):
        RunSpec("m", "main", 0.7, 1, 1, "behavioral", 8, prompt_variant="v2")


def test_runspec_label_and_source_carry_the_variant():
    base = RunSpec("qwen2.5:3b", "main", 0.7, 1, 30, "behavioral", 8)
    var = RunSpec("qwen2.5:3b", "main", 0.7, 1, 30, "behavioral", 8, prompt_variant="no-horizon")
    assert base.label == "qwen2.5_3b__main__T0.7__behavioral"
    assert var.label == "qwen2.5_3b__main__T0.7__behavioral__no-horizon"
    assert var.source == "qwen2.5:3b [main, T0.7, no-horizon]"


def test_mocks_answer_variant_prompts():
    mock = BayesianMock()
    for variant in prompts.VARIANTS:
        msgs = prompts.build_main_messages(TRIAL, 3, ["0", "0"], variant=variant)
        resp = mock.chat(msgs, temperature=0, seed=1, num_predict=8)
        assert 0 <= int(resp.text) <= 99

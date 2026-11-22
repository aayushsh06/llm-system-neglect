import math

import pytest

from regime_shift.parsing import final_number, parse_estimate


@pytest.mark.parametrize(
    "text,status,value",
    [
        ("85", "ok", 85),
        (" 7 ", "ok", 7),
        ("00", "ok", 0),
        ("99.", "ok", 99),
        ("85%", "ok_percent", 85),
        ("85 percent", "ok_percent", 85),
        ("Probability: 85", "ok_wrapped", 85),
        ("I think 60.", "ok_wrapped", 60),
        ("0.85", "ok_decimal", 85),
        (".5", "ok_decimal", 50),
        ("100", "clamped_100", 99),
        ("100%", "clamped_100", 99),
        ("150", "out_of_range", None),
        ("Between 40 and 50", "ambiguous", None),
        ("Period 3: 45", "ambiguous", None),
        ("I cannot say.", "no_number", None),
        ("", "empty", None),
        ("   \n", "empty", None),
        (None, "empty", None),
    ],
)
def test_parse_cases(text, status, value):
    r = parse_estimate(text)
    assert r.status == status
    assert r.value == value
    if value is None:
        assert math.isnan(r.estimate)
    else:
        assert r.estimate == pytest.approx(value / 100)


def test_no_clamp_option():
    assert parse_estimate("100", clamp_100=False).status == "out_of_range"


@pytest.mark.parametrize(
    "text, status, value",
    [
        ("42 Explanation: Drawing a blue ball", "ok_wrapped", 42),
        ("1 Given that the switch probability is", "ok_wrapped", 1),
        ("10 Explanation: Since the trial has 10 periods", "ok_wrapped", 10),
        ("**Answer: 17**", "ok_wrapped", 17),
        ("Probability: 85", "ok_wrapped", 85),
        ("The answer is 42.", "ok_wrapped", 42),
        ("... so about 1%.\n\n1", "ambiguous", None),  # two numbers: final_number() handles this
        (
            "Period 1: prior BLUE = 0.05",
            "embedded",
            None,
        ),  # 1 and 0.05 -> two numbers, neither an answer
        ("Prior: before period 1, P(switched) = ", "embedded", None),
        ("Prior before the first period: switch probability 1%,", "embedded", None),
        ("Let me compute the posterior sequentially.\n\nPeriod 1", "embedded", None),
        ("Period 3 of 10", "ambiguous", None),
        ("This is period 10", "embedded", None),
        ("Let me compute carefully.", "no_number", None),
    ],
)
def test_wrapped_numbers_are_taken_only_from_answer_positions(text, status, value):
    r = parse_estimate(text)
    assert r.status == status and r.value == value
    assert r.ok == (status.startswith("ok"))


def test_final_number_for_reasoning_replies():
    assert final_number("Setting up ... Posterior P(BLUE) = 0.0058 -> about 1%.\n\n1") == 1
    assert final_number("The probability is 42.") == 42
    assert final_number("17%") == 17
    assert final_number("Period 3 of 10") is None  # echoed prompt number
    assert final_number("I can't give a reliable number for this one.") is None
    assert final_number("0.85") is None  # a decimal is not a final integer
    assert final_number("") is None and final_number(None) is None

"""Prompt construction for the model protocol, and a parser that reads the task
state back out of a prompt (used by the mock backends).

The OSF archive contains no written subject instructions, so the wording below
was written from the paper's Methods section and reuses the strings the task
code put on screen ("Switch Probability= 1%", "Color Ratio= [ 0.6  0.4 ]").
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .task import BLUE, N_PERIODS, RED, Trial, red_fraction

PROMPT_VERSION = "v1"

# Prompt variants. "v1" is the main prompt; the other two are control prompts:
# - "ratio-words": the colour ratio is given as small-integer odds ("3 red balls for every
#   2 blue balls") with no proportions, percentages or the number d on screen, to test
#   whether models copy the displayed minority share (40 / 25 / 10).
# - "no-horizon": the period line says "Period k." instead of "Period k of 10." and the
#   instructions do not state the number of periods, to test the last-period jump.
VARIANTS: tuple[str, ...] = ("v1", "ratio-words", "no-horizon")

COLOR = {RED: "RED", BLUE: "BLUE"}


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {VARIANTS}")


def system_prompt(variant: str = "v1") -> str:
    """The system prompt for ``variant``; only "no-horizon" changes it."""
    _check_variant(variant)
    if variant == "no-horizon":
        return SYSTEM_PROMPT.replace(
            f"A trial has {N_PERIODS} periods and", "A trial has several periods and"
        )
    return SYSTEM_PROMPT


def odds_integers(d: float) -> tuple[int, int]:
    """d as a ratio of small integers: 1.5 -> (3, 2), 3 -> (3, 1), 9 -> (9, 1)."""
    for denom in (1, 2, 3, 4, 5):
        num = d * denom
        if abs(num - round(num)) < 1e-9:
            return int(round(num)), denom
    raise ValueError(f"d = {d} is not a small-integer ratio")


SYSTEM_PROMPT = (
    "You are taking part in a decision experiment about detecting a change in the "
    "environment. The environment is called a regime. There are two regimes, RED and BLUE, "
    "and each is represented by an urn of red and blue balls. In the RED urn the ratio of red "
    "balls to blue balls is d:1; in the BLUE urn the ratio of blue balls to red balls is the "
    "same d:1. The number d is the color ratio, or signal diagnosticity, and you are told it "
    "before each trial, together with the switch probability.\n\n"
    f"A trial has {N_PERIODS} periods and always starts in the RED regime. Before each period, "
    "including the first, the regime can switch from RED to BLUE with the stated switch "
    "probability. The regime can switch only once: once it is BLUE it stays BLUE for the rest "
    "of the trial. In each period one ball is drawn, with replacement, from the urn of the "
    "current regime, and you are shown its color. You are not told whether the regime has "
    "switched.\n\n"
    "After each ball, estimate the probability that the current regime is BLUE. Answer with a "
    "whole number from 0 to 99 (percent) and nothing else: no words, no symbols, no "
    "explanation."
)


def trial_information(q: float, d: float, variant: str = "v1") -> str:
    """The pre-trial information screen, in the task code's wording."""
    _check_variant(variant)
    if variant == "ratio-words":
        a, b = odds_integers(d)
        ball_b = "ball" if b == 1 else "balls"
        return (
            f"Trial information. Switch Probability= {q * 100:g}%. "
            f"Color ratio: in the RED urn there are {a} red balls for every {b} blue {ball_b}; "
            f"in the BLUE urn there are {a} blue balls for every {b} red {ball_b}."
        )
    pr = red_fraction(d)
    pb = 1.0 - pr
    return (
        f"Trial information. Switch Probability= {q * 100:g}%. "
        f"Color Ratio= [ {pr:g}  {pb:g} ] (d = {d:g}: the RED urn is {pr:.0%} red and {pb:.0%} "
        f"blue; the BLUE urn is {pr:.0%} blue and {pb:.0%} red)."
    )


def signals_so_far(signals: Sequence[int]) -> str:
    """``(1, 0, 0)`` -> ``"RED, BLUE, BLUE"``."""
    return ", ".join(COLOR[s] for s in signals)


def period_message(period: int, signals: Sequence[int], variant: str = "v1") -> str:
    """The user turn for ``period`` (1-based) given the signals drawn up to and including it."""
    _check_variant(variant)
    if not 1 <= period <= len(signals):
        raise ValueError("period must index into signals")
    seen = signals[:period]
    head = f"Period {period}." if variant == "no-horizon" else f"Period {period} of {N_PERIODS}."
    ask = "Probability (0-99) that the regime is now BLUE?"
    return (
        f"{head} Ball drawn: {COLOR[seen[-1]]}. "
        f"All balls drawn so far (periods 1 to {period}): {signals_so_far(seen)}. {ask}"
    )


def build_main_messages(
    trial: Trial,
    period: int,
    previous_answers: Sequence[str],
    variant: str = "v1",
) -> list[dict[str, str]]:
    """Conversation for the multi-turn ("main") mode up to the question at ``period``.

    ``previous_answers`` are the model's raw replies for periods 1..period-1;
    they are replayed verbatim as assistant turns.
    """
    if len(previous_answers) != period - 1:
        raise ValueError("need exactly period-1 previous answers")
    system = system_prompt(variant)
    messages = [{"role": "system", "content": system}]
    for t in range(1, period + 1):
        content = period_message(t, trial.signals, variant)
        if t == 1:
            content = trial_information(trial.q, trial.d, variant) + "\n\n" + content
        messages.append({"role": "user", "content": content})
        if t < period:
            messages.append({"role": "assistant", "content": previous_answers[t - 1]})
    return messages


def build_stateless_messages(
    trial: Trial, period: int, variant: str = "v1"
) -> list[dict[str, str]]:
    """Single-turn prompt for stateless mode: trial information and the history up to ``period``."""
    system = system_prompt(variant)
    content = trial_information(trial.q, trial.d, variant) + "\n\n"
    content += period_message(period, trial.signals, variant)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


_Q = re.compile(r"Switch Probability= ([0-9.]+)%")
_D = re.compile(r"\(d = ([0-9.]+):")
_ODDS = re.compile(r"there are (\d+) red balls for every (\d+) blue")
_HIST = re.compile(r"All balls drawn so far \(periods 1 to \d+\): ([A-Z, ]+)\.")


def parse_prompt_state(messages: Sequence[dict[str, str]]) -> tuple[float, float, tuple[int, ...]]:
    """Recover ``(q, d, signals so far)`` from a prompt built by this module.

    Used by the mock backends so that they answer from the same text a real
    model sees.
    """
    users = [m["content"] for m in messages if m["role"] == "user"]
    if not users:
        raise ValueError("no user message")
    mq = _Q.search(users[0])
    md = _D.search(users[0])
    mo = _ODDS.search(users[0])
    if not (mq and (md or mo)):
        raise ValueError("trial information not found in first user message")
    d = float(md.group(1)) if md else int(mo.group(1)) / int(mo.group(2))
    mh = _HIST.search(users[-1])
    if not mh:
        raise ValueError("signal history not found in last user message")
    names = [s.strip() for s in mh.group(1).split(",") if s.strip()]
    inv = {v: k for k, v in COLOR.items()}
    try:
        signals = tuple(inv[n] for n in names)
    except KeyError as exc:
        raise ValueError(f"unknown color {exc}") from exc
    return float(mq.group(1)) / 100.0, d, signals

"""Parse a model's free-text reply into a probability estimate.

Every reply gets exactly one status so that failure rates can be reported.
Only the ``ok*`` statuses and ``clamped_100`` yield an estimate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

OK_STATUSES: frozenset[str] = frozenset(
    {"ok", "ok_percent", "ok_wrapped", "ok_decimal", "clamped_100"}
)
ALL_STATUSES: tuple[str, ...] = (
    "ok",
    "ok_percent",
    "ok_wrapped",
    "ok_decimal",
    "clamped_100",
    "out_of_range",
    "embedded",  # a number exists but only inside prose (e.g. echoed "Period 1"): not usable
    "ambiguous",
    "no_number",
    "empty",
    "backend_error",  # not a parse outcome: the call itself failed (reported, never persisted)
)

_BARE = re.compile(r"^\s*(\d{1,3})\s*(%|percent)?\s*[.!]?\s*$", re.IGNORECASE)
_DECIMAL = re.compile(r"^\s*(0?\.\d+|1\.0+|0|1)\s*$")
_INT = re.compile(r"(?<![\d.])(\d{1,3})(?!\d)(?!\.\d)")
# A wrapped answer is accepted only where a model puts its answer: as the FIRST token of the
# reply (optionally after a short label such as "Answer:" or "**"), or as the LAST token
# (optionally followed by % and punctuation). A number that only appears inside prose, such
# as an echoed "Period 1" or "switch probability 5%", is never taken as the answer.
_LABEL = r"(?:\**\s*(?:answer|probability|estimate|p\(blue\)|blue|p)\s*[:=]?\s*)?"
_LEADING = re.compile(
    r"^\W{0,3}" + _LABEL + r"(\d{1,3})(?!\d)(?!\.\d)\s*(%|percent)?(?=\W|$)", re.IGNORECASE
)
_TRAILING = re.compile(
    r"(?<![\d.])(\d{1,3})(?!\d)(?!\.\d)\s*(%|percent)?\s*[.!)\]*]*\s*$", re.IGNORECASE
)
_ECHO_BEFORE = re.compile(
    r"(?:\bof|\bperiods?|\bto|\bratio|\bprobability\s*=?|\bd\s*=|\bq\s*=)\s*$", re.IGNORECASE
)


def final_number(text: str | None) -> int | None:
    """The last integer token of a reply (``"... about 1%.\n\n1"`` -> 1), or None.

    Used for the "reasoning allowed" API condition, where only the final number is kept and
    replayed. The integer must end the reply (apart from %/punctuation) and must not be an
    echoed prompt number such as the 10 in "Period 3 of 10".
    """
    raw = "" if text is None else text.strip()
    m = _TRAILING.search(raw)
    if not m or _ECHO_BEFORE.search(raw[: m.start(1)]):
        return None
    return int(m.group(1))


@dataclass(frozen=True)
class ParseResult:
    """Outcome of :func:`parse_estimate`: a status, the integer read (if any) and the raw reply."""

    status: str
    value: int | None
    raw: str

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES

    @property
    def estimate(self) -> float:
        """Probability in [0, 0.99], or NaN when the reply was not usable."""
        return self.value / 100.0 if self.ok and self.value is not None else math.nan


def _range_check(value: int, status: str, raw: str, clamp_100: bool) -> ParseResult:
    if 0 <= value <= 99:
        return ParseResult(status, value, raw)
    if value == 100 and clamp_100:
        return ParseResult("clamped_100", 99, raw)
    return ParseResult("out_of_range", None, raw)


def parse_estimate(text: str | None, clamp_100: bool = True) -> ParseResult:
    """Parse a reply such as ``"85"``, ``"85%"``, ``"Probability: 85"`` or ``"0.85"``.

    Rules, in order:
    - empty or whitespace -> ``empty``
    - a bare integer (optionally with ``%``) -> ``ok`` / ``ok_percent``
    - a bare decimal fraction in [0, 1] -> ``ok_decimal`` (``0.85`` -> 85)
    - a leading integer followed by text ("42 Explanation: ...", "Answer: 42") -> ``ok_wrapped``
    - a single integer that ends the reply ("The answer is 42.") -> ``ok_wrapped``
    - a single integer only inside prose ("Period 1: prior BLUE = ...") -> ``embedded`` (unusable)
    - more than one integer and no leading one -> ``ambiguous``; none -> ``no_number``
    - 100 becomes 99 with status ``clamped_100`` when ``clamp_100`` is true,
      other values outside 0..99 -> ``out_of_range``
    """
    raw = "" if text is None else text
    stripped = raw.strip()
    if not stripped:
        return ParseResult("empty", None, raw)
    m = _BARE.match(stripped)
    if m:
        status = "ok_percent" if m.group(2) else "ok"
        return _range_check(int(m.group(1)), status, raw, clamp_100)
    m = _DECIMAL.match(stripped)
    if m:
        frac = float(m.group(1))
        if 0.0 <= frac <= 1.0:
            return _range_check(int(round(frac * 100)), "ok_decimal", raw, clamp_100)
    m = _LEADING.match(stripped)
    if m:
        return _range_check(int(m.group(1)), "ok_wrapped", raw, clamp_100)
    ints = _INT.findall(stripped)
    if not ints:
        return ParseResult("no_number", None, raw)
    tail = final_number(stripped)
    if tail is not None and len(ints) == 1:
        return _range_check(tail, "ok_wrapped", raw, clamp_100)
    if len(ints) == 1:
        return ParseResult("embedded", None, raw)
    return ParseResult("ambiguous", None, raw)

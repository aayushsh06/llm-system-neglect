"""Parallel, budget-capped runner for a ladder of hosted Claude models.

One asyncio event loop and one connection pool drive every model in the ladder at once:
turns within a trial are sequential (the conversation grows one period at a time), trials
run concurrently per model, and all models run concurrently. Supported protocols are the
multi-turn ``main`` mode and the single-call ``stateless`` mode, each in the ``number-only``
condition (thinking off + JSON answer schema) or the ``reasoning`` condition (adaptive
thinking + the same schema). Sampling is always the provider default: no ``temperature``,
``top_p``, ``top_k`` or ``seed`` is ever sent.

Invariants enforced here, independent of the caller:

* Spend. One shared :class:`Ledger` bounds both a project cap and a per-process stage budget.
  A conservative estimate is reserved immediately before every request and the metered cost is
  committed after it; a reservation that would breach either cap halts the run before the
  request goes out. Each process persists its own ledger file after every commit and
  :func:`baseline_spend_usd` sums those files with the single-model runner's spend, so an
  interrupted run is never forgotten.
* Global breaker. Any ``stop_reason == "refusal"``, any HTTP 403, HTTP 402 or the provider's
  spend-limit error code, and any writer failure stop every model at once.
* Per-model breaker. An unusable or non-JSON answer, an answer outside 0-99, a reply cut at
  ``max_tokens``, thinking tokens or thinking blocks in the number-only condition, a returned
  model id that differs from the one requested, an error rate above 1%, excessive retries, or
  clamped replies above 1% pause that model only.
* Concurrency. Adaptive per-model limit (start 4, +4 after every 20 clean calls while the
  rate-limit headers show headroom, halve on 429, honour ``retry-after``, ceiling 32) under a
  global connection limit; one warm-up call per model runs alone so the cached prefix is
  written once.
* Files. One writer task per result file fed by a queue, an exclusive lock per result file,
  idempotent ``(run, subject, trial, period)`` keys, resume without duplicates or gaps. A
  partially finished trial replays its stored integers as assistant turns and nothing else; a
  main-mode trial with a hole in its stored periods is skipped and reported.

Model facts (ids, prices, cache minimums, thinking switch, retirement) in :data:`LADDER`
follow the provider's model list and documentation.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import datetime as _dt
import fcntl
import hashlib
import inspect
import json
import os
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PartialRunSpec, RunSpec
from ..parsing import final_number
from ..prompts import build_main_messages, build_stateless_messages, system_prompt
from ..task import N_PERIODS, Trial
from .anthropic_backend import PRICES, PRICES_DATE, Price, _scrub, call_cost_usd
from .backends import ANSWER_SCHEMA, number_from_json
from .runner import _row, completed_rows, read_jsonl, result_key

SEED = 20251120
TRIAL_SET = "behavioral"
API_TEMPERATURE = 1.0  # label only; the parameter is never sent
CONDITION = "number-only"
MAX_TOKENS = 32
REASONING_MAX_TOKENS = 8192  # reasoning allowed: same budget as the single-model runner
MODES = ("main", "stateless")
CONDITIONS = ("number-only", "reasoning")
ANTHROPIC_VERSION = "2026-10-21"  # the SDK default API version; it has no sampling parameters
CATALOGUE_DATE = "2026-11-21"  # revision of the provider's model list that LADDER reflects

# Fingerprint of the v1 prompt as built by regime_shift.prompts: the system prompt, and every
# user turn of every (subject, trial, period) of the behavioral set.
PROMPT_V1_SYSTEM_SHA256 = "7fc5c273a02a7a43ee803bd30ba2dfd8c895f528fd096e9f1c3f1af3e8477063"
PROMPT_V1_USER_TURNS_SHA256 = "c585798071ae87b5fae0c170ce9866cf6e0fffcaf6482e1c6db68be98028a4ea"

API_RUN_FIELDS = [
    "finished", "spec_label", "file", "model", "condition", "thinking_param", "refusals",
    "refusal_log", "truncations", "subjects", "n_calls", "n_errors", "seconds", "stopped",
    "complete", "calls", "cost_usd", "input_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens", "output_tokens", "thinking_tokens", "max_dollars",
    "max_calls", "stop_reasons",
]  # fmt: skip

USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "thinking_tokens",
)


@dataclass(frozen=True)
class LadderModel:
    """One rung of the ladder: the facts about a model that the runner relies on."""

    alias: str  # file/label name (undated)
    model_id: str  # exact id sent and expected back
    display: str
    price: Price  # USD per MTok: input, output, 5m cache write, cache read
    cache_min_tokens: int  # minimum cacheable prefix
    thinking_off: Mapping[str, str]  # the `thinking` value that turns thinking off
    lifecycle: str
    retirement: str  # from the provider's deprecations page
    tokenizer: str  # "old" (<= 4.6) or "new" (4.7+ and 5.x; ~30% more tokens)
    created: str  # release date
    structured_outputs: bool = True
    note: str = ""


LADDER: dict[str, LadderModel] = {
    "claude-haiku-4-5": LadderModel(
        alias="claude-haiku-4-5",
        model_id="claude-haiku-4-5-20251001",
        display="Haiku 4.5",
        price=Price(1.0, 5.0, 1.25, 0.10),
        cache_min_tokens=4096,
        thinking_off={"type": "disabled"},
        lifecycle="active",
        retirement="not sooner than 2026-10-15",
        tokenizer="old",
        created="2025-10-15",
    ),
    "claude-sonnet-4-6": LadderModel(
        alias="claude-sonnet-4-6",
        model_id="claude-sonnet-4-6",
        display="Sonnet 4.6",
        price=Price(3.0, 15.0, 3.75, 0.30),
        cache_min_tokens=1024,
        thinking_off={"type": "disabled"},
        lifecycle="active",
        retirement="not sooner than 2027-02-17",
        tokenizer="old",
        created="2026-02-17",
        note="shares the Sonnet 4.x rate-limit bucket",
    ),
    "claude-sonnet-5": LadderModel(
        alias="claude-sonnet-5",
        model_id="claude-sonnet-5",
        display="Sonnet 5",
        price=Price(2.0, 10.0, 2.50, 0.20),
        cache_min_tokens=1024,
        thinking_off={"type": "disabled"},
        lifecycle="active",
        retirement="not sooner than 2027-06-30",
        tokenizer="new",
        created="2026-06-29",
        note="thinking is ON by default; the request sends {type: disabled} explicitly",
    ),
    "claude-haiku-5-5": LadderModel(
        alias="claude-haiku-5-5",
        model_id="claude-haiku-5-5",
        display="Haiku 5.5",
        price=PRICES["claude-haiku-5-5"],
        cache_min_tokens=512,
        thinking_off={"type": "disabled"},
        lifecycle="active",
        retirement="not sooner than 2027-10-07",
        tokenizer="new",
        created="2026-10-07",
    ),
    "claude-sonnet-5-5": LadderModel(
        alias="claude-sonnet-5-5",
        model_id="claude-sonnet-5-5",
        display="Sonnet 5.5",
        price=PRICES["claude-sonnet-5-5"],
        cache_min_tokens=512,
        thinking_off={"type": "between_tools"},
        lifecycle="active",
        retirement="not sooner than 2027-09-28",
        tokenizer="new",
        created="2026-09-28",
        note="rejects {type: disabled}; between_tools = no up-front thinking",
    ),
}

# start order: nearest retirement first
RUN_ORDER = (
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
)


class BudgetExceededError(RuntimeError):
    """A reservation would breach the project cap or the stage budget."""


class HaltedError(RuntimeError):
    """The global breaker is set; no further requests are sent."""


class CallError(RuntimeError):
    """A request failed after retries or with a non-retryable status (counted as an error)."""


class RateLimitedOutError(RuntimeError):
    """Every attempt was rate-limited; not a model error, the trial is resumed later."""


# ---------------------------------------------------------------------------------------------
# prompt / mapping checks


def prompt_fingerprint(trials_by_subject: Mapping[int, Sequence[Trial]]) -> dict[str, str]:
    """SHA-256 of the v1 system prompt and of every user turn the main protocol would send."""
    sp = system_prompt("v1")
    hp = hashlib.sha256()
    n = 0
    for s in sorted(trials_by_subject):
        for t in trials_by_subject[s]:
            for k in range(1, N_PERIODS + 1):
                msgs = build_main_messages(t, k, ["50"] * (k - 1))
                if msgs[0]["role"] != "system" or msgs[0]["content"] != sp:
                    raise AssertionError("v1 system prompt is not the first message")
                hp.update(f"{s}|{t.trial}|{k}|".encode())
                hp.update(msgs[-1]["content"].encode())
                hp.update(b"\n")
                n += 1
    return {
        "system_sha256": hashlib.sha256(sp.encode()).hexdigest(),
        "user_turns_sha256": hp.hexdigest(),
        "n_user_turns": str(n),
    }


def check_prompt_and_mapping(trials_by_subject: Mapping[int, Sequence[Trial]]) -> dict[str, str]:
    """Raise unless simulated subject i is given human subject i's trials (every trial carries
    its subject number) and the prompt builder reproduces the pinned v1 fingerprint."""
    for s, trials in trials_by_subject.items():
        bad = [t.trial for t in trials if t.subject != s]
        if bad:
            raise AssertionError(f"subject {s} holds trials of another subject: {bad[:5]}")
    fp = prompt_fingerprint(trials_by_subject)
    if fp["system_sha256"] != PROMPT_V1_SYSTEM_SHA256:
        raise AssertionError(f"system prompt changed: {fp['system_sha256']}")
    if fp["user_turns_sha256"] != PROMPT_V1_USER_TURNS_SHA256:
        raise AssertionError(f"user turns changed: {fp['user_turns_sha256']}")
    return fp


# ---------------------------------------------------------------------------------------------
# request


def make_spec(
    alias: str, n_subjects: int, mode: str = "main", condition: str = CONDITION
) -> RunSpec:
    """The run specification (file name, label) for one ladder model."""
    if mode not in MODES or condition not in CONDITIONS:
        raise ValueError(f"mode must be one of {MODES} and condition one of {CONDITIONS}")
    cls = PartialRunSpec if n_subjects < 30 else RunSpec
    return cls(
        model=alias,
        mode=mode,
        temperature=API_TEMPERATURE,
        seed=SEED,
        n_subjects=n_subjects,
        trial_set=TRIAL_SET,
        num_predict=16,
        condition=condition,
    )


def thinking_for(lm: LadderModel, condition: str) -> tuple[dict, int, str]:
    """(``thinking`` value, max_tokens, description) for a condition."""
    if condition == "reasoning":
        return (
            {"type": "adaptive"},
            REASONING_MAX_TOKENS,
            "adaptive thinking (default effort) + JSON answer schema",
        )
    return (
        dict(lm.thinking_off),
        MAX_TOKENS,
        f"thinking {dict(lm.thinking_off)['type']} + JSON answer schema",
    )


def build_request(
    lm: LadderModel, messages: Sequence[Mapping[str, str]], condition: str = CONDITION
) -> dict:
    """Messages API payload: dated model id, thinking off (or adaptive for "reasoning"), JSON
    answer schema, cache marker, provider-default sampling (no temperature / top_p / top_k /
    seed)."""
    system = [m["content"] for m in messages if m["role"] == "system"]
    turns = [
        {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
    ]
    thinking, max_tokens, _ = thinking_for(lm, condition)
    kwargs: dict = {
        "model": lm.model_id,
        "max_tokens": max_tokens,
        "messages": turns,
        "thinking": thinking,
        "output_config": {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        "cache_control": {"type": "ephemeral"},
    }
    if system:
        kwargs["system"] = "\n\n".join(system)
    for forbidden in ("temperature", "top_p", "top_k", "seed"):
        assert forbidden not in kwargs
    return kwargs


def estimate_cost_usd(lm: LadderModel, kwargs: Mapping) -> float:
    """Conservative pre-call estimate for the ledger reservation: uncached input at 3.1 chars
    per token, +30% for the newer tokenizer, +220 tokens of schema overhead, a full
    ``max_tokens`` reply, and a 1.5x margin on top. Measured period-10 prompts (1,455 tokens
    on the 5.5 tokenizer) come to about 0.8x of this estimate."""
    chars = len(kwargs.get("system", "")) + sum(len(m["content"]) for m in kwargs["messages"])
    tokens = chars / 3.1 * (1.3 if lm.tokenizer == "new" else 1.0) + 220
    return (
        1.5
        * (tokens * lm.price.input + int(kwargs.get("max_tokens", MAX_TOKENS)) * lm.price.output)
        / 1e6
    )


# ---------------------------------------------------------------------------------------------
# ledger


def baseline_spend_usd(root: Path) -> dict[str, float]:
    """Spend by every earlier process, counted once each.

    * runs of the single-model runner: the larger of (a) their rows in results/api_runs.csv
      and (b) the cost field of their result rows (rows written by this runner carry
      ``runner == "ladder"`` and csv rows written by it say "(ladder runner)"; both are excluded
      here because they are covered by the ledger files);
    * runs of this runner: the ledger file each process persists after every commit under
      results/ladder/ledgers/ (this also covers calls whose row was never written: refusals,
      truncations, pauses, crashes mid-run).
    """
    csv_total = 0.0
    p = root / "results" / "api_runs.csv"
    if p.exists():
        with p.open() as fh:
            for r in csv.DictReader(fh):
                if "(ladder runner)" in (r.get("thinking_param") or ""):
                    continue
                with contextlib.suppress(ValueError):
                    csv_total += float(r.get("cost_usd") or 0)
    jsonl_total = 0.0
    for f in (root / "results").rglob("claude-*.jsonl"):
        for r in read_jsonl(f):
            if r.get("runner") == "ladder":
                continue
            jsonl_total += float(r.get("cost_usd") or 0)
    ledgers = 0.0
    n_ledgers = 0
    for f in (root / "results" / "ladder" / "ledgers").glob("*.json"):
        try:
            ledgers += float(json.loads(f.read_text()).get("committed", 0.0))
            n_ledgers += 1
        except (ValueError, OSError):
            continue
    legacy = max(csv_total, jsonl_total)
    return {
        "csv": round(csv_total, 4),
        "jsonl": round(jsonl_total, 4),
        "legacy": round(legacy, 4),
        "ladder_ledgers": round(ledgers, 4),
        "n_ledgers": n_ledgers,
        "baseline": round(legacy + ledgers, 4),
    }


class Ledger:
    """Shared spend ledger, safe under asyncio concurrency, persisted after every commit.

    ``project_cap`` bounds baseline + committed + reserved; ``stage_budget`` bounds this
    process's committed + reserved. A reservation is taken immediately before a request is
    sent and either committed (metered cost) or released. Each process writes its own fresh
    ledger file and never loads one, so :func:`baseline_spend_usd` can sum them.
    """

    def __init__(
        self, project_cap: float, baseline: float, stage_budget: float, path: Path | None = None
    ) -> None:
        self.project_cap = float(project_cap)
        self.baseline = float(baseline)
        self.stage_budget = float(stage_budget)
        self.path = path
        self.committed = 0.0
        self.reserved = 0.0
        self.calls = 0
        self.rejected = 0
        self._lock = asyncio.Lock()
        if path is not None and path.exists():
            raise FileExistsError(
                f"ledger file already exists: {path} (one fresh file per process)"
            )
        self._persist()

    @property
    def total(self) -> float:
        return self.baseline + self.committed

    def _would_breach(self, est: float) -> str | None:
        if self.baseline + self.committed + self.reserved + est > self.project_cap:
            return (
                f"project cap ${self.project_cap:.2f}: baseline ${self.baseline:.2f} + "
                f"committed ${self.committed:.4f} + reserved ${self.reserved:.4f} + "
                f"est ${est:.4f}"
            )
        if self.committed + self.reserved + est > self.stage_budget:
            return (
                f"stage budget ${self.stage_budget:.2f}: committed ${self.committed:.4f} + "
                f"reserved ${self.reserved:.4f} + est ${est:.4f}"
            )
        return None

    async def reserve(self, est: float) -> float:
        async with self._lock:
            why = self._would_breach(est)
            if why:
                self.rejected += 1
                raise BudgetExceededError(why)
            self.reserved += est
            return est

    async def release(self, est: float) -> None:
        async with self._lock:
            self.reserved -= est

    async def commit(self, est: float, actual: float) -> float:
        async with self._lock:
            self.reserved -= est
            self.committed += actual
            self.calls += 1
            self._persist()
            return self.baseline + self.committed

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "committed": round(self.committed, 6),
                    "calls": self.calls,
                    "baseline": self.baseline,
                    "project_cap": self.project_cap,
                    "stage_budget": self.stage_budget,
                    "updated": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                }
            )
        )
        os.replace(tmp, self.path)

    def summary(self) -> dict:
        return {
            "baseline_usd": round(self.baseline, 4),
            "committed_usd": round(self.committed, 6),
            "reserved_usd": round(self.reserved, 6),
            "total_usd": round(self.total, 4),
            "project_cap_usd": self.project_cap,
            "stage_budget_usd": self.stage_budget,
            "calls": self.calls,
            "rejected": self.rejected,
            "left_under_cap_usd": round(self.project_cap - self.total, 4),
        }


# ---------------------------------------------------------------------------------------------
# breakers and concurrency


class Breakers:
    """The global halt (stops every model) and the per-model pauses."""

    def __init__(self) -> None:
        self.halt = asyncio.Event()
        self.halt_info: dict | None = None
        self.paused: dict[str, dict] = {}

    def trip_global(self, reason: str, info: dict | None = None) -> None:
        if self.halt_info is None:
            self.halt_info = {"reason": reason, "time": time.time(), **(info or {})}
        self.halt.set()

    def pause_model(self, alias: str, reason: str, info: dict | None = None) -> None:
        self.paused.setdefault(alias, {"reason": reason, "time": time.time(), **(info or {})})

    def is_paused(self, alias: str) -> bool:
        return alias in self.paused


HEADER_KEYS = (
    "anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-input-tokens-limit", "anthropic-ratelimit-input-tokens-remaining",
    "anthropic-ratelimit-output-tokens-limit", "anthropic-ratelimit-output-tokens-remaining",
    "anthropic-ratelimit-tokens-limit", "anthropic-ratelimit-tokens-remaining", "retry-after",
)  # fmt: skip


def pick_headers(headers: Mapping | None) -> dict[str, str]:
    """The rate-limit headers of a response, as strings."""
    if not headers:
        return {}
    out = {}
    for k in HEADER_KEYS:
        v = headers.get(k)
        if v is not None:
            out[k] = str(v)
    return out


def headroom(h: Mapping[str, str], fraction: float = 0.2) -> bool:
    """True when at least one (limit, remaining) pair is present and every pair present shows
    more than ``fraction`` left. Without headers there is no evidence of headroom: False."""
    seen = False
    for kind in ("requests", "input-tokens", "output-tokens", "tokens"):
        lim, rem = (
            h.get(f"anthropic-ratelimit-{kind}-limit"),
            h.get(f"anthropic-ratelimit-{kind}-remaining"),
        )
        if lim and rem:
            try:
                if float(rem) < fraction * float(lim):
                    return False
                seen = True
            except ValueError:
                continue
    return seen


class Gate:
    """Adaptive per-model concurrency: additive increase, halve on 429, honour retry-after."""

    def __init__(
        self, start: int = 4, ceiling: int = 32, floor: int = 1, ramp_every: int = 20, step: int = 4
    ) -> None:
        self.limit = start
        self.ceiling = ceiling
        self.floor = floor
        self.ramp_every = ramp_every
        self.step = step
        self.active = 0
        self.successes = 0
        self.pause_until = 0.0
        self.n429 = 0
        self.last_headers: dict[str, str] = {}
        self.history: list[tuple[float, int, str]] = [(time.time(), start, "start")]
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cond:
            while True:
                wait = self.pause_until - time.time()
                if self.active < self.limit and wait <= 0:
                    self.active += 1
                    return
                if wait > 0:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._cond.wait(), timeout=wait)
                else:
                    await self._cond.wait()

    async def release(self) -> None:
        async with self._cond:
            self.active -= 1
            self._cond.notify_all()

    async def on_success(self, headers: Mapping[str, str]) -> None:
        async with self._cond:
            if headers:
                self.last_headers = dict(headers)
            self.successes += 1
            if (
                self.successes >= self.ramp_every
                and self.limit < self.ceiling
                and headroom(self.last_headers)
            ):
                self.limit = min(self.ceiling, self.limit + self.step)
                self.successes = 0
                self.history.append((time.time(), self.limit, "ramp"))
                self._cond.notify_all()

    async def on_rate_limited(self, retry_after: float | None) -> None:
        async with self._cond:
            self.n429 += 1
            self.limit = max(self.floor, self.limit // 2)
            self.successes = 0
            self.pause_until = max(
                self.pause_until, time.time() + (retry_after if retry_after else 2.0)
            )
            self.history.append((time.time(), self.limit, f"429 retry-after={retry_after}"))
            self._cond.notify_all()


# ---------------------------------------------------------------------------------------------
# writer


class QueueWriter:
    """One task appends rows to one file in arrival order; callers only enqueue. A write
    failure is reported through ``on_failure`` and every later ``put`` raises."""

    def __init__(self, path: Path, on_failure: Callable[[Exception], None] | None = None) -> None:
        self.path = path
        self.queue: asyncio.Queue = asyncio.Queue()
        self.written = 0
        self.failure: Exception | None = None
        self._on_failure = on_failure
        self._task: asyncio.Task | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            # terminate a torn last line so the next row starts fresh
            with path.open("rb") as fh:
                fh.seek(-1, 2)
                if fh.read(1) != b"\n":
                    with path.open("a") as fh2:
                        fh2.write("\n")

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            with self.path.open("a") as fh:
                while True:
                    row = await self.queue.get()
                    if row is None:
                        break
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    self.written += 1
        except Exception as exc:  # noqa: BLE001
            self.failure = exc
            if self._on_failure is not None:
                self._on_failure(exc)

    async def put(self, row: dict) -> None:
        if self.failure is not None or (self._task is not None and self._task.done()):
            raise RuntimeError(f"writer for {self.path.name} is not running: {self.failure!r}")
        await self.queue.put(row)

    async def close(self) -> None:
        await self.queue.put(None)
        if self._task is not None:
            await self._task


# ---------------------------------------------------------------------------------------------
# per-model state


@dataclass
class ModelState:
    """Everything the runner tracks for one model: work left, counters, gate, writer."""

    lm: LadderModel
    spec: RunSpec
    path: Path
    done: dict[str, dict]
    writer: QueueWriter
    gate: Gate
    pending: list[Trial]
    calls: int = 0
    calls_started: int = 0
    errors: int = 0
    retries: int = 0
    timeouts: int = 0
    n429: int = 0
    rate_limited_giveups: int = 0
    cost_usd: float = 0.0
    usage: dict[str, int] = field(default_factory=lambda: dict.fromkeys(USAGE_KEYS, 0))
    stop_reasons: dict[str, int] = field(default_factory=dict)
    call_times: list[float] = field(default_factory=list)
    error_log: list[dict] = field(default_factory=list)
    refusal_log: list[dict] = field(default_factory=list)
    truncations: int = 0
    unusable: int = 0
    json_fallbacks: int = 0
    out_of_range: int = 0
    clamped: int = 0
    model_mismatch: int = 0
    nonzero_thinking: int = 0
    api_models: dict[str, int] = field(default_factory=dict)
    inconsistent_trials: list[int] = field(default_factory=list)
    thinking_field_missing: bool = False
    lock_handle: object = None
    started: float = 0.0
    finished: float = 0.0
    warmup_done: bool = False

    @property
    def attempts(self) -> int:
        return self.calls + self.errors

    def rate(self, window: float = 60.0) -> float:
        """Completed calls per second over the last ``window`` seconds."""
        now = time.time()
        recent = [t for t in self.call_times if t > now - window]
        if len(recent) < 2:
            return 0.0
        return len(recent) / max(1e-9, min(window, now - recent[0]))


class _Digest:
    """Minimal stand-in for a backend so ``_row`` records the model id as the digest."""

    def __init__(self, model_id: str) -> None:
        self._id = model_id

    def digest(self) -> str:
        return self._id


# ---------------------------------------------------------------------------------------------
# runner


class LadderRunner:
    """Drives every prepared model concurrently under the ledger, gates and breakers."""

    def __init__(
        self,
        client,
        ledger: Ledger,
        *,
        out_dir: Path,
        global_limit: int = 96,
        start_concurrency: int = 4,
        ceiling: int = 32,
        max_attempts: int = 6,
        log: Callable[[str], None] | None = print,
        progress_every_s: float = 30.0,
        max_calls_per_model: int | None = None,
        sleep: Callable[[float], object] | None = None,
        subject_sequential: bool = False,
    ) -> None:
        self.client = client
        self.ledger = ledger
        self.out_dir = Path(out_dir)
        self.global_sem = asyncio.Semaphore(global_limit)
        self.start_concurrency = start_concurrency
        self.ceiling = ceiling
        self.max_attempts = max_attempts
        self.log = log or (lambda s: None)
        self.progress_every_s = progress_every_s
        self.max_calls_per_model = max_calls_per_model
        self.subject_sequential = subject_sequential  # finish each subject before the next
        self.breakers = Breakers()
        self.states: dict[str, ModelState] = {}
        self.failures: list[str] = []
        self._sleep = sleep or asyncio.sleep

    # -- setup -------------------------------------------------------------------------------
    def prepare(
        self,
        aliases: Sequence[str],
        n_subjects: int,
        trials_by_subject: Mapping[int, Sequence[Trial]],
        trial_limit: int | None = None,
        mode: str = "main",
        condition: str = CONDITION,
    ) -> None:
        """Lock each result file, read what is already on disk and work out the pending trials."""
        subjects = sorted(trials_by_subject)[:n_subjects]
        for alias in sorted(aliases, key=lambda a: RUN_ORDER.index(a) if a in RUN_ORDER else 99):
            lm = LADDER[alias]
            spec = make_spec(alias, n_subjects, mode, condition)
            path = self.out_dir / spec.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            # one process per result file: an exclusive lock held for the runner's lifetime
            lock_path = path.with_suffix(".lock")
            fh = lock_path.open("a")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                raise RuntimeError(
                    f"{path.name} is being written by another process (lock {lock_path.name})"
                ) from None
            done = completed_rows(path)
            trials = [t for s_ in subjects for t in trials_by_subject[s_]]
            if trial_limit:
                trials = trials[:trial_limit]
            pending: list[Trial] = []
            inconsistent: list[int] = []
            for t in trials:
                have = [
                    result_key(spec, t.subject, t.trial, p) in done for p in range(1, N_PERIODS + 1)
                ]
                k = sum(have)
                if mode == "main" and have != [True] * k + [False] * (N_PERIODS - k):
                    inconsistent.append(t.trial)  # a hole in the middle: never resumed blindly
                    continue
                if k < N_PERIODS:
                    pending.append(t)
            writer = QueueWriter(
                path,
                on_failure=lambda exc, a=alias: self.breakers.trip_global(
                    "writer failure", {"model": a, "error": repr(exc)}
                ),
            )
            ms = ModelState(
                lm, spec, path, done, writer, Gate(self.start_concurrency, self.ceiling), pending
            )
            ms.inconsistent_trials = inconsistent
            ms.lock_handle = fh
            if inconsistent:
                self.log(
                    f"[{alias}] {len(inconsistent)} trial(s) have a hole in their stored periods "
                    f"and are skipped: {inconsistent[:10]}"
                )
            self.states[alias] = ms

    # -- one request -------------------------------------------------------------------------
    async def _request(
        self, ms: ModelState, kwargs: dict, est: float
    ) -> tuple[object, dict[str, str], float, str | None]:
        """Send one request. The ledger reservation is taken right before the send, while the
        gate and the global slot are held; it is released on every failure path except a
        timeout or an unparseable response, where the estimate is committed (the call may have
        been billed). On success the reservation is still held and the caller commits."""
        last = "no attempt"
        last_status: int | None = None
        for attempt in range(self.max_attempts):
            if self.breakers.halt.is_set():
                raise HaltedError(str(self.breakers.halt_info))
            await ms.gate.acquire()
            try:
                async with self.global_sem:
                    if self.breakers.halt.is_set():  # tripped while this task was queued
                        raise HaltedError(str(self.breakers.halt_info))
                    try:
                        await self.ledger.reserve(est)
                    except BudgetExceededError as exc:
                        self.breakers.trip_global("budget", {"detail": str(exc)})
                        raise HaltedError(str(exc)) from None
                    t0 = time.time()
                    try:
                        raw = await self.client.messages.with_raw_response.create(**kwargs)
                    except Exception as exc:  # noqa: BLE001
                        status = getattr(exc, "status_code", None)
                        last_status = status
                        hdrs = getattr(getattr(exc, "response", None), "headers", None) or {}
                        body = getattr(exc, "body", None)
                        code = None
                        if isinstance(body, dict):
                            err = body.get("error") or {}
                            code = (err.get("type"), (err.get("details") or {}).get("error_code"))
                        name = type(exc).__name__
                        last = _scrub(f"{name} {status}: {exc}")[:300]
                        if name == "APITimeoutError":
                            ms.timeouts += 1
                            await self.ledger.commit(est, est)  # may have been billed: count it
                        else:
                            await self.ledger.release(est)
                        spend_code = bool(
                            code and code[1] and str(code[1]).endswith("spend_limit_reached")
                        )
                        if status == 402 or spend_code:
                            self.breakers.trip_global(
                                "provider spend limit",
                                {"model": ms.lm.alias, "body": _scrub(str(body))[:300]},
                            )
                            raise HaltedError(last) from None
                        if status == 429:
                            try:
                                ra: float | None = float(hdrs.get("retry-after"))
                            except (TypeError, ValueError):
                                ra = None
                            ms.n429 += 1
                            await ms.gate.on_rate_limited(ra)
                            continue
                        if status == 403:
                            self.breakers.trip_global(
                                "403 (access suspended?)",
                                {
                                    "model": ms.lm.alias,
                                    "body": _scrub(str(body))[:300],
                                    "request_id": hdrs.get("request-id"),
                                },
                            )
                            raise HaltedError(last) from None
                        if status in (408, 409, 500, 502, 503, 504, 529) or name in (
                            "APIConnectionError",
                            "APITimeoutError",
                        ):
                            ms.retries += 1
                            await self._sleep(min(60.0, 2**attempt + random.random()))
                            continue
                        raise CallError(last) from None
                    duration = time.time() - t0
                    headers = pick_headers(getattr(raw, "headers", None))
                    try:
                        msg = raw.parse()
                        if inspect.isawaitable(msg):
                            msg = await msg
                    except Exception as exc:  # noqa: BLE001
                        await self.ledger.commit(est, est)  # the call was made and billed
                        raise CallError(
                            _scrub(f"response could not be parsed: {exc!r}")[:300]
                        ) from None
                    req_id = getattr(raw, "request_id", None) or (
                        getattr(raw, "headers", {}) or {}
                    ).get("request-id")
                    await ms.gate.on_success(headers)
                    return msg, headers, duration, req_id
            finally:
                await ms.gate.release()
        if last_status == 429:
            ms.rate_limited_giveups += 1
            raise RateLimitedOutError(
                f"gave up after {self.max_attempts} rate-limited attempts; last: {last}"
            )
        raise CallError(f"gave up after {self.max_attempts} attempts; last: {last}")

    def _check_error_rate(self, ms: ModelState) -> None:
        if (ms.attempts >= 100 and ms.errors / ms.attempts > 0.01) or (
            ms.attempts < 100 and ms.errors >= 2
        ):
            self.breakers.pause_model(
                ms.lm.alias, "error rate > 1%", {"errors": ms.errors, "attempts": ms.attempts}
            )
        if ms.attempts >= 50 and ms.retries > 0.5 * ms.attempts:
            self.breakers.pause_model(
                ms.lm.alias,
                "excessive retries (5xx/timeouts)",
                {"retries": ms.retries, "attempts": ms.attempts},
            )

    # -- one trial ---------------------------------------------------------------------------
    async def _run_trial(
        self, ms: ModelState, trial: Trial, only_first_period: bool = False
    ) -> None:
        """Run the pending periods of one trial in order, replaying stored integers as history."""
        previous: list[str] = []
        for period in range(1, N_PERIODS + 1):
            key = result_key(ms.spec, trial.subject, trial.trial, period)
            if key in ms.done:
                previous.append(str(ms.done[key].get("raw_text") or ""))
                continue
            if self.breakers.halt.is_set() or self.breakers.is_paused(ms.lm.alias):
                return
            if (
                self.max_calls_per_model is not None
                and ms.calls_started >= self.max_calls_per_model
            ):
                return
            ms.calls_started += 1
            if ms.spec.mode == "stateless":
                messages = build_stateless_messages(trial, period)  # one call, no history
            else:
                messages = build_main_messages(trial, period, previous)
            kwargs = build_request(ms.lm, messages, ms.spec.condition)
            est = estimate_cost_usd(ms.lm, kwargs)
            try:
                msg, headers, duration, req_id = await self._request(ms, kwargs, est)
            except HaltedError:
                return
            except RateLimitedOutError as exc:
                self.log(f"[{ms.lm.alias}] {key}: {exc}")
                return
            except CallError as exc:
                ms.errors += 1
                ms.error_log.append({"key": key, "error": str(exc), "time": time.time()})
                self.log(f"[{ms.lm.alias}] error {key}: {exc}")
                self._check_error_rate(ms)
                return
            # the reservation is still held: read the usage and commit the metered cost first
            try:
                u = getattr(msg, "usage", None)
                details = getattr(u, "output_tokens_details", None)
                usage = {
                    "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
                    "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
                    "cache_read_input_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
                    "cache_creation_input_tokens": int(
                        getattr(u, "cache_creation_input_tokens", 0) or 0
                    ),
                    "thinking_tokens": int(getattr(details, "thinking_tokens", 0) or 0),
                }
                cost = call_cost_usd(ms.lm.price, usage)
            except Exception as exc:  # noqa: BLE001
                await self.ledger.commit(est, est)
                ms.errors += 1
                ms.error_log.append(
                    {"key": key, "error": f"usage unreadable: {exc!r}", "time": time.time()}
                )
                self._check_error_rate(ms)
                return
            cumulative = await self.ledger.commit(est, cost)
            if details is None and not ms.thinking_field_missing:
                ms.thinking_field_missing = True
                self.log(
                    f"[{ms.lm.alias}] note: usage.output_tokens_details is absent; thinking is "
                    "checked from the content blocks only"
                )
            try:
                ms.cost_usd += cost
                for k, v in usage.items():
                    ms.usage[k] += v
                stop_reason = getattr(msg, "stop_reason", None)
                ms.stop_reasons[str(stop_reason)] = ms.stop_reasons.get(str(stop_reason), 0) + 1
                sd = getattr(msg, "stop_details", None)
                category = getattr(sd, "category", None) if sd is not None else None
                explanation = getattr(sd, "explanation", None) if sd is not None else None
                api_model = getattr(msg, "model", None)
                ms.api_models[str(api_model)] = ms.api_models.get(str(api_model), 0) + 1
                blocks = getattr(msg, "content", None) or []
                text = "".join(
                    getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text"
                )
                thinking_blocks = sum(
                    1 for b in blocks if getattr(b, "type", "") in ("thinking", "redacted_thinking")
                )
                info = {
                    "key": key,
                    "request_id": req_id,
                    "stop_reason": stop_reason,
                    "category": category,
                    "explanation": explanation,
                    "raw": text[:120],
                    "api_model": api_model,
                    "usage": usage,
                }
                if stop_reason == "refusal":
                    ms.refusal_log.append(
                        {"time": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"), **info}
                    )
                    self.breakers.trip_global("refusal", {"model": ms.lm.alias, **info})
                    self.log(f"[{ms.lm.alias}] REFUSAL {info}")
                    return
                if stop_reason == "max_tokens":
                    ms.truncations += 1
                    self.breakers.pause_model(ms.lm.alias, "truncated reply", info)
                    return
                if ms.spec.condition == "number-only" and (
                    usage["thinking_tokens"] > 0 or thinking_blocks
                ):
                    ms.nonzero_thinking += 1
                    self.breakers.pause_model(
                        ms.lm.alias,
                        "non-zero thinking tokens",
                        {**info, "thinking_blocks": thinking_blocks},
                    )
                    return
                if api_model != ms.lm.model_id:
                    ms.model_mismatch += 1
                    self.breakers.pause_model(ms.lm.alias, "returned model id differs", info)
                    return
                try:
                    number = number_from_json(text)
                except (ValueError, KeyError, TypeError) as exc:
                    # the schema should make this impossible; a non-JSON reply violates
                    # "number only"
                    n = final_number(text)
                    if n is None:
                        ms.unusable += 1
                        self.breakers.pause_model(
                            ms.lm.alias, "unusable answer", {**info, "json_error": repr(exc)}
                        )
                    else:
                        ms.json_fallbacks += 1
                        self.breakers.pause_model(
                            ms.lm.alias, "non-JSON answer", {**info, "json_error": repr(exc)}
                        )
                    return
                value = int(number)
                clamped = value == 100
                if clamped:
                    # same rule as the single-model runner: a reply of exactly 100 is kept with
                    # parse status clamped_100 (value 99) and replayed verbatim
                    ms.clamped += 1
                elif not 0 <= value <= 99:
                    ms.out_of_range += 1
                    self.breakers.pause_model(ms.lm.alias, "answer outside 0-99", info)
                    return
                answer_text = "100" if clamped else str(value)
                extra = {
                    "api_model": api_model,
                    "api_date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
                    "answer_condition": ms.spec.condition,
                    "thinking_param": thinking_for(ms.lm, ms.spec.condition)[2],
                    "answer_format": "json_schema",
                    "api_raw_text": text,
                    "effort": None,
                    "truncations_so_far": ms.truncations,
                    "temperature_sent": False,
                    "seed_sent": False,
                    "sampling": "provider default",
                    "anthropic_version": ANTHROPIC_VERSION,
                    "max_tokens": kwargs["max_tokens"],
                    "stop_reason": stop_reason,
                    "stop_category": category,
                    "stop_explanation": explanation,
                    "cost_usd": round(cost, 8),
                    "cumulative_cost_usd": round(cumulative, 6),
                    "prices_date": PRICES_DATE,
                    "request_id": req_id,
                    "concurrency_limit": ms.gate.limit,
                    "runner": "ladder",
                    "thinking_tokens_reported": details is not None,
                    **usage,
                }
                row = _row(
                    ms.spec, _Digest(ms.lm.model_id), trial, period, str(value), duration, extra
                )
                if row.get("parse_status") not in ("ok", "clamped_100"):
                    ms.out_of_range += 1
                    self.breakers.pause_model(
                        ms.lm.alias, f"parse status {row.get('parse_status')}", info
                    )
                    return
                ms.done[key] = row  # before the enqueue, so no other worker can redo this key
                await ms.writer.put(row)
                previous.append(answer_text)
                ms.calls += 1
                ms.call_times.append(time.time())
                if ms.calls >= 100 and ms.clamped / ms.calls > 0.01:
                    self.breakers.pause_model(
                        ms.lm.alias,
                        "clamped replies > 1%",
                        {"clamped": ms.clamped, "calls": ms.calls},
                    )
            except Exception as exc:  # noqa: BLE001
                # the cost is committed; nothing is written for this period (a rerun redoes it)
                ms.done.pop(key, None)
                ms.errors += 1
                ms.error_log.append(
                    {"key": key, "error": _scrub(repr(exc))[:300], "time": time.time()}
                )
                self.log(f"[{ms.lm.alias}] unexpected error at {key}: {_scrub(repr(exc))[:200]}")
                self._check_error_rate(ms)
                return
            if only_first_period:
                return

    # -- one model ---------------------------------------------------------------------------
    async def _run_model(self, ms: ModelState) -> None:
        """Warm the cache with one call, then fan the pending trials out over ``ceiling``
        workers."""
        ms.started = time.time()
        ms.writer.start()
        try:
            queue: asyncio.Queue = asyncio.Queue()
            if ms.pending and not self.breakers.halt.is_set():
                # cache warm-up: one call alone, so the shared prefix is written once
                await self._run_trial(ms, ms.pending[0], only_first_period=True)
                ms.warmup_done = True

            async def worker() -> None:
                while not self.breakers.halt.is_set() and not self.breakers.is_paused(ms.lm.alias):
                    try:
                        t = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    await self._run_trial(ms, t)

            if self.subject_sequential:
                groups: dict[int, list[Trial]] = {}
                for t in ms.pending:
                    groups.setdefault(t.subject, []).append(t)
                batches = [groups[k] for k in sorted(groups)]
            else:
                batches = [list(ms.pending)]
            for batch in batches:
                if self.breakers.halt.is_set() or self.breakers.is_paused(ms.lm.alias):
                    break
                for t in batch:
                    queue.put_nowait(t)
                results = await asyncio.gather(
                    *(worker() for _ in range(self.ceiling)), return_exceptions=True
                )
                for r in results:
                    if isinstance(r, BaseException):
                        self.breakers.pause_model(
                            ms.lm.alias, "worker crashed", {"error": _scrub(repr(r))[:300]}
                        )
                        self.log(f"[{ms.lm.alias}] worker crashed: {_scrub(repr(r))[:300]}")
        finally:
            ms.finished = time.time()
            await ms.writer.close()

    async def _progress(self) -> None:
        while True:
            await asyncio.sleep(self.progress_every_s)
            parts = []
            for a, ms in self.states.items():
                flag = (
                    " HALT"
                    if self.breakers.halt.is_set()
                    else (" PAUSED" if self.breakers.is_paused(a) else "")
                )
                parts.append(
                    f"{a}: {len(ms.done)} rows, {ms.rate():.1f}/s, c={ms.gate.limit}, "
                    f"${ms.cost_usd:.3f}{flag}"
                )
            self.log(f"[ladder] total ${self.ledger.total:.3f} | " + " | ".join(parts))

    async def run(self) -> dict:
        """Run every prepared model to completion, halt or pause; return :meth:`summary`."""
        prog = asyncio.create_task(self._progress())
        try:
            results = await asyncio.gather(
                *(self._run_model(ms) for ms in self.states.values()), return_exceptions=True
            )
            for ms, r in zip(self.states.values(), results, strict=True):
                if isinstance(r, BaseException):
                    self.failures.append(f"{ms.lm.alias}: {_scrub(repr(r))[:300]}")
                    self.log(f"[{ms.lm.alias}] model task failed: {_scrub(repr(r))[:300]}")
        finally:
            prog.cancel()
            for ms in self.states.values():
                with contextlib.suppress(OSError):
                    if ms.lock_handle is not None:
                        ms.lock_handle.close()
        return self.summary()

    # -- reporting ---------------------------------------------------------------------------
    def summary(self) -> dict:
        halt = self.breakers.halt_info
        out: dict = {
            "ledger": self.ledger.summary(),
            "halt": halt,
            "stopped_by_budget": bool(halt and halt.get("reason") == "budget"),
            "paused": self.breakers.paused,
            "failures": list(self.failures),
            "models": {},
        }
        for a, ms in self.states.items():
            secs = max(1e-9, (ms.finished or time.time()) - ms.started) if ms.started else 0.0
            n_in = (
                ms.usage["input_tokens"]
                + ms.usage["cache_read_input_tokens"]
                + ms.usage["cache_creation_input_tokens"]
            )
            out["models"][a] = {
                "model_id": ms.lm.model_id,
                "file": str(ms.path),
                "rows_on_disk": len(ms.done),
                "calls": ms.calls,
                "calls_started": ms.calls_started,
                "errors": ms.errors,
                "retries": ms.retries,
                "timeouts": ms.timeouts,
                "n429": ms.n429,
                "rate_limited_giveups": ms.rate_limited_giveups,
                "cost_usd": round(ms.cost_usd, 6),
                "cost_per_call_usd": round(ms.cost_usd / ms.calls, 7) if ms.calls else None,
                "seconds": round(secs, 1),
                "calls_per_second": round(ms.calls / secs, 3) if ms.calls and secs else 0.0,
                "concurrency_final": ms.gate.limit,
                "concurrency_history": ms.gate.history[-12:],
                "last_headers": ms.gate.last_headers,
                "usage": dict(ms.usage),
                "cached_input_share": (
                    round(ms.usage["cache_read_input_tokens"] / n_in, 3) if n_in else None
                ),
                "stop_reasons": ms.stop_reasons,
                "api_models": ms.api_models,
                "truncations": ms.truncations,
                "unusable": ms.unusable,
                "json_fallbacks": ms.json_fallbacks,
                "out_of_range": ms.out_of_range,
                "clamped": ms.clamped,
                "model_mismatch": ms.model_mismatch,
                "nonzero_thinking": ms.nonzero_thinking,
                "thinking_field_missing": ms.thinking_field_missing,
                "inconsistent_trials": ms.inconsistent_trials[:20],
                "refusals": len(ms.refusal_log),
                "error_log": ms.error_log[:20],
                "warmup_done": ms.warmup_done,
            }
        return out


# ---------------------------------------------------------------------------------------------
# post-run checks


def completeness(path: Path, spec: RunSpec, trials: Sequence[Trial]) -> dict:
    """Compare the keys on disk with the keys expected for ``trials``."""
    rows = read_jsonl(path)
    keys = [r.get("key") for r in rows]
    expected = {
        result_key(spec, t.subject, t.trial, p) for t in trials for p in range(1, N_PERIODS + 1)
    }
    seen: dict[str, int] = {}
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
    dups = sorted(k for k, n in seen.items() if n > 1)
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    return {
        "rows": len(rows),
        "expected": len(expected),
        "duplicates": len(dups),
        "missing": len(missing),
        "extra": len(extra),
        "complete": not dups and not missing and not extra,
        "missing_sample": missing[:5],
        "duplicate_sample": dups[:5],
    }


def replay_consistency(path: Path) -> dict:
    """Every row's stored answer is a bare integer in 0-100 (what a rerun would replay; 100 is
    the clamped case)."""
    rows = read_jsonl(path)
    bad = [
        r["key"]
        for r in rows
        if not (str(r.get("raw_text", "")).isdigit() and 0 <= int(r["raw_text"]) <= 100)
    ]
    return {"rows": len(rows), "non_integer_raw_text": len(bad), "sample": bad[:5]}


def compare_with_existing(new_path: Path, old_path: Path) -> dict:
    """Row-by-row comparison on shared keys (share identical, largest and mean absolute
    difference) plus distribution summaries for sampling models."""
    new = {r["key"]: r for r in read_jsonl(new_path)}
    old = {r["key"]: r for r in read_jsonl(old_path)}
    shared = sorted(set(new) & set(old))
    pairs = [
        (float(new[k]["value"]), float(old[k]["value"]))
        for k in shared
        if new[k].get("value") is not None and old[k].get("value") is not None
    ]
    if not pairs:
        return {"shared": 0}
    diffs = [abs(a - b) for a, b in pairs]
    by_period: dict[int, list[float]] = {}
    for k in shared:
        by_period.setdefault(int(new[k]["period"]), []).append(
            float(new[k]["value"]) - float(old[k]["value"])
        )
    return {
        "shared": len(shared),
        "new_only": len(set(new) - set(old)),
        "share_identical": round(sum(d == 0 for d in diffs) / len(diffs), 4),
        "max_abs_diff": max(diffs),
        "mean_abs_diff": round(statistics.mean(diffs), 3),
        "new_mean": round(statistics.mean(a for a, _ in pairs), 3),
        "old_mean": round(statistics.mean(b for _, b in pairs), 3),
        "new_sd": round(statistics.pstdev([a for a, _ in pairs]), 3),
        "old_sd": round(statistics.pstdev([b for _, b in pairs]), 3),
        "mean_diff_by_period": {
            p: round(statistics.mean(v), 2) for p, v in sorted(by_period.items())
        },
    }


def projection(
    lm: LadderModel,
    calls: int = 21600,
    input_tokens_per_call_new: float = 1046.0,
    output_tokens_per_call: float = 10.0,
) -> dict:
    """Full-run cost from the measured 5.5-tokenizer input length (1,046 tokens per call on
    average, 683 at period 1 and 1,455 at period 10): uncached, and with caching applied only
    where the prefix can reach the model's minimum cacheable length."""
    scale = 1.0 if lm.tokenizer == "new" else 1 / 1.3
    per_call = input_tokens_per_call_new * scale
    uncached = calls * (per_call * lm.price.input + output_tokens_per_call * lm.price.output) / 1e6
    cacheable = [k for k in range(1, 11) if (683 + 86 * (k - 1)) * scale >= lm.cache_min_tokens]
    share = len(cacheable) / 10
    cached_cost = (
        calls
        * (
            (1 - share) * per_call * lm.price.input
            + share
            * (0.9 * per_call * lm.price.cache_read + 0.1 * per_call * lm.price.cache_write_5m)
            + output_tokens_per_call * lm.price.output
        )
        / 1e6
    )
    return {
        "cost_uncached_usd": round(uncached, 2),
        "cost_cached_usd": round(cached_cost, 2),
        "cacheable_periods": len(cacheable),
        "input_tokens_per_call": round(per_call),
    }


def catalogue_table() -> str:
    """Markdown table of the ladder with a projected full-run cost per model."""
    lines = [
        "| alias | dated id | lifecycle | retirement | thinking off | structured output "
        "| cache min (tok) | $/MTok in/out | tokenizer | full run uncached / cached |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in RUN_ORDER:
        lm = LADDER[a]
        pr = projection(lm)
        lines.append(
            f"| {a} | {lm.model_id} | {lm.lifecycle} | {lm.retirement} "
            f"| {dict(lm.thinking_off)['type']} | {'yes' if lm.structured_outputs else 'no'} "
            f"| {lm.cache_min_tokens} | {lm.price.input:g} / {lm.price.output:g} "
            f"| {lm.tokenizer} | ${pr['cost_uncached_usd']:.0f} / ${pr['cost_cached_usd']:.0f} |"
        )
    return "\n".join(lines)

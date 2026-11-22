"""Anthropic Messages API backend (Claude models) behind the ``Backend`` interface.

Same prompt, same multi-turn protocol, same JSONL logging and resume as the Ollama backend;
only the transport differs. Request shape (per the provider's documentation):

* Sampling is the provider default. The 5.x models reject ``temperature`` / ``top_p`` /
  ``top_k``, so the runner's ``temperature`` and ``seed`` arguments are never sent; every row
  records ``temperature_sent = False`` and ``seed_sent = False``.
* Condition "number-only": thinking off plus a JSON answer schema (``output_config.format``)
  so the reply is a single integer. ``claude-haiku-5-5`` accepts ``{"type": "disabled"}``;
  ``claude-sonnet-5-5`` rejects it and uses ``{"type": "between_tools"}`` (no up-front
  thinking); ``claude-opus-5-5`` and ``claude-fable-5-2`` cannot turn thinking off at all.
  Without the schema these models write their arithmetic into the visible text.
* Conditions "reasoning[-effort]": ``{"type": "adaptive"}`` thinking, optional effort, with
  the same JSON answer schema (the grammar constrains only the final text block; thinking is
  unconstrained). Only the number is kept and replayed, never any reasoning text; a reply cut
  at ``max_tokens`` is an error, not a number.
* Thinking tokens are billed as output tokens and reported in
  ``usage.output_tokens_details.thinking_tokens``.
* Prompt caching: a top-level ``cache_control = {"type": "ephemeral"}`` caches the prefix
  automatically (5-minute TTL; minimum cacheable prefix 512 tokens on these models).
* Rate limits answer 429 with a ``retry-after`` header; 529 means overloaded.

The API key is read by the SDK from ``ANTHROPIC_API_KEY``. This module never reads the key
into its own state, never logs it, and scrubs it from any exception text it re-raises.
"""

from __future__ import annotations

import datetime as _dt
import os
import random
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..parsing import final_number
from .backends import ANSWER_SCHEMA, ChatResponse, Message, StopRun, number_from_json

PRICES_DATE = "2026-11-21"  # the pricing-page revision the PRICES table reflects; written to rows


@dataclass(frozen=True)
class Price:
    """USD per million tokens, from the provider's pricing page."""

    input: float
    output: float
    cache_write_5m: float
    cache_read: float


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, 0.125, 0.01),  # prompts up to 100k tokens
    "claude-sonnet-5-5": Price(2.0, 10.0, 2.50, 0.10),
    "claude-opus-5-5": Price(4.0, 20.0, 5.00, 0.20),
    "claude-fable-5-2": Price(8.0, 40.0, 10.00, 0.20),
}

CONDITIONS = ("number-only", "reasoning", "reasoning-low", "reasoning-medium", "reasoning-high")


class ThinkingNotDisableableError(ValueError):
    """The model cannot turn thinking off, so it cannot run the number-only condition."""


class MissingAPIKeyError(RuntimeError):
    """``ANTHROPIC_API_KEY`` is not set; no client is built."""


class RefusedError(RuntimeError):
    """The API answered with ``stop_reason == "refusal"`` (a safety classifier intervened).

    Raised instead of returning an empty reply so the runner logs it as an error and never
    replays an empty assistant turn. Repeated refusals can get a model's access suspended for
    the whole organisation, so the backend stops the run after ``max_refusals`` (default 1).
    """


class UnusableReplyError(RuntimeError):
    """No final number could be taken from the reply (or the JSON answer was malformed).

    Raised so that nothing but a number is ever replayed as an assistant turn.
    """


def thinking_config(model: str, condition: str) -> tuple[dict, dict | None, str]:
    """Return (``thinking`` parameter, ``output_config`` or None, human-readable description).

    ``number-only``: thinking off (haiku: disabled; sonnet: between_tools; opus/fable cannot)
    plus the JSON answer schema. ``reasoning[-effort]``: adaptive thinking, optional effort,
    and the same JSON answer schema.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    fmt = {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
    if condition == "number-only":
        if model.startswith("claude-haiku-5-5"):
            return {"type": "disabled"}, fmt, "thinking disabled + JSON answer schema"
        if model.startswith("claude-sonnet-5-5"):
            return (
                {"type": "between_tools"},
                fmt,
                "thinking between_tools (up-front thinking off) + JSON answer schema",
            )
        raise ThinkingNotDisableableError(
            f"{model} cannot turn thinking off (adaptive thinking is always on), so it cannot "
            "run the number-only condition; use 'reasoning-low' and report the thinking tokens"
        )
    # reasoning allowed: the grammar applies only to the final text block; thinking is free
    if condition == "reasoning":
        return {"type": "adaptive"}, fmt, "adaptive thinking (default effort) + JSON answer schema"
    effort = condition.split("-", 1)[1]
    return (
        {"type": "adaptive"},
        {"effort": effort, **fmt},
        f"adaptive thinking (effort {effort}) + JSON answer schema",
    )


def call_cost_usd(price: Price, usage: dict) -> float:
    """Metered cost of one call from its ``usage`` counters."""
    return (
        usage.get("input_tokens", 0) * price.input
        + usage.get("cache_creation_input_tokens", 0) * price.cache_write_5m
        + usage.get("cache_read_input_tokens", 0) * price.cache_read
        + usage.get("output_tokens", 0) * price.output
    ) / 1e6


def load_env_file(path: str | os.PathLike = ".env", override: bool = False) -> list[str]:
    """Load KEY=VALUE lines from a .env file into ``os.environ``; returns the names set.

    Values are never printed or returned. Lines starting with ``#`` and blank lines are
    ignored; ``export KEY=VALUE`` and quoted values are accepted. Variables already in the
    environment win unless ``override`` is True.
    """
    p = Path(path)
    if not p.exists():
        return []
    names: list[str] = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
            names.append(key)
    return names


def _scrub(text: str) -> str:
    """Replace the API key, if it appears in ``text``, with a placeholder."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    return text.replace(key, "<ANTHROPIC_API_KEY>") if key and key in text else text


@dataclass
class SpendTracker:
    """Running totals for one backend; ``check`` enforces the dollar and call caps."""

    max_dollars: float
    max_calls: int
    calls: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    stop_reasons: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        """Raise StopRun before a call would exceed either cap."""
        if self.cost_usd >= self.max_dollars:
            raise StopRun(f"budget: ${self.cost_usd:.4f} spent >= cap ${self.max_dollars:.2f}")
        if self.calls >= self.max_calls:
            raise StopRun(f"budget: {self.calls} calls >= cap {self.max_calls}")

    def add(self, usage: dict, cost: float, stop_reason: str | None) -> None:
        self.calls += 1
        self.cost_usd += cost
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.thinking_tokens += usage.get("thinking_tokens", 0)
        self.cache_read_input_tokens += usage.get("cache_read_input_tokens", 0)
        self.cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0)
        self.stop_reasons[stop_reason or "none"] = (
            self.stop_reasons.get(stop_reason or "none", 0) + 1
        )

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "stop_reasons": dict(self.stop_reasons),
            "max_dollars": self.max_dollars,
            "max_calls": self.max_calls,
        }

    def format(self) -> str:
        return (
            f"spend ${self.cost_usd:.3f} of ${self.max_dollars:.2f} over {self.calls} calls "
            f"(in {self.input_tokens:,} + cached {self.cache_read_input_tokens:,}, "
            f"out {self.output_tokens:,}, thinking {self.thinking_tokens:,})"
        )


class AnthropicBackend:
    """Claude via the Messages API with hard spend caps, caching and rate-limit backoff."""

    def __init__(
        self,
        model: str,
        condition: str = "number-only",
        *,
        max_dollars: float,
        max_calls: int,
        max_tokens: int | None = None,
        client=None,
        prices: dict[str, Price] | None = None,
        cache: bool = True,
        max_retries: int = 8,
        timeout: float = 120.0,
        backoff_attempts: int = 6,
        max_refusals: int = 1,
    ) -> None:
        prices = prices or PRICES
        if model not in prices:
            raise ValueError(f"no price on file for {model}; known: {sorted(prices)}")
        self.model = model
        self.condition = condition
        self.thinking_param, self.output_config, self.thinking_desc = thinking_config(
            model, condition
        )
        self.name = f"{model}@{condition}"
        self.max_tokens = int(max_tokens or (32 if condition == "number-only" else 8192))
        self.refusal_log: list[dict] = []
        self.truncations = 0
        self.price = prices[model]
        self.cache = cache
        self.backoff_attempts = backoff_attempts
        self.spend = SpendTracker(max_dollars=float(max_dollars), max_calls=int(max_calls))
        self.max_refusals = int(max_refusals)
        self.refusals = 0
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise MissingAPIKeyError(
                    "ANTHROPIC_API_KEY is not set in the environment; set it and rerun"
                )
            import anthropic  # the SDK reads the key from the environment itself

            client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        self.client = client

    # -- helpers -----------------------------------------------------------------
    def request_kwargs(self, messages: Sequence[Message]) -> dict:
        """Messages API payload: system prompt, turns, thinking, answer schema, cache marker."""
        system = [m["content"] for m in messages if m["role"] == "system"]
        turns = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": turns,
            "thinking": dict(self.thinking_param),
        }
        if system:
            kwargs["system"] = "\n\n".join(system)
        if self.output_config:
            kwargs["output_config"] = dict(self.output_config)
        if self.cache:
            kwargs["cache_control"] = {"type": "ephemeral"}
        return kwargs

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in (408, 409, 429, 500, 502, 503, 504, 529):
            return True
        name = type(exc).__name__
        return name in ("APIConnectionError", "APITimeoutError", "RateLimitError")

    def _create(self, kwargs: dict):
        """Send one request with exponential backoff; a non-retryable error is re-raised with
        the key scrubbed."""
        last: Exception | None = None
        for attempt in range(self.backoff_attempts):
            try:
                return self.client.messages.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if not self._retryable(exc):
                    raise RuntimeError(_scrub(f"{type(exc).__name__}: {exc}")) from None
                retry_after = None
                headers = getattr(getattr(exc, "response", None), "headers", None)
                if headers is not None:
                    try:
                        retry_after = float(headers.get("retry-after"))
                    except (TypeError, ValueError):
                        retry_after = None
                wait = retry_after if retry_after else min(60.0, 2**attempt + random.random())
                time.sleep(wait)
        raise RuntimeError(
            _scrub(f"gave up after {self.backoff_attempts} attempts: {type(last).__name__}: {last}")
        )

    # -- Backend interface -------------------------------------------------------
    def chat(
        self, messages: Sequence[Message], *, temperature: float, seed: int, num_predict: int
    ) -> ChatResponse:
        """One call. ``temperature``, ``seed`` and ``num_predict`` are accepted for the
        ``Backend`` interface and deliberately not sent."""
        with self.spend.lock:
            self.spend.check()
        kwargs = self.request_kwargs(messages)
        t0 = time.time()
        resp = self._create(kwargs)
        duration = time.time() - t0
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        u = resp.usage
        details = getattr(u, "output_tokens_details", None)
        usage = {
            "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
            "cache_creation_input_tokens": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            "thinking_tokens": int(getattr(details, "thinking_tokens", 0) or 0),
        }
        cost = call_cost_usd(self.price, usage)
        stop_reason = getattr(resp, "stop_reason", None)
        details = getattr(resp, "stop_details", None)
        stop_category = getattr(details, "category", None) if details is not None else None
        with self.spend.lock:
            self.spend.add(usage, cost, stop_reason)
            cumulative = self.spend.cost_usd
            if stop_reason == "refusal":
                self.refusals += 1
                n_ref = self.refusals
        stop_explanation = getattr(details, "explanation", None) if details is not None else None
        if stop_reason == "refusal":
            self.refusal_log.append(
                {
                    "time": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                    "category": stop_category,
                    "explanation": stop_explanation,
                    "n_messages": len(kwargs["messages"]),
                    "request_id": getattr(resp, "_request_id", None),
                }
            )
            msg = (
                f"stop_reason=refusal (category {stop_category!r}, "
                f"explanation {stop_explanation!r}, request {getattr(resp, '_request_id', None)})"
            )
            if n_ref >= self.max_refusals:
                raise StopRun(f"refusals: {n_ref} >= cap {self.max_refusals}; {msg}")
            raise RefusedError(msg)
        if stop_reason == "max_tokens":
            with self.spend.lock:
                self.truncations += 1
            raise UnusableReplyError(
                f"reply truncated at max_tokens={self.max_tokens} (output {usage['output_tokens']} "
                f"tokens, {usage['thinking_tokens']} thinking); not replayed"
            )
        api_raw_text = text
        try:
            text = number_from_json(text)  # both conditions answer {"probability": N}
        except (ValueError, KeyError, TypeError) as exc:
            n = final_number(text)  # last resort: a bare final integer in a non-JSON reply
            if n is None:
                raise UnusableReplyError(
                    f"JSON answer not parseable: {exc} ({api_raw_text[:80]!r})"
                ) from None
            text = str(n)
        # only the number is kept and replayed as the assistant turn, never reasoning text
        extra = {
            "api_model": getattr(resp, "model", self.model),
            "api_date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
            "answer_condition": self.condition,  # "condition" is the trial's q/d cell
            "thinking_param": self.thinking_desc,
            "answer_format": "json_schema",
            "api_raw_text": api_raw_text,
            "effort": (self.output_config or {}).get("effort"),
            "truncations_so_far": self.truncations,
            "temperature_sent": False,
            "seed_sent": False,
            "max_tokens": self.max_tokens,
            "stop_reason": stop_reason,
            "stop_category": stop_category,
            "stop_explanation": stop_explanation,
            "cost_usd": round(cost, 8),
            "cumulative_cost_usd": round(cumulative, 6),
            "prices_date": PRICES_DATE,
            **usage,
        }
        return ChatResponse(
            text=text,
            model=self.model,
            digest=extra["api_model"],
            duration_s=duration,
            eval_count=usage["output_tokens"],
            extra=extra,
        )

    def digest(self) -> str | None:
        return self.model

    def format_spend(self) -> str:
        return self.spend.format()

"""Model backends: Ollama, plus two mocks that answer from the task mathematics.

All backends expose ``chat(messages, temperature=..., seed=..., num_predict=...)``
and return a :class:`ChatResponse`. The mocks read the task state back out of
the prompt text, so they exercise the same prompt the real models see.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..bayes import forward_filter, system_neglect_posterior
from ..prompts import parse_prompt_state
from ..task import d_level, q_level

Message = dict[str, str]


# JSON answer schema for the "number only" condition (API models and the Ollama bridge check).
# It constrains the shape of the reply; the prompt text itself is unchanged.
ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {"probability": {"type": "integer"}},
    "required": ["probability"],
    "additionalProperties": False,
}


def number_from_json(text: str) -> str:
    """``'{"probability": 42}'`` -> ``'42'``; raises ValueError when it is not that shape."""
    obj = json.loads(text)
    value = obj["probability"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"probability is not a number: {value!r}")
    return str(int(round(value)))


class StopRun(RuntimeError):
    """Raised by a backend when a hard cap (spend, calls) is reached: the run stops cleanly."""


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model: str
    digest: str | None = None
    duration_s: float | None = None
    eval_count: int | None = None
    extra: dict | None = None  # per-call fields merged into the JSONL row (API usage etc.)


class Backend(Protocol):
    name: str

    def chat(
        self, messages: Sequence[Message], *, temperature: float, seed: int, num_predict: int
    ) -> ChatResponse: ...

    def digest(self) -> str | None: ...


def _format_answer(p: float) -> str:
    """A probability as the whole-percent integer 0..99 a human would key in."""
    return str(int(np.clip(np.rint(p * 100), 0, 99)))


class BayesianMock:
    """Answers with the exact Bayesian posterior, rounded to a whole percent."""

    name = "mock-bayesian"

    def chat(
        self, messages: Sequence[Message], *, temperature: float, seed: int, num_predict: int
    ) -> ChatResponse:
        q, d, signals = parse_prompt_state(messages)
        p = forward_filter(q, d, signals)[-1]
        return ChatResponse(_format_answer(p), self.name, "synthetic", 0.0, 1)

    def digest(self) -> str | None:
        return "synthetic"


class NeglectMock:
    """Answers from the system-neglect model with known alphas and betas plus Gaussian noise.

    ``junk_rate`` makes a fraction of replies unparseable, to exercise the
    failure accounting. Deterministic given ``seed``.
    """

    name = "mock-neglect"

    def __init__(
        self,
        alphas: Sequence[float] = (3.0, 1.0, 0.6),
        betas: Sequence[float] = (1.6, 0.8, 0.5),
        noise_sd: float = 0.05,
        junk_rate: float = 0.0,
    ) -> None:
        self.alphas = tuple(float(a) for a in alphas)
        self.betas = tuple(float(b) for b in betas)
        self.noise_sd = float(noise_sd)
        self.junk_rate = float(junk_rate)

    def chat(
        self, messages: Sequence[Message], *, temperature: float, seed: int, num_predict: int
    ) -> ChatResponse:
        q, d, signals = parse_prompt_state(messages)
        rng = np.random.default_rng(seed)
        if self.junk_rate and rng.random() < self.junk_rate:
            junk = rng.choice(["I am not sure.", "Probably around 40 or 50.", "", "150", "N/A"])
            return ChatResponse(str(junk), self.name, "synthetic", 0.0, 1)
        p = system_neglect_posterior(
            q, d, signals, self.alphas[q_level(q)], self.betas[d_level(d)]
        )[-1]
        scale = self.noise_sd * (temperature / 0.7 if temperature > 0 else 0.0)
        p = float(np.clip(p + rng.normal(0.0, scale) if scale > 0 else p, 0.0, 0.99))
        return ChatResponse(_format_answer(p), self.name, "synthetic", 0.0, 1)

    def digest(self) -> str | None:
        return "synthetic"


class OllamaBackend:
    """Ollama ``/api/chat`` client. Temperature, seed and ``num_predict`` are set on every call."""

    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
        retries: int = 3,
        num_ctx: int = 2048,
        keep_alive: str = "30m",
        format: dict | None = None,
        think: bool | None = None,
    ) -> None:
        import requests

        self._requests = requests
        self.format = format  # JSON schema -> Ollama structured output ("number only")
        # Ollama's `think` switch. Reasoning models are run with think=False so that the
        # reply is the answer itself and the token limit is not spent on hidden reasoning.
        self.think = think
        self.name = model
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._digest: str | None = None

    def chat(
        self, messages: Sequence[Message], *, temperature: float, seed: int, num_predict: int
    ) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": float(temperature),
                "seed": int(seed),
                "num_predict": int(num_predict),
                "num_ctx": int(self.num_ctx),
            },
        }
        if self.format is not None:
            payload["format"] = self.format
        if self.think is not None:
            payload["think"] = self.think
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                r = self._requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
                text = body.get("message", {}).get("content", "")
                dur = body.get("total_duration")
                extra = None
                thinking = body.get("message", {}).get("thinking") or ""
                if self.think is not None or thinking:
                    extra = {"think_option": self.think, "thinking_text_len": len(thinking)}
                if self.format is not None:
                    extra = {**(extra or {}), "api_raw_text": text, "answer_format": "json_schema"}
                    try:
                        text = number_from_json(text)
                    except (ValueError, KeyError, TypeError) as exc:
                        text = ""  # logged as an unusable reply, never as a number
                        extra["json_error"] = f"{type(exc).__name__}: {exc}"[:200]
                return ChatResponse(
                    text=text,
                    model=self.model,
                    digest=self.digest(),
                    duration_s=(dur / 1e9) if dur else time.time() - t0,
                    eval_count=body.get("eval_count"),
                    extra=extra,
                )
            except Exception as exc:  # network / server errors: retry with backoff
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"ollama chat failed after {self.retries} attempts: {last_exc}")

    def digest(self) -> str | None:
        if self._digest is None:
            try:
                r = self._requests.get(f"{self.host}/api/tags", timeout=30)
                r.raise_for_status()
                for m in r.json().get("models", []):
                    if m.get("name") == self.model or m.get("model") == self.model:
                        self._digest = m.get("digest")
                        break
            except Exception:
                return None
        return self._digest

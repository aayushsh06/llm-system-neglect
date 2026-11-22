"""Anthropic backend: mocked API (no network), key never written, spend cap stops the run."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from regime_shift.collection import anthropic_backend as ab
from regime_shift.collection.backends import ANSWER_SCHEMA, StopRun
from regime_shift.collection.runner import read_jsonl, run_spec
from regime_shift.config import PartialRunSpec, RunSpec
from regime_shift.results import load_results
from regime_shift.task import generate_subject_trials

FAKE_KEY = "sk-ant-api03-FAKE-SECRET-KEY-0123456789"


def _usage(inp=500, out=9, think=0, cread=0, cwrite=0):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=cread,
        cache_creation_input_tokens=cwrite,
        output_tokens_details=SimpleNamespace(thinking_tokens=think),
    )


class FakeMessages:
    """Stands in for ``anthropic.Anthropic().messages``; records every request."""

    def __init__(
        self,
        text='{"probability": 42}',
        usage=None,
        stop_reason="end_turn",
        fail_on: int | None = None,
        stop_details=None,
    ):
        self.text, self.usage, self.stop_reason = text, usage, stop_reason
        self.fail_on, self.stop_details = fail_on, stop_details
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise ValueError(f"poisoned error mentioning {FAKE_KEY}")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            model=kwargs["model"],
            stop_reason=self.stop_reason,
            stop_details=self.stop_details,
            usage=self.usage or _usage(),
            _request_id="req_test",
        )


def fake_client(**kw):
    return SimpleNamespace(messages=FakeMessages(**kw))


def spec(model="claude-sonnet-5-5", condition="number-only", n_subjects=1, partial=False):
    cls = PartialRunSpec if partial else RunSpec
    return cls(model, "main", 1.0, 7, n_subjects, "behavioral", 16, condition=condition)


def trials(n_subjects=1):
    return {s: generate_subject_trials(s, 11, n_blocks=1) for s in range(1, n_subjects + 1)}


def test_conditions_follow_the_docs():
    fmt = {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
    t, oc, _ = ab.thinking_config("claude-haiku-5-5", "number-only")
    assert t == {"type": "disabled"} and oc == fmt
    t, oc, _ = ab.thinking_config("claude-sonnet-5-5", "number-only")
    assert t == {"type": "between_tools"} and oc == fmt
    for m in ("claude-opus-5-5", "claude-fable-5-2"):
        with pytest.raises(ab.ThinkingNotDisableableError):
            ab.thinking_config(m, "number-only")
    t, oc, _ = ab.thinking_config("claude-opus-5-5", "reasoning-low")
    assert t == {"type": "adaptive"} and oc == {"effort": "low", **fmt}
    assert ab.thinking_config("claude-opus-5-5", "reasoning")[:2] == ({"type": "adaptive"}, fmt)
    with pytest.raises(ValueError):
        ab.thinking_config("claude-sonnet-5-5", "off")


def test_cost_formula_uses_all_four_prices():
    p = ab.Price(2.0, 10.0, 2.5, 0.1)
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cache_creation_input_tokens": 200_000,
        "cache_read_input_tokens": 3_000_000,
    }
    assert ab.call_cost_usd(p, usage) == pytest.approx(2.0 + 1.0 + 0.5 + 0.3)


def test_number_only_run_writes_usage_fields_and_sends_no_temperature(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    client = fake_client(usage=_usage(inp=600, out=9, think=0, cread=400, cwrite=50))
    be = ab.AnthropicBackend(
        "claude-sonnet-5-5", "number-only", max_dollars=5, max_calls=10_000, client=client
    )
    sp = spec()
    st = run_spec(sp, be, trials(), tmp_path, progress=None)
    assert st.complete and st.n_calls == 90 and st.stopped is None
    rows = read_jsonl(tmp_path / sp.filename)
    r = rows[0]
    assert r["api_model"] == "claude-sonnet-5-5" and r["stop_reason"] == "end_turn"
    assert r["answer_condition"] == "number-only" and r["answer_format"] == "json_schema"
    assert r["condition"] in range(1, 10)  # the trial cell is untouched
    assert r["raw_text"] == "42" and r["api_raw_text"] == '{"probability": 42}'
    assert r["parse_status"] == "ok" and r["estimate"] == pytest.approx(0.42)
    assert r["temperature_sent"] is False and r["seed_sent"] is False
    assert (r["input_tokens"], r["output_tokens"], r["thinking_tokens"]) == (600, 9, 0)
    assert r["cache_read_input_tokens"] == 400 and r["cache_creation_input_tokens"] == 50
    expected = (600 * 2.0 + 50 * 2.5 + 400 * 0.1 + 9 * 10.0) / 1e6
    assert r["cost_usd"] == pytest.approx(expected, rel=1e-6)
    assert r["api_date"][:2] == "20" and r["prices_date"] == ab.PRICES_DATE
    req = client.messages.calls[0]
    assert "temperature" not in req and "top_p" not in req and "seed" not in req
    assert req["thinking"] == {"type": "between_tools"} and req["max_tokens"] == 32
    assert req["output_config"] == {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert all(m["role"] != "system" for m in req["messages"]) and "system" in req
    longest = max(client.messages.calls, key=lambda k: len(k["messages"]))
    assert len(longest["messages"]) == 19
    # only the bare number is replayed as the assistant turn, never the JSON
    assert all(m["content"] == "42" for m in longest["messages"] if m["role"] == "assistant")
    df = load_results(tmp_path)
    assert len(df) == 90 and set(df.source) == {"claude-sonnet-5-5 [main, T1, number only]"}
    assert sp.label.endswith("__number-only")
    assert spec(partial=True, n_subjects=5).filename.endswith("__number-only__PARTIAL_s1-5.jsonl")


def test_reasoning_run_uses_the_same_schema_and_replays_only_the_number(tmp_path):
    client = fake_client(text='{"probability": 1}', usage=_usage(inp=600, out=250, think=240))
    be = ab.AnthropicBackend(
        "claude-opus-5-5", "reasoning-low", max_dollars=50, max_calls=10_000, client=client
    )
    sp = spec("claude-opus-5-5", "reasoning-low")
    st = run_spec(sp, be, trials(), tmp_path, progress=None)
    assert st.complete
    req = client.messages.calls[0]
    assert req["thinking"] == {"type": "adaptive"} and req["max_tokens"] == 8192
    assert req["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
    }
    r = read_jsonl(tmp_path / sp.filename)[0]
    assert r["raw_text"] == "1" and r["api_raw_text"] == '{"probability": 1}'
    assert r["thinking_tokens"] == 240 and r["output_tokens"] == 250 and r["effort"] == "low"
    longest = max(client.messages.calls, key=lambda k: len(k["messages"]))
    assert all(m["content"] == "1" for m in longest["messages"] if m["role"] == "assistant")
    src = load_results(tmp_path).source.iloc[0]
    assert src == "claude-opus-5-5 [main, T1, reasoning allowed (effort low)]"
    # a non-JSON reply still yields its final integer; no number or a truncated reply is an error
    be2 = ab.AnthropicBackend(
        "claude-opus-5-5",
        "reasoning",
        max_dollars=1,
        max_calls=10,
        client=fake_client(text="about 1%.\n\n1"),
    )
    msgs = [{"role": "user", "content": "x"}]
    assert be2.chat(msgs, temperature=1.0, seed=1, num_predict=8).text == "1"
    for kw in ({"text": "I cannot give a number for this one."}, {"stop_reason": "max_tokens"}):
        be3 = ab.AnthropicBackend(
            "claude-opus-5-5", "reasoning", max_dollars=1, max_calls=10, client=fake_client(**kw)
        )
        with pytest.raises(ab.UnusableReplyError):
            be3.chat(msgs, temperature=1.0, seed=1, num_predict=8)
    assert be3.truncations == 1


def test_key_never_appears_in_rows_logs_or_backend_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    client = fake_client(fail_on=3)  # one poisoned exception whose text contains the key
    be = ab.AnthropicBackend(
        "claude-sonnet-5-5", "number-only", max_dollars=5, max_calls=10_000, client=client
    )
    sp = spec()
    st = run_spec(sp, be, trials(), tmp_path, progress=None)
    assert st.n_errors == 1
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert any(p.suffix == ".log" for p in written)
    for p in written:
        assert FAKE_KEY not in p.read_text()
    assert "<ANTHROPIC_API_KEY>" in (tmp_path / sp.filename).with_suffix(".errors.log").read_text()
    assert FAKE_KEY not in repr(vars(be)) and FAKE_KEY not in repr(be.spend.summary())
    for row in read_jsonl(tmp_path / sp.filename):
        assert FAKE_KEY not in json.dumps(row)


def test_missing_key_stops_before_any_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ab.MissingAPIKeyError):
        ab.AnthropicBackend("claude-sonnet-5-5", "number-only", max_dollars=1, max_calls=1)


def test_spend_cap_stops_the_run_cleanly(tmp_path):
    # each call costs 600 * $2/M + 9 * $10/M = $0.00129; cap after ~4 calls
    client = fake_client(usage=_usage(inp=600, out=9))
    be = ab.AnthropicBackend(
        "claude-sonnet-5-5", "number-only", max_dollars=0.005, max_calls=10_000, client=client
    )
    sp = spec()
    st = run_spec(sp, be, trials(), tmp_path, progress=None, parallel=3)
    rows = read_jsonl(tmp_path / sp.filename)
    assert st.stopped and st.stopped.startswith("budget:")
    assert not st.complete and st.n_errors == 0
    assert 4 <= len(rows) <= 5 and len(client.messages.calls) == len(rows)
    assert be.spend.cost_usd >= 0.005 and be.spend.cost_usd < 0.005 + 0.00129 * 3
    be2 = ab.AnthropicBackend(
        "claude-sonnet-5-5", "number-only", max_dollars=5, max_calls=10_000, client=fake_client()
    )
    st2 = run_spec(sp, be2, trials(), tmp_path, progress=None)
    assert (
        st2.complete
        and st2.n_done_before == len(rows)
        and len(read_jsonl(tmp_path / sp.filename)) == 90
    )


def test_call_cap_stops_the_run_cleanly(tmp_path):
    be = ab.AnthropicBackend(
        "claude-haiku-5-5", "number-only", max_dollars=10, max_calls=7, client=fake_client()
    )
    sp = spec("claude-haiku-5-5")
    st = run_spec(sp, be, trials(), tmp_path, progress=None)
    assert st.stopped == "budget: 7 calls >= cap 7" and st.n_calls == 7
    with pytest.raises(StopRun):
        be.chat([{"role": "user", "content": "x"}], temperature=1.0, seed=1, num_predict=8)


def test_first_refusal_stops_the_run_and_records_the_category(tmp_path):
    details = SimpleNamespace(category="reasoning_extraction", explanation="asks for reasoning")
    client = fake_client(stop_reason="refusal", text="", stop_details=details)
    be = ab.AnthropicBackend(
        "claude-sonnet-5-5", "number-only", max_dollars=5, max_calls=10_000, client=client
    )
    sp = spec()
    st = run_spec(sp, be, trials(), tmp_path, progress=None)
    assert st.stopped and st.stopped.startswith("refusals: 1 >= cap 1") and st.n_errors == 0
    assert read_jsonl(tmp_path / sp.filename) == [] and len(client.messages.calls) == 1
    assert be.refusal_log[0]["category"] == "reasoning_extraction"
    assert be.refusal_log[0]["explanation"] == "asks for reasoning"
    assert "reasoning_extraction" in st.stopped and "asks for reasoning" in st.stopped
    # with a higher cap, earlier refusals are errors (trial breaks, nothing replayed)
    client = fake_client(stop_reason="refusal", text="", stop_details=details)
    be = ab.AnthropicBackend(
        "claude-sonnet-5-5",
        "number-only",
        max_dollars=5,
        max_calls=10_000,
        client=client,
        max_refusals=3,
    )
    st = run_spec(sp, be, trials(), tmp_path, progress=None)
    assert st.n_errors == 2 and st.stopped.startswith("refusals: 3")
    assert "reasoning_extraction" in (tmp_path / sp.filename).with_suffix(".errors.log").read_text()


def test_rate_limit_backoff_retries_then_succeeds(monkeypatch):
    class RateLimitedError(Exception):  # the SDK's RateLimitError shape
        status_code = 429

        def __init__(self):
            self.response = SimpleNamespace(headers={"retry-after": "0"})

    class Flaky(FakeMessages):
        def create(self, **kwargs):
            if len(self.calls) < 2:
                self.calls.append(kwargs)
                raise RateLimitedError()
            return super().create(**kwargs)

    client = SimpleNamespace(messages=Flaky())
    be = ab.AnthropicBackend(
        "claude-sonnet-5-5", "number-only", max_dollars=1, max_calls=10, client=client
    )
    monkeypatch.setattr(ab.time, "sleep", lambda s: None)
    r = be.chat([{"role": "user", "content": "x"}], temperature=1.0, seed=1, num_predict=8)
    assert r.text == "42" and len(client.messages.calls) == 3
    bad = SimpleNamespace(messages=FakeMessages(fail_on=1))
    be = ab.AnthropicBackend(
        "claude-sonnet-5-5", "number-only", max_dollars=1, max_calls=10, client=bad
    )
    with pytest.raises(RuntimeError):
        be.chat([{"role": "user", "content": "x"}], temperature=1.0, seed=1, num_predict=8)
    assert len(bad.messages.calls) == 1


def test_env_file_loader_sets_the_key_without_exposing_it(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"# comment\nexport ANTHROPIC_API_KEY='{FAKE_KEY}'\nOTHER=1\n\n")
    names = ab.load_env_file(env)
    assert names == ["ANTHROPIC_API_KEY", "OTHER"]
    assert os.environ["ANTHROPIC_API_KEY"] == FAKE_KEY and os.environ["OTHER"] == "1"
    assert FAKE_KEY not in repr(names) and FAKE_KEY not in capsys.readouterr().out
    monkeypatch.setenv("ANTHROPIC_API_KEY", "already-set")
    monkeypatch.delenv("OTHER")
    assert ab.load_env_file(env) == ["OTHER"] and os.environ["ANTHROPIC_API_KEY"] == "already-set"
    assert ab.load_env_file(tmp_path / "missing.env") == []
    be = ab.AnthropicBackend(
        "claude-sonnet-5-5", "number-only", max_dollars=1, max_calls=1, client=fake_client()
    )
    assert "already-set" not in repr(vars(be))

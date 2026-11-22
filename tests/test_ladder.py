"""Mocked-API tests for the hosted-model ladder runner (no network, no key)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import httpx
import pytest

from regime_shift.collection import ladder
from regime_shift.collection.backends import ANSWER_SCHEMA
from regime_shift.human_data import human_trials
from regime_shift.prompts import build_main_messages, build_stateless_messages, system_prompt
from regime_shift.task import BLUE, RED, Trial

# ---------------------------------------------------------------------------- fakes
# Minimal stand-ins for the SDK's response objects; the runner reads them with getattr only.


@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Details:
    thinking_tokens: int = 0


@dataclass
class _Usage:
    input_tokens: int = 700
    output_tokens: int = 9
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens_details: _Details = field(default_factory=_Details)


@dataclass
class _StopDetails:
    category: str
    explanation: str


@dataclass
class _Msg:
    model: str
    content: list
    usage: _Usage
    stop_reason: str = "end_turn"
    stop_details: object | None = None


class _Raw:
    def __init__(self, msg, headers=None, request_id="req_x"):
        self._msg = msg
        self.headers = headers or {
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-requests-remaining": "900",
            "request-id": request_id,
        }
        self.request_id = request_id

    def parse(self):
        return self._msg


def _ok(model_id: str, value: int = 42, **kw) -> _Msg:
    return _Msg(model_id, [_Block("text", json.dumps({"probability": value}))], _Usage(**kw))


class FakeClient:
    """Mimics ``client.messages.with_raw_response.create``; ``script`` decides each reply."""

    def __init__(self, script=None, latency: float = 0.001):
        self.script = script
        self.latency = latency
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.events: list[tuple[str, int, float]] = []
        self.messages = self
        self.with_raw_response = self

    async def create(self, **kwargs):
        n = len(self.calls)
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.events.append(("start", n, time.time()))
        try:
            await asyncio.sleep(self.latency)
            if self.script is not None:
                out = self.script(kwargs, n)
                if isinstance(out, Exception):
                    raise out
                if out is not None:
                    return out
            return _Raw(_ok(kwargs["model"]))
        finally:
            self.active -= 1
            self.events.append(("end", n, time.time()))


def _api_error(cls, status: int, headers=None, body=None):
    resp = httpx.Response(
        status, headers=headers or {}, request=httpx.Request("POST", "https://api/x")
    )
    return getattr(anthropic, cls)("boom", response=resp, body=body)


def _trials(n_subjects: int = 2, per_subject: int = 3) -> dict[int, list[Trial]]:
    out = {}
    for s in range(1, n_subjects + 1):
        out[s] = [
            Trial(
                s,
                t,
                0,
                1,
                t,
                0.05,
                3.0,
                tuple((BLUE if (t >> i) & 1 else RED) for i in range(10)),
                None,
            )
            for t in range(1, per_subject + 1)
        ]
    return out


def _runner(tmp_path, client, cap=50.0, stage=50.0, **kw) -> ladder.LadderRunner:
    ledger = ladder.Ledger(cap, 0.0, stage, tmp_path / f"ledger_{uuid.uuid4().hex}.json")
    return ladder.LadderRunner(
        client, ledger, out_dir=tmp_path / "out", log=None, progress_every_s=3600, **kw
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------- request payload


@pytest.mark.parametrize("alias", sorted(ladder.LADDER))
def test_request_payload_per_model(alias):
    lm = ladder.LADDER[alias]
    t = _trials(1, 1)[1][0]
    kw = ladder.build_request(lm, build_main_messages(t, 3, ["50", "60"]))
    assert kw["model"] == lm.model_id
    assert kw["thinking"] == dict(lm.thinking_off)
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
    assert kw["cache_control"] == {"type": "ephemeral"}
    assert kw["max_tokens"] == ladder.MAX_TOKENS
    assert kw["system"] == system_prompt("v1")
    for forbidden in ("temperature", "top_p", "top_k", "seed", "metadata"):
        assert forbidden not in kw
    roles = [m["role"] for m in kw["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in kw["messages"] if m["role"] == "assistant"] == ["50", "60"]
    # thinking really is off for every ladder model
    assert kw["thinking"]["type"] in ("disabled", "between_tools")


def test_estimate_covers_a_real_sized_call():
    lm = ladder.LADDER["claude-sonnet-4-6"]
    t = _trials(1, 1)[1][0]
    kw = ladder.build_request(lm, build_main_messages(t, 10, ["50"] * 9))
    est = ladder.estimate_cost_usd(lm, kw)
    actual = ladder.call_cost_usd(lm.price, {"input_tokens": 1455, "output_tokens": 12})
    assert est >= actual


# ---------------------------------------------------------------------------- ledger


def test_ledger_never_exceeds_cap_under_concurrent_workers(tmp_path):
    async def go():
        led = ladder.Ledger(
            project_cap=1.0, baseline=0.4, stage_budget=0.5, path=tmp_path / "l.json"
        )
        ok = 0

        async def worker():
            nonlocal ok
            for _ in range(20):
                try:
                    est = await led.reserve(0.01)
                except ladder.BudgetExceededError:
                    return
                await asyncio.sleep(0)
                await led.commit(est, 0.01)
                ok += 1

        await asyncio.gather(*(worker() for _ in range(50)))
        return led, ok

    led, ok = _run(go())
    assert led.committed <= 0.5 + 1e-9 and led.total <= 1.0 + 1e-9
    assert led.rejected > 0 and ok == led.calls
    # persisted after every commit; a second process never loads it but sums it as baseline
    on_disk = json.loads((tmp_path / "l.json").read_text())
    assert on_disk["committed"] == pytest.approx(led.committed) and on_disk["calls"] == led.calls
    with pytest.raises(FileExistsError):
        ladder.Ledger(1.0, 0.4, 0.5, tmp_path / "l.json")


def test_baseline_sums_legacy_runs_and_ledger_files_once(tmp_path):
    root = tmp_path
    res = root / "results"
    (res / "partial").mkdir(parents=True)
    (res / "ladder" / "ledgers").mkdir(parents=True)
    (res / "ladder" / "stage2").mkdir(parents=True)
    # legacy run: csv row and rows agree
    (res / "api_runs.csv").write_text(
        "thinking_param,cost_usd,file\nthinking disabled,1.0,a\nthinking x (ladder runner),9.0,b\n"
    )
    (res / "claude-haiku-5-5__x.jsonl").write_text(
        json.dumps({"cost_usd": 0.6}) + "\n" + json.dumps({"cost_usd": 0.6}) + "\n"
    )
    # ladder rows anywhere are excluded (covered by ledgers)
    (res / "claude-sonnet-5__main.jsonl").write_text(
        json.dumps({"cost_usd": 5.0, "runner": "ladder"}) + "\n"
    )
    (res / "ladder" / "stage2" / "claude-sonnet-5__y.jsonl").write_text(
        json.dumps({"cost_usd": 7.0, "runner": "ladder"}) + "\n"
    )
    (res / "ladder" / "ledgers" / "full_1.json").write_text(json.dumps({"committed": 2.5}))
    (res / "ladder" / "ledgers" / "canary_2.json").write_text(json.dumps({"committed": 0.25}))
    b = ladder.baseline_spend_usd(root)
    assert b["csv"] == 1.0 and b["jsonl"] == 1.2 and b["legacy"] == 1.2
    assert b["ladder_ledgers"] == 2.75 and b["n_ledgers"] == 2
    assert b["baseline"] == pytest.approx(3.95)


def test_ledger_project_cap_binds_before_stage_budget(tmp_path):
    led = ladder.Ledger(project_cap=10.0, baseline=9.99, stage_budget=5.0)
    with pytest.raises(ladder.BudgetExceededError, match="project cap"):
        _run(led.reserve(0.02))


def test_budget_breach_halts_run_before_the_request(tmp_path):
    client = FakeClient()
    r = _runner(tmp_path, client, cap=0.0005, stage=0.0005)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 2))
    s = _run(r.run())
    assert s["halt"]["reason"] == "budget"
    assert (
        len(client.calls) <= 1
    )  # the estimate exceeded the cap, nothing (or at most the warm-up) was sent


# ---------------------------------------------------------------------------- breakers


def test_global_breaker_stops_every_model_at_a_refusal(tmp_path):
    def script(kw, n):
        if n == 12:
            m = _ok(kw["model"])
            m.content = []
            m.stop_reason = "refusal"
            m.stop_details = _StopDetails("reasoning_extraction", "nope")
            return _Raw(m, request_id="req_refusal")
        return None

    client = FakeClient(script, latency=0.005)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5", "claude-sonnet-4-6", "claude-sonnet-5"], 2, _trials(2, 4))
    s = _run(r.run())
    assert s["halt"]["reason"] == "refusal"
    assert s["halt"]["request_id"] == "req_refusal"
    assert s["halt"]["category"] == "reasoning_extraction"
    halt_t = s["halt"]["time"]
    started_after = [e for e in client.events if e[0] == "start" and e[2] > halt_t + 1e-6]
    assert started_after == []
    total_rows = sum(m["rows_on_disk"] for m in s["models"].values())
    assert total_rows < 3 * 2 * 4 * 10
    # every row on disk is a usable integer; the refusal was never written
    for m in s["models"].values():
        for row in ladder.read_jsonl(Path(m["file"])):
            assert row["raw_text"].isdigit() and row["stop_reason"] == "end_turn"


def test_403_trips_the_global_breaker(tmp_path):
    def script(kw, n):
        if n == 3:
            return _api_error(
                "PermissionDeniedError",
                403,
                body={"error": {"type": "permission_error", "message": "model_access_suspended"}},
            )
        return None

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5", "claude-sonnet-5"], 1, _trials(1, 3))
    s = _run(r.run())
    assert s["halt"]["reason"].startswith("403")
    assert "model_access_suspended" in s["halt"]["body"]


def test_per_model_breaker_pauses_only_that_model(tmp_path):
    def script(kw, n):
        if kw["model"] == "claude-sonnet-5":
            return _Raw(_ok(kw["model"], output_tokens_details=_Details(thinking_tokens=7)))
        return None

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5", "claude-sonnet-5"], 1, _trials(1, 3))
    s = _run(r.run())
    assert s["halt"] is None
    assert s["paused"]["claude-sonnet-5"]["reason"] == "non-zero thinking tokens"
    assert s["models"]["claude-sonnet-5"]["rows_on_disk"] == 0
    assert s["models"]["claude-haiku-4-5"]["rows_on_disk"] == 30


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("truncate", "truncated reply"),
        ("garbage", "unusable answer"),
        ("other-model", "returned model id differs"),
    ],
)
def test_per_model_breaker_on_bad_replies(tmp_path, variant, reason):
    def script(kw, n):
        m = _ok(kw["model"])
        if variant == "truncate":
            m.stop_reason = "max_tokens"
        elif variant == "garbage":
            m.content = [_Block("text", "I think about 40 percent, maybe")]
        else:
            m.model = "claude-haiku-4-5-20240101"
        return _Raw(m)

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 2))
    s = _run(r.run())
    assert s["paused"]["claude-haiku-4-5"]["reason"] == reason
    assert s["models"]["claude-haiku-4-5"]["rows_on_disk"] == 0


def test_error_rate_breaker(tmp_path):
    def script(kw, n):
        if n in (2, 5):
            return _api_error(
                "BadRequestError",
                400,
                body={"error": {"type": "invalid_request_error", "message": "x"}},
            )
        return None

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 6))
    s = _run(r.run())
    assert s["paused"]["claude-haiku-4-5"]["reason"] == "error rate > 1%"
    assert s["models"]["claude-haiku-4-5"]["errors"] == 2


# ---------------------------------------------------------------------------- rate limits


def test_429_halves_concurrency_and_honours_retry_after(tmp_path):
    hit = {"n": 0}

    def script(kw, n):
        if n == 6 and hit["n"] == 0:
            hit["n"] += 1
            return _api_error("RateLimitError", 429, headers={"retry-after": "0.3"})
        return None

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 8))
    t0 = time.time()
    s = _run(r.run())
    m = s["models"]["claude-haiku-4-5"]
    assert (
        m["n429"] == 1 and m["errors"] == 0 and m["rows_on_disk"] == 80 and m["calls_started"] == 80
    )
    kinds = [h[2] for h in m["concurrency_history"]]
    assert any(k.startswith("429") for k in kinds)
    halved = [h[1] for h in m["concurrency_history"] if h[2].startswith("429")][0]
    assert halved == 2
    assert time.time() - t0 >= 0.3


def test_reservations_are_released_on_every_failure_path(tmp_path):
    def script(kw, n):
        if n % 4 == 1:
            return _api_error(
                "BadRequestError",
                400,
                body={"error": {"type": "invalid_request_error", "message": "x"}},
            )
        if n % 4 == 2:
            m = _ok(kw["model"])
            m.stop_reason = "max_tokens"
            return _Raw(m)
        return None

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5", "claude-sonnet-5"], 1, _trials(1, 3))
    s = _run(r.run())
    assert s["ledger"]["reserved_usd"] == 0
    assert s["ledger"]["committed_usd"] == pytest.approx(
        sum(m["cost_usd"] for m in s["models"].values()), abs=1e-9
    )


def test_unexpected_exception_after_the_call_is_contained(tmp_path):
    class Weird:
        model = "claude-haiku-4-5-20251001"
        usage = _Usage()
        stop_reason = "end_turn"
        stop_details = None

        @property
        def content(self):
            raise RuntimeError("pydantic went wrong")

    def script(kw, n):
        return _Raw(Weird()) if n == 2 else None

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 2))
    s = _run(r.run())
    assert s["failures"] == [] and s["ledger"]["reserved_usd"] == 0
    m = s["models"]["claude-haiku-4-5"]
    assert m["errors"] == 1 and "pydantic" in m["error_log"][0]["error"]
    # the failed period was not written and its trial stopped there (a conversation cannot
    # skip a turn); the other trial completed; what is on disk is a clean prefix per trial
    assert 10 <= m["rows_on_disk"] < 20
    r2 = _runner(tmp_path / "again", FakeClient())
    r2.out_dir = r.out_dir
    r.states["claude-haiku-4-5"].lock_handle.close()
    r2.prepare(["claude-haiku-4-5"], 1, _trials(1, 2))
    assert r2.states["claude-haiku-4-5"].inconsistent_trials == []


def test_out_of_range_and_non_json_answers_pause_the_model(tmp_path):
    for payload, reason in ((150, "answer outside 0-99"), (-5, "answer outside 0-99")):
        client = FakeClient(lambda kw, n, v=payload: _Raw(_ok(kw["model"], value=v)))
        r = _runner(tmp_path / str(payload), client)
        r.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
        s = _run(r.run())
        assert s["paused"]["claude-haiku-4-5"]["reason"] == reason
        assert s["models"]["claude-haiku-4-5"]["rows_on_disk"] == 0
    client = FakeClient(
        lambda kw, n: _Raw(_Msg(kw["model"], [_Block("text", "about 40")], _Usage()))
    )
    r = _runner(tmp_path / "nonjson", client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
    s = _run(r.run())
    assert s["paused"]["claude-haiku-4-5"]["reason"] == "non-JSON answer"
    assert s["models"]["claude-haiku-4-5"]["rows_on_disk"] == 0


def test_a_reply_of_100_is_clamped_recorded_and_replayed(tmp_path):
    def script(kw, n):
        return _Raw(_ok(kw["model"], value=100 if n == 3 else 42))

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 2))
    s = _run(r.run())
    m = s["models"]["claude-haiku-4-5"]
    assert s["paused"] == {} and m["rows_on_disk"] == 20 and m["clamped"] == 1
    rows = ladder.read_jsonl(Path(m["file"]))
    cl = [row for row in rows if row["parse_status"] == "clamped_100"]
    assert len(cl) == 1 and cl[0]["raw_text"] == "100" and cl[0]["value"] == 99
    # the verbatim reply is what later turns of that trial replay
    later = [
        kw
        for kw in client.calls
        if any(msg["content"] == "100" for msg in kw["messages"] if msg["role"] == "assistant")
    ]
    assert later, "the clamped reply was not replayed"
    assert ladder.replay_consistency(Path(m["file"]))["non_integer_raw_text"] == 0


def test_clamped_replies_above_one_percent_pause_the_model(tmp_path):
    client = FakeClient(lambda kw, n: _Raw(_ok(kw["model"], value=100 if n % 20 == 0 else 42)))
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 15))
    s = _run(r.run())
    assert s["paused"]["claude-haiku-4-5"]["reason"] == "clamped replies > 1%"
    assert s["models"]["claude-haiku-4-5"]["clamped"] >= 2


def test_thinking_block_in_content_pauses_even_without_usage_details(tmp_path):
    def script(kw, n):
        m = _Msg(
            kw["model"],
            [_Block("thinking", "hmm"), _Block("text", json.dumps({"probability": 3}))],
            _Usage(output_tokens_details=None),
        )
        return _Raw(m)

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
    s = _run(r.run())
    assert s["paused"]["claude-haiku-4-5"]["reason"] == "non-zero thinking tokens"
    assert s["models"]["claude-haiku-4-5"]["thinking_field_missing"] is True


def test_second_process_on_the_same_file_is_refused(tmp_path):
    r1 = _runner(tmp_path, FakeClient())
    r1.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
    r2 = _runner(tmp_path, FakeClient())
    with pytest.raises(RuntimeError, match="another process"):
        r2.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
    r1.states["claude-haiku-4-5"].lock_handle.close()
    r3 = _runner(tmp_path, FakeClient())
    r3.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))  # lock released: fine


def test_trial_with_a_hole_is_skipped_not_resumed(tmp_path):
    client = FakeClient()
    r = _runner(tmp_path, client)
    trials = _trials(1, 2)
    r.prepare(["claude-haiku-4-5"], 1, trials)
    s = _run(r.run())
    path = Path(s["models"]["claude-haiku-4-5"]["file"])
    rows = ladder.read_jsonl(path)
    spec = ladder.make_spec("claude-haiku-4-5", 1)
    hole = ladder.result_key(spec, 1, 1, 4)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["key"] != hole))
    client2 = FakeClient()
    r2 = _runner(tmp_path, client2)
    r2.prepare(["claude-haiku-4-5"], 1, trials)
    ms = r2.states["claude-haiku-4-5"]
    assert ms.inconsistent_trials == [1] and ms.pending == []
    s2 = _run(r2.run())
    assert client2.calls == [] and s2["models"]["claude-haiku-4-5"]["inconsistent_trials"] == [1]


def test_rate_limited_exhaustion_is_not_a_model_error(tmp_path):
    client = FakeClient(
        lambda kw, n: _api_error("RateLimitError", 429, headers={"retry-after": "0.01"})
    )
    r = _runner(tmp_path, client, max_attempts=3)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
    s = _run(r.run())
    m = s["models"]["claude-haiku-4-5"]
    assert m["errors"] == 0 and m["rate_limited_giveups"] >= 1 and m["rows_on_disk"] == 0
    assert (
        s["paused"] == {} and s["ledger"]["committed_usd"] == 0 and s["ledger"]["reserved_usd"] == 0
    )
    assert len(client.calls) <= 6  # at most two trial attempts (warm-up and worker) of 3 tries each
    assert m["concurrency_final"] == 1  # halved down to the floor


def test_persistent_5xx_uses_exactly_max_attempts_then_counts_one_error(tmp_path):
    client = FakeClient(lambda kw, n: _api_error("InternalServerError", 500))
    sleeps = []

    async def fake_sleep(x):
        sleeps.append(x)

    r = _runner(tmp_path, client, max_attempts=4, sleep=fake_sleep)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
    s = _run(r.run())
    m = s["models"]["claude-haiku-4-5"]
    assert m["errors"] >= 1 and m["retries"] >= 4 and len(sleeps) == m["retries"]
    assert s["ledger"]["reserved_usd"] == 0 and s["ledger"]["committed_usd"] == 0


def test_spend_limit_429_and_402_trip_the_global_breaker(tmp_path):
    body = {
        "error": {
            "type": "rate_limit_error",
            "details": {"error_code": "enforced_spend_limit_reached"},
        }
    }
    client = FakeClient(lambda kw, n: _api_error("RateLimitError", 429, body=body))
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
    s = _run(r.run())
    assert s["halt"]["reason"] == "provider spend limit" and len(client.calls) == 1


def test_writer_failure_trips_the_global_breaker(tmp_path, monkeypatch):
    client = FakeClient(latency=0.005)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 4))
    w = r.states["claude-haiku-4-5"].writer
    real_dumps = json.dumps
    state = {"n": 0}

    def bad_dumps(obj, **kw):
        state["n"] += 1
        if state["n"] == 3:
            raise OSError("disk full")
        return real_dumps(obj, **kw)

    monkeypatch.setattr(ladder.json, "dumps", bad_dumps)
    s = _run(r.run())
    assert s["halt"]["reason"] == "writer failure" and w.failure is not None
    assert len(client.calls) < 40


def test_headroom_requires_headers():
    assert ladder.headroom({}) is False
    assert (
        ladder.headroom(
            {
                "anthropic-ratelimit-requests-limit": "100",
                "anthropic-ratelimit-requests-remaining": "50",
            }
        )
        is True
    )
    assert (
        ladder.headroom(
            {
                "anthropic-ratelimit-requests-limit": "100",
                "anthropic-ratelimit-requests-remaining": "5",
            }
        )
        is False
    )


def test_concurrency_ramps_only_with_headroom(tmp_path):
    def script(kw, n):
        return _Raw(
            _ok(kw["model"]),
            headers={
                "anthropic-ratelimit-requests-limit": "1000",
                "anthropic-ratelimit-requests-remaining": "50",
            },
        )

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 10))
    s = _run(r.run())
    assert s["models"]["claude-haiku-4-5"]["concurrency_final"] == 4  # never ramped: 5% headroom
    client2 = FakeClient()
    r2 = _runner(tmp_path / "b", client2)
    r2.prepare(["claude-haiku-4-5"], 1, _trials(1, 10))
    s2 = _run(r2.run())
    assert s2["models"]["claude-haiku-4-5"]["concurrency_final"] > 4
    assert client2.max_active <= s2["models"]["claude-haiku-4-5"]["concurrency_final"]


def test_warmup_call_runs_alone_before_fan_out(tmp_path):
    client = FakeClient(latency=0.02)
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 6))
    _run(r.run())
    ev = client.events
    first_end = next(e for e in ev if e[0] == "end" and e[1] == 0)[2]
    second_start = next(e for e in ev if e[0] == "start" and e[1] == 1)[2]
    assert second_start >= first_end
    assert client.max_active > 1  # and after the warm-up it did fan out


# ---------------------------------------------------------------------------- resume + writer


def test_resume_leaves_no_duplicates_or_gaps_and_replays_stored_integers(tmp_path):
    def script(kw, n):
        return _Raw(_ok(kw["model"], value=(n * 7) % 100))

    client = FakeClient(script)
    r = _runner(tmp_path, client, max_calls_per_model=23)
    trials = _trials(1, 4)
    r.prepare(["claude-haiku-4-5"], 1, trials)
    s = _run(r.run())
    path = Path(s["models"]["claude-haiku-4-5"]["file"])
    first = ladder.read_jsonl(path)
    assert len(first) == 23 == len(client.calls)  # the call cap is exact
    client2 = FakeClient(script)
    r2 = _runner(tmp_path, client2)
    r2.prepare(["claude-haiku-4-5"], 1, trials)
    s2 = _run(r2.run())
    spec = ladder.make_spec("claude-haiku-4-5", 1)
    comp = ladder.completeness(path, spec, trials[1])
    assert comp["complete"] and comp["rows"] == 40 and comp["duplicates"] == 0
    assert ladder.replay_consistency(path)["non_integer_raw_text"] == 0
    stored = {(row["trial"], row["period"]): row["raw_text"] for row in first}
    # every assistant turn the second run sent equals the stored integer of that period
    checked = 0
    for kw in client2.calls:
        assistant = [m["content"] for m in kw["messages"] if m["role"] == "assistant"]
        user_turns = [m["content"] for m in kw["messages"] if m["role"] == "user"]
        k = len(assistant) + 1
        # identify the trial from ALL its user turns (periods 1..k), not just the first ball
        cands = [
            t
            for t in trials[1]
            if [m["content"] for m in build_main_messages(t, k, assistant) if m["role"] == "user"]
            == user_turns
        ]
        if k == 1:
            continue
        assert len(cands) == 1, (k, [t.trial for t in cands])
        trial_no = cands[0].trial
        for kk, a in enumerate(assistant, start=1):
            if (trial_no, kk) in stored:
                assert a == stored[(trial_no, kk)]
                checked += 1
    assert checked > 0
    assert len(ladder.read_jsonl(path)) == 40
    assert s2["halt"] is None and s2["paused"] == {}


def test_writer_never_interleaves_rows(tmp_path):
    async def go():
        w = ladder.QueueWriter(tmp_path / "w.jsonl")
        w.start()
        rows = [{"key": f"k{i}", "blob": "x" * (500 + i % 7)} for i in range(400)]
        await asyncio.gather(*(w.put(r) for r in rows))
        await w.close()
        return w

    w = _run(go())
    lines = (tmp_path / "w.jsonl").read_text().splitlines()
    assert len(lines) == 400 == w.written
    parsed = [json.loads(line) for line in lines]
    assert {p["key"] for p in parsed} == {f"k{i}" for i in range(400)}


def test_torn_last_line_is_terminated_and_redone(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    spec = ladder.make_spec("claude-haiku-4-5", 1)
    (out / spec.filename).write_text('{"key": "junk", "half": tr')
    client = FakeClient()
    r = _runner(tmp_path, client)
    r.prepare(["claude-haiku-4-5"], 1, _trials(1, 1))
    _run(r.run())
    rows = ladder.read_jsonl(out / spec.filename)
    assert len(rows) == 10 and all(row["raw_text"] == "42" for row in rows)


# ---------------------------------------------------------------------------- rows


def test_row_layout_matches_the_single_model_runner(tmp_path):
    client = FakeClient()
    r = _runner(tmp_path, client)
    r.prepare(["claude-sonnet-5"], 1, _trials(1, 1))
    s = _run(r.run())
    row = ladder.read_jsonl(Path(s["models"]["claude-sonnet-5"]["file"]))[0]
    for k in (
        "key",
        "source",
        "model_tag",
        "model_digest",
        "subject",
        "trial",
        "period",
        "q",
        "d",
        "signal",
        "raw_text",
        "value",
        "estimate",
        "parse_status",
        "api_model",
        "thinking_param",
        "api_raw_text",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
        "stop_reason",
        "temperature_sent",
        "seed_sent",
        "sampling",
        "anthropic_version",
        "request_id",
    ):
        assert k in row, k
    assert row["temperature_sent"] is False and row["sampling"] == "provider default"
    assert row["model_tag"] == "claude-sonnet-5" and row["model_digest"] == "claude-sonnet-5"
    assert row["source"] == "claude-sonnet-5 [main, T1, number only, PARTIAL s1-1]"
    assert row["parse_status"] == "ok" and row["value"] == 42


# ---------------------------------------------------------------------------- prompt + mapping


def test_prompt_fingerprint_and_subject_mapping(human_all):
    tbs = human_trials(human_all[human_all.session == 0].reset_index(drop=True))
    fp = ladder.check_prompt_and_mapping(tbs)
    assert fp["n_user_turns"] == "21600"
    # simulated subject i answers human subject i's sequences, in the archive's trial order
    sub = human_all[human_all.session == 0]
    for s in (1, 17, 30):
        archive = sub[sub.subject == s]
        first = tbs[s][0]
        got = tuple(
            int(x) for x in archive[archive.trial == first.trial].sort_values("period").signal
        )
        assert got == first.signals and first.subject == s
    # a swapped subject is caught
    swapped = dict(tbs)
    swapped[1], swapped[2] = tbs[2], tbs[1]
    with pytest.raises(AssertionError, match="another subject"):
        ladder.check_prompt_and_mapping(swapped)


def test_catalogue_and_projection():
    tab = ladder.catalogue_table()
    assert "claude-haiku-4-5-20251001" in tab and "claude-sonnet-5-5" in tab
    assert all(alias in tab for alias in ladder.LADDER)
    p = ladder.projection(ladder.LADDER["claude-haiku-4-5"])
    assert p["cacheable_periods"] == 0 and p["cost_uncached_usd"] == p["cost_cached_usd"]
    p55 = ladder.projection(ladder.LADDER["claude-sonnet-5-5"])
    assert p55["cacheable_periods"] == 10 and p55["cost_cached_usd"] < p55["cost_uncached_usd"]


# ---------------------------------------------------------------------------- stateless + reasoning


def test_stateless_request_is_single_turn_and_resume_allows_holes(tmp_path):
    client = FakeClient()
    r = _runner(tmp_path, client)
    trials = _trials(1, 2)
    r.prepare(["claude-sonnet-5-5"], 1, trials, mode="stateless")
    s = _run(r.run())
    m = s["models"]["claude-sonnet-5-5"]
    assert m["rows_on_disk"] == 20 and s["paused"] == {}
    for kw in client.calls:
        roles = [msg["role"] for msg in kw["messages"]]
        assert roles == ["user"]  # no history, no assistant turns
        assert "All balls drawn so far" in kw["messages"][0]["content"]
    assert "stateless" in m["file"] and "number-only" in m["file"]
    # a hole in a stateless file is simply redone, not treated as inconsistent
    path = Path(m["file"])
    rows = ladder.read_jsonl(path)
    spec = ladder.make_spec("claude-sonnet-5-5", 1, "stateless")
    hole = ladder.result_key(spec, 1, 1, 4)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["key"] != hole))
    r.states["claude-sonnet-5-5"].lock_handle.close()
    client2 = FakeClient()
    r2 = _runner(tmp_path, client2)
    r2.prepare(["claude-sonnet-5-5"], 1, trials, mode="stateless")
    assert (
        r2.states["claude-sonnet-5-5"].inconsistent_trials == []
        and len(r2.states["claude-sonnet-5-5"].pending) == 1
    )
    _run(r2.run())
    assert len(client2.calls) == 1 and ladder.completeness(path, spec, trials[1])["complete"]


def test_reasoning_request_and_breakers(tmp_path):
    lm = ladder.LADDER["claude-sonnet-5-5"]
    kw = ladder.build_request(lm, build_main_messages(_trials(1, 1)[1][0], 2, ["50"]), "reasoning")
    assert (
        kw["thinking"] == {"type": "adaptive"} and kw["max_tokens"] == ladder.REASONING_MAX_TOKENS
    )
    assert kw["output_config"]["format"]["schema"] == ANSWER_SCHEMA
    assert "temperature" not in kw
    # the estimate reserves for the full reasoning budget
    assert (
        ladder.estimate_cost_usd(lm, kw) > 1.5 * ladder.REASONING_MAX_TOKENS * lm.price.output / 1e6
    )

    # thinking tokens are allowed in the reasoning condition; truncation still pauses
    def script(kw, n):
        m = _ok(kw["model"], output_tokens=700, output_tokens_details=_Details(thinking_tokens=650))
        m.content = [_Block("thinking", "..."), *m.content]
        if n == 7:
            m.stop_reason = "max_tokens"
        return _Raw(m)

    client = FakeClient(script)
    r = _runner(tmp_path, client)
    r.prepare(["claude-sonnet-5-5"], 1, _trials(1, 2), condition="reasoning")
    s = _run(r.run())
    m = s["models"]["claude-sonnet-5-5"]
    assert s["paused"]["claude-sonnet-5-5"]["reason"] == "truncated reply"
    assert (
        m["usage"]["thinking_tokens"] == 650 * (m["calls"] + m["truncations"]) and m["calls"] >= 6
    )
    row = ladder.read_jsonl(Path(m["file"]))[0]
    assert row["answer_condition"] == "reasoning" and row["thinking_tokens"] == 650
    assert (
        row["thinking_param"].startswith("adaptive thinking")
        and row["max_tokens"] == ladder.REASONING_MAX_TOKENS
    )
    assert "reasoning" in m["file"] and "PARTIAL_s1-1" in m["file"]


def test_number_only_still_pauses_on_thinking(tmp_path):
    client = FakeClient(
        lambda kw, n: _Raw(_ok(kw["model"], output_tokens_details=_Details(thinking_tokens=5)))
    )
    r = _runner(tmp_path, client)
    r.prepare(["claude-sonnet-5-5"], 1, _trials(1, 1))
    s = _run(r.run())
    assert s["paused"]["claude-sonnet-5-5"]["reason"] == "non-zero thinking tokens"


def test_make_spec_rejects_unknown_mode_or_condition():
    with pytest.raises(ValueError):
        ladder.make_spec("claude-sonnet-5-5", 30, "betting")
    with pytest.raises(ValueError):
        ladder.make_spec("claude-sonnet-5-5", 30, "main", "reasoning-low")


def test_subject_sequential_budget_stop_leaves_whole_subjects(tmp_path):
    # budget allows roughly 1.6 subjects' worth of calls: with subject-sequential ordering the
    # first subject completes and only the second is cut
    client = FakeClient()
    led = ladder.Ledger(50.0, 0.0, 0.0, None)
    r = ladder.LadderRunner(
        client,
        led,
        out_dir=tmp_path / "out",
        log=None,
        progress_every_s=3600,
        subject_sequential=True,
    )
    trials = _trials(3, 4)  # 3 subjects x 4 trials x 10 periods = 120 calls
    r.prepare(["claude-haiku-4-5"], 3, trials, mode="stateless")
    lm = ladder.LADDER["claude-haiku-4-5"]
    one = ladder.estimate_cost_usd(
        lm, ladder.build_request(lm, build_stateless_messages(trials[1][0], 5))
    )
    actual = ladder.call_cost_usd(lm.price, {"input_tokens": 700, "output_tokens": 9})
    led.stage_budget = actual * 64 + one  # ~1.6 subjects of metered cost plus one reservation
    s = _run(r.run())
    rows = ladder.read_jsonl(Path(s["models"]["claude-haiku-4-5"]["file"]))
    per = {}
    for row in rows:
        per[row["subject"]] = per.get(row["subject"], 0) + 1
    assert per.get(1) == 40 and 0 < per.get(2, 0) < 40 and 3 not in per
    assert s["stopped_by_budget"]

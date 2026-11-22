import json

import numpy as np
import pytest

from regime_shift.collection.backends import (
    ANSWER_SCHEMA,
    BayesianMock,
    NeglectMock,
    OllamaBackend,
    number_from_json,
)
from regime_shift.collection.runner import call_seed, read_jsonl, result_key, run_spec
from regime_shift.config import ExperimentConfig, RunSpec
from regime_shift.results import complete_trials_only, load_results
from regime_shift.task import generate_subject_trials


def trials(n_subjects=2, n_blocks=1):
    return {s: generate_subject_trials(s, 11, n_blocks=n_blocks) for s in range(1, n_subjects + 1)}


def spec(mode="main", temperature=0.7, n_subjects=2):
    return RunSpec(
        "mock-bayesian", mode, temperature, 5, n_subjects, "behavioral", 8, synthetic=True
    )


def test_run_resume_and_skip(tmp_path):
    sp = spec()
    st = run_spec(sp, BayesianMock(), trials(), tmp_path, progress=None)
    assert st.n_calls == 2 * 9 * 10 and st.complete and st.n_errors == 0
    rows = read_jsonl(tmp_path / sp.filename)
    assert len(rows) == 180 and all(r["parse_status"] == "ok" for r in rows)
    assert rows[0]["synthetic"] is True and "synthetic" in sp.filename
    # rerun: nothing to do
    st2 = run_spec(sp, BayesianMock(), trials(), tmp_path, progress=None)
    assert st2.n_calls == 0 and st2.n_done_before == 180 and st2.complete


def test_resume_mid_trial_replays_previous_answers(tmp_path):
    sp = spec(n_subjects=1)
    run_spec(sp, BayesianMock(), trials(1), tmp_path, progress=None)
    path = tmp_path / sp.filename
    rows = read_jsonl(path)
    # drop the last 4 periods of trial 3 and a torn final line
    keep = [r for r in rows if not (r["trial"] == 3 and r["period"] > 6)]
    with path.open("w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
        fh.write('{"key": "torn')
    st = run_spec(sp, BayesianMock(), trials(1), tmp_path, progress=None)
    assert st.n_calls == 4 and st.complete
    after = {r["key"]: r for r in read_jsonl(path)}
    assert len(after) == 90
    for r in rows:
        assert after[r["key"]]["raw_text"] == r["raw_text"]  # same answers (deterministic)


def test_stateless_and_greedy_modes(tmp_path):
    for sp in (spec("stateless"), spec("main", 0.0)):
        st = run_spec(sp, NeglectMock(noise_sd=0.05), trials(1, 1), tmp_path, progress=None)
        assert st.complete
    df = load_results(tmp_path)
    assert set(df["mode"]) == {"stateless", "main"}
    assert set(df.temperature) == {0.7, 0.0}
    assert df.model_digest.eq("synthetic").all()


def test_seeds_are_deterministic_and_distinct():
    sp = spec()
    assert call_seed(sp, 1, 2, 3) == call_seed(sp, 1, 2, 3)
    seeds = {
        call_seed(sp, s, t, p) for s in range(1, 3) for t in range(1, 10) for p in range(1, 11)
    }
    assert len(seeds) == 180
    assert result_key(sp, 1, 2, 3).endswith("|s1|t2|p3")


def test_parallel_matches_serial(tmp_path):
    sp = spec()
    run_spec(sp, NeglectMock(noise_sd=0.05), trials(), tmp_path / "a", progress=None)
    run_spec(sp, NeglectMock(noise_sd=0.05), trials(), tmp_path / "b", parallel=4, progress=None)
    a = load_results(tmp_path / "a").set_index(["subject", "trial", "period"]).raw_text
    b = load_results(tmp_path / "b").set_index(["subject", "trial", "period"]).raw_text
    assert a.sort_index().equals(b.sort_index())


def test_junk_is_counted_not_dropped(tmp_path):
    sp = spec()
    run_spec(sp, NeglectMock(noise_sd=0.05, junk_rate=0.3), trials(), tmp_path, progress=None)
    df = load_results(tmp_path)
    assert len(df) == 180
    assert (df.parse_status != "ok").mean() > 0.1
    assert (
        df.estimate.isna().sum()
        == (
            ~df.parse_status.isin(["ok", "ok_percent", "ok_wrapped", "ok_decimal", "clamped_100"])
        ).sum()
    )


def test_complete_trials_only(tmp_path):
    sp = spec(n_subjects=1)
    run_spec(sp, BayesianMock(), trials(1), tmp_path, progress=None)
    df = load_results(tmp_path)
    df = df[~((df.trial == 2) & (df.period == 10))]
    kept, incomplete = complete_trials_only(df)
    assert len(incomplete) == 1 and int(incomplete.trial.iloc[0]) == 2
    assert len(kept) == 80 and (kept.groupby("trial").period.count() == 10).all()


def test_config_runs_order_and_pilot():
    cfg = ExperimentConfig(models=("a", "b"), pilot=True)
    runs = cfg.runs()
    assert [r.model for r in runs] == ["a", "a", "a"]
    assert [(r.mode, r.temperature) for r in runs] == [
        ("main", 0.7),
        ("main", 0.0),
        ("stateless", 0.7),
    ]
    assert all(r.n_subjects == 5 for r in runs)
    full = ExperimentConfig(models=("a", "b")).runs()
    assert [(r.model, r.mode, r.temperature) for r in full] == [
        ("a", "main", 0.7),
        ("b", "main", 0.7),
        ("a", "main", 0.0),
        ("b", "main", 0.0),
        ("a", "stateless", 0.7),
        ("b", "stateless", 0.7),
    ]
    assert np.isfinite(full[0].seed)


def test_loader_gives_one_file_one_source_label(tmp_path):
    """Timing rows written under the full-run label inside a PARTIAL file get the file's label."""
    sp = spec(n_subjects=1)
    run_spec(sp, BayesianMock(), trials(1), tmp_path, progress=None)
    path = tmp_path / sp.filename
    rows = read_jsonl(path)
    assert len(rows) == 90
    for r in rows[:20]:  # the first 20 rows pretend to be the timing run's label
        r["source"] = "mock-bayesian [main, T0.7]"
    for r in rows[20:]:
        r["source"] = "mock-bayesian [main, T0.7, PARTIAL s1-10]"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    df = load_results(tmp_path)
    assert len(df) == 90
    assert set(df.source) == {"mock-bayesian [main, T0.7, PARTIAL s1-10]"}


def test_ollama_structured_output_extracts_the_number():
    """The Ollama bridge check sends the JSON schema and stores the bare number."""
    sent = {}

    class R:
        def __init__(self, content):
            self._c = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": self._c}, "total_duration": 1e9, "eval_count": 7}

    class Requests:
        @staticmethod
        def post(url, json, timeout):
            sent.update(json)
            return R('{"probability": 37}')

    be = OllamaBackend("qwen2.5:14b", format=ANSWER_SCHEMA)
    be._requests = Requests()
    r = be.chat([{"role": "user", "content": "x"}], temperature=0.7, seed=1, num_predict=16)
    assert sent["format"] == ANSWER_SCHEMA and sent["options"]["num_predict"] == 16
    assert r.text == "37" and r.extra["api_raw_text"] == '{"probability": 37}'
    assert number_from_json('{"probability": 4.0}') == "4"
    with pytest.raises((ValueError, KeyError)):
        number_from_json('{"answer": 4}')
    # without a schema no "format" field is sent
    be2 = OllamaBackend("qwen2.5:14b")
    be2._requests = Requests()
    sent.clear()
    be2.chat([{"role": "user", "content": "x"}], temperature=0.7, seed=1, num_predict=8)
    assert "format" not in sent


def test_ollama_think_option_is_sent_and_thinking_text_measured():
    sent = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {"content": "17", "thinking": ""},
                "total_duration": 1e9,
                "eval_count": 2,
            }

    class Requests:
        @staticmethod
        def post(url, json, timeout):
            sent.update(json)
            return R()

    be = OllamaBackend("granite4.2:8b", think=False)
    be._requests = Requests()
    r = be.chat([{"role": "user", "content": "x"}], temperature=0.7, seed=1, num_predict=8)
    assert sent["think"] is False and r.text == "17"
    assert r.extra == {"think_option": False, "thinking_text_len": 0}
    be2 = OllamaBackend("qwen2.5:14b")
    be2._requests = Requests()
    sent.clear()
    be2.chat([{"role": "user", "content": "x"}], temperature=0.7, seed=1, num_predict=8)
    assert "think" not in sent

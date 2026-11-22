"""Run a :class:`RunSpec` against a backend with incremental JSONL output and resume.

One JSONL row per model call, written as soon as the call returns. A rerun
reads the file, skips completed (subject, trial, period) keys and, in MAIN
mode, replays the stored replies of a partially completed trial as assistant
turns before continuing it. Calls that raise are logged and left for the next
rerun; they are never written as fake rows.
"""

from __future__ import annotations

import json
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..config import RunSpec
from ..parsing import parse_estimate
from ..prompts import build_main_messages, build_stateless_messages
from ..task import N_PERIODS, Trial
from .backends import Backend, StopRun


def result_key(spec: RunSpec, subject: int, trial: int, period: int) -> str:
    return f"{spec.label}|s{subject}|t{trial}|p{period}"


def call_seed(spec: RunSpec, subject: int, trial: int, period: int) -> int:
    """Explicit per-call seed, deterministic in (run, subject, trial, period)."""
    return zlib.crc32(result_key(spec, subject, trial, period).encode()) ^ (spec.seed & 0xFFFFFFFF)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a disconnect; that call is simply redone
    return rows


def completed_rows(path: Path) -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(path) if "key" in r}


@dataclass
class RunStats:
    spec_label: str
    n_expected: int
    n_done_before: int
    n_calls: int
    n_errors: int
    seconds: float
    stopped: str | None = None  # set when a backend raised StopRun (spend / call cap)

    @property
    def calls_per_second(self) -> float:
        return self.n_calls / self.seconds if self.seconds > 0 else float("nan")

    @property
    def complete(self) -> bool:
        return self.n_done_before + self.n_calls >= self.n_expected and self.n_errors == 0


class _Writer:
    """Append-only JSONL writer shared by the worker threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        # A disconnect can leave a torn last line without a newline; terminate it so the
        # next row starts cleanly (the torn row itself is ignored by read_jsonl and redone).
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as fh:
                fh.seek(-1, 2)
                last = fh.read(1)
            if last != b"\n":
                with path.open("a") as fh:
                    fh.write("\n")

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self.lock, self.path.open("a") as fh:
            fh.write(line + "\n")
            fh.flush()


def _row(
    spec: RunSpec,
    backend: Backend,
    trial: Trial,
    period: int,
    text: str,
    duration: float | None,
    extra: dict | None = None,
) -> dict:
    """The JSONL row for one call: run and trial identifiers, the raw reply and its parse."""
    row = {
        "key": result_key(spec, trial.subject, trial.trial, period),
        "run_label": spec.label,
        "source": spec.source,
        "model_tag": spec.model,
        "model_digest": backend.digest(),
        "mode": spec.mode,
        "temperature": spec.temperature,
        "seed": call_seed(spec, trial.subject, trial.trial, period),
        "num_predict": spec.num_predict,
        "prompt_version": spec.prompt_variant,
        "trial_set": spec.trial_set,
        "synthetic": spec.synthetic,
        "subject": trial.subject,
        "session": trial.session,
        "block": trial.block,
        "trial": trial.trial,
        "trial_in_block": trial.trial_in_block,
        "condition": trial.condition,
        "q": trial.q,
        "d": trial.d,
        "period": period,
        "signal": trial.signals[period - 1],
        "change_period": trial.change_period,
        "raw_text": text,
        "duration_s": duration,
        "timestamp": time.time(),
    }
    if extra:
        row.update({k: v for k, v in extra.items() if k not in row})
    pr = parse_estimate(text)
    row.update(parse_status=pr.status, value=pr.value, estimate=pr.estimate if pr.ok else None)
    return row


class _Stop:
    """Shared stop flag: set by the first StopRun; every other trial then ends quietly."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reason: str | None = None

    def set(self, reason: str) -> None:
        self.reason = self.reason or reason
        self.event.set()


def _run_trial(
    spec: RunSpec,
    backend: Backend,
    trial: Trial,
    done: dict[str, dict],
    writer: _Writer,
    stop: _Stop | None = None,
) -> tuple[int, int]:
    """Run one trial; returns (calls made, errors)."""
    calls = errors = 0
    previous: list[str] = []
    for period in range(1, N_PERIODS + 1):
        if stop is not None and stop.event.is_set():
            break
        key = result_key(spec, trial.subject, trial.trial, period)
        if key in done:
            previous.append(done[key].get("raw_text") or "")
            continue
        if spec.mode == "stateless":
            messages = build_stateless_messages(trial, period, variant=spec.prompt_variant)
        else:
            messages = build_main_messages(trial, period, previous, variant=spec.prompt_variant)
        try:
            resp = backend.chat(
                messages,
                temperature=spec.temperature,
                seed=call_seed(spec, trial.subject, trial.trial, period),
                num_predict=spec.num_predict,
            )
        except StopRun as exc:
            if stop is not None:
                stop.set(str(exc))
            break  # nothing written for this period; a rerun resumes here
        except Exception as exc:
            errors += 1
            writer_log = writer.path.with_suffix(".errors.log")
            with writer.lock, writer_log.open("a") as fh:
                fh.write(f"{time.time():.0f}\t{key}\t{exc!r}\n")
            if spec.mode != "stateless":
                break  # cannot continue a conversation with a missing turn; rerun resumes here
            continue
        row = _row(spec, backend, trial, period, resp.text, resp.duration_s, resp.extra)
        writer.write(row)
        done[key] = row
        previous.append(resp.text)
        calls += 1
    return calls, errors


def run_spec(
    spec: RunSpec,
    backend: Backend,
    trials_by_subject: dict[int, Sequence[Trial]],
    out_dir: str | Path,
    parallel: int = 1,
    progress: Callable[[str], None] | None = print,
    progress_every: int | None = None,
) -> RunStats:
    """Run every trial of the first ``spec.n_subjects`` subjects, resuming from ``out_dir``.

    ``progress_every`` is in trials; by default about ten progress lines per run.
    """
    out_path = Path(out_dir) / spec.filename
    done = completed_rows(out_path)
    subjects = sorted(trials_by_subject)[: spec.n_subjects]
    trials: list[Trial] = [t for s in subjects for t in trials_by_subject[s]]
    n_expected = len(trials) * N_PERIODS
    n_before = sum(1 for k in done if k.startswith(spec.label + "|"))
    writer = _Writer(out_path)
    t0 = time.time()
    calls = errors = 0
    pending = [
        t
        for t in trials
        if not all(result_key(spec, t.subject, t.trial, p) in done for p in range(1, N_PERIODS + 1))
    ]
    if progress:
        progress(
            f"[{spec.label}] {n_before}/{n_expected} calls already done; "
            f"{len(pending)} trials to run with parallel={parallel}"
        )
    finished = 0
    every = progress_every or max(1, len(pending) // 10)
    stop = _Stop()
    spend = getattr(backend, "format_spend", None)

    def report() -> None:
        if progress and finished % every == 0:
            el = time.time() - t0
            rate = calls / el if el > 0 else float("nan")
            extra = f"; {spend()}" if callable(spend) else ""
            progress(f"[{spec.label}] {finished}/{len(pending)} trials, {rate:.2f} calls/s{extra}")

    if parallel <= 1:
        for t in pending:
            c, e = _run_trial(spec, backend, t, done, writer, stop)
            calls += c
            errors += e
            finished += 1
            report()
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(_run_trial, spec, backend, t, done, writer, stop) for t in pending]
            for f in as_completed(futs):
                c, e = f.result()
                calls += c
                errors += e
                finished += 1
                report()
    stats = RunStats(spec.label, n_expected, n_before, calls, errors, time.time() - t0, stop.reason)
    if stop.reason and progress:
        progress(f"[{spec.label}] STOPPED: {stop.reason}")
    if progress:
        progress(
            f"[{spec.label}] done: {calls} calls in {stats.seconds:.0f}s "
            f"({stats.calls_per_second:.2f}/s), {errors} errors, complete={stats.complete}"
        )
    return stats

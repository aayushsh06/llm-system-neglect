"""Driver for regime_shift.collection.ladder: run a ladder of hosted Claude models with budget caps.

    python scripts/collection/run_ladder.py catalogue  # model table, no API call
    python scripts/collection/run_ladder.py check  # prompt hash + mapping, no API call
    python scripts/collection/run_ladder.py preflight --budget 1  # one stateless call per model
    python scripts/collection/run_ladder.py regression --budget 5  # subject 1 vs the existing files
    python scripts/collection/run_ladder.py canary --budget 5  # 20 calls per model, concurrency 2
    python scripts/collection/run_ladder.py one-subject --budget 5  # subject 1 per model
    python scripts/collection/run_ladder.py full --budget 150  # 30 subjects per model
    python scripts/collection/run_ladder.py stateless-ladder --budget 20  # 15 subjects, stateless
    python scripts/collection/run_ladder.py sonnet-probe --budget 5  # reasoning cost probe
    python scripts/collection/run_ladder.py sonnet-reasoning --budget 30  # 5 subjects, reasoning
    python scripts/collection/run_ladder.py sonnet-stateless --budget 30

Every API stage reads the spend baseline from results/api_runs.csv and the result files, enforces
the project cap (default $185) and the stage budget before each request, and writes a JSON report
to results/ladder/reports/. The smoke stages (regression, canary, one-subject, sonnet-probe) write
under results/ladder/ (never into results/); the others write to results/ or results/partial/.
The API key is read by the SDK from ANTHROPIC_API_KEY (loaded from .env) and is never written
anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from regime_shift import pipeline  # noqa: E402
from regime_shift.collection import ladder  # noqa: E402
from regime_shift.collection.anthropic_backend import load_env_file  # noqa: E402

DEFAULT_LADDER = ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-sonnet-5")
REPORTS = ROOT / "results" / "ladder" / "reports"
LEDGERS = ROOT / "results" / "ladder" / "ledgers"
PREFLIGHT_MODELS = (
    "claude-haiku-4-5",
    "claude-haiku-5-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
)
STATELESS_LADDER = (
    "claude-haiku-5-5",
    "claude-haiku-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)
STAGES = (
    "catalogue",
    "check",
    "preflight",
    "regression",
    "canary",
    "one-subject",
    "full",
    "sonnet-stateless",
    "sonnet-probe",
    "sonnet-reasoning",
    "stateless-ladder",
)


def log(msg: str) -> None:
    print(time.strftime("%F %T"), msg, flush=True)


def make_client(global_limit: int):
    """Async SDK client with a connection pool sized for the global concurrency limit."""
    import anthropic
    import httpx

    http = anthropic.DefaultAsyncHttpxClient(
        limits=httpx.Limits(
            max_connections=global_limit + 8, max_keepalive_connections=global_limit + 8
        ),
        timeout=60.0,
    )
    return anthropic.AsyncAnthropic(max_retries=0, timeout=60.0, http_client=http)


def key_or_exit() -> None:
    loaded = load_env_file(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (environment or .env); nothing was run.")
        sys.exit(2)
    source = "from .env" if "ANTHROPIC_API_KEY" in loaded else "from the environment"
    log(f"API key: found ({source})")


def write_report(stage: str, payload: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(payload, indent=1, default=str))
    return p


def runner_path(out: Path, alias: str) -> Path:
    return out / ladder.make_spec(alias, 1).filename


def append_api_runs(
    summary: dict, states: dict, n_subjects: int, seconds: float, stage: str
) -> None:
    """Append one line per model to results/api_runs.csv (same columns as run_anthropic.py)."""
    path = ROOT / "results" / "api_runs.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ladder.API_RUN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for alias, m in summary["models"].items():
            ms = states[alias]
            stopped = (
                (summary["halt"] or {}).get("reason")
                or (summary["paused"].get(alias) or {}).get("reason")
                or ""
            )
            w.writerow(
                {
                    "finished": time.strftime("%F %T"),
                    "spec_label": ms.spec.label + (f"__{stage}" if stage != "full" else ""),
                    "file": str(ms.path.relative_to(ROOT)),
                    "model": m["model_id"],
                    "condition": ms.spec.condition,
                    "thinking_param": ladder.thinking_for(ms.lm, ms.spec.condition)[2]
                    + " (ladder runner)",
                    "refusals": m["refusals"],
                    "refusal_log": json.dumps(ms.refusal_log),
                    "truncations": m["truncations"],
                    "subjects": n_subjects,
                    "n_calls": m["calls"],
                    "n_errors": m["errors"],
                    "seconds": round(seconds),
                    "stopped": stopped,
                    "complete": bool(m.get("completeness", {}).get("complete"))
                    and m["errors"] == 0,
                    "calls": m["calls"],
                    "cost_usd": m["cost_usd"],
                    **m["usage"],
                    "max_dollars": summary["ledger"]["stage_budget_usd"],
                    "max_calls": "",
                    "stop_reasons": json.dumps(m["stop_reasons"]),
                }
            )


async def run_stage(
    args,
    aliases,
    n_subjects,
    out_dir,
    trial_limit=None,
    max_calls_per_model=None,
    start=4,
    ceiling=32,
    stage="stage",
    mode="main",
    condition=ladder.CONDITION,
    subject_sequential=False,
):
    """Check the prompt, build the ledger, run every alias, append the csv lines; returns the
    summary (None after --dry-run)."""
    human = pipeline.prepare_human("data/human", ladder.TRIAL_SET)
    tbs = human["trials_by_subject"]
    fp = ladder.check_prompt_and_mapping(tbs)
    log(
        f"prompt v1 fingerprint ok: system {fp['system_sha256'][:12]}, user turns "
        f"{fp['user_turns_sha256'][:12]} ({fp['n_user_turns']} turns); subject mapping ok"
    )
    base = ladder.baseline_spend_usd(ROOT)
    ledger_path = (
        None
        if args.dry_run
        else LEDGERS / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    )
    ledger = ladder.Ledger(args.project_cap, base["baseline"], args.budget, ledger_path)
    log(
        f"ledger: baseline ${base['baseline']:.2f} = legacy runs ${base['legacy']:.2f} "
        f"(max of csv ${base['csv']:.2f}, jsonl ${base['jsonl']:.2f}) "
        f"+ {base['n_ledgers']} ladder ledger files ${base['ladder_ledgers']:.2f}; "
        f"project cap ${args.project_cap:.2f}; stage budget ${args.budget:.2f}; "
        f"left under cap ${args.project_cap - base['baseline']:.2f}"
    )
    if args.dry_run:
        from regime_shift.prompts import build_main_messages

        for a in aliases:
            kw = ladder.build_request(
                ladder.LADDER[a], build_main_messages(tbs[1][0], 2, ["50"]), condition
            )
            kw["system"] = kw["system"][:60] + "..."
            print(a, json.dumps(kw)[:600])
        return None
    key_or_exit()
    client = make_client(args.global_limit)
    runner = ladder.LadderRunner(
        client,
        ledger,
        out_dir=out_dir,
        global_limit=args.global_limit,
        start_concurrency=start,
        ceiling=ceiling,
        log=log,
        progress_every_s=args.progress,
        max_calls_per_model=max_calls_per_model,
        subject_sequential=subject_sequential,
    )
    runner.prepare(
        aliases, n_subjects, tbs, trial_limit=trial_limit, mode=mode, condition=condition
    )
    for a, ms in runner.states.items():
        log(
            f"[{a}] {len(ms.done)} rows on disk, {len(ms.pending)} trials pending -> "
            f"{ms.path.relative_to(ROOT)}"
        )
    t0 = time.time()
    try:
        summary = await runner.run()
    finally:
        await client.close()
    secs = time.time() - t0
    subjects = sorted(tbs)[:n_subjects]
    for a, ms in runner.states.items():
        trials = [t for s in subjects for t in tbs[s]]
        if trial_limit:
            trials = trials[:trial_limit]
        summary["models"][a]["completeness"] = ladder.completeness(ms.path, ms.spec, trials)
        summary["models"][a]["replay"] = ladder.replay_consistency(ms.path)
    summary["seconds"] = round(secs, 1)
    summary["stage"] = stage
    append_api_runs(summary, runner.states, n_subjects, secs, stage)
    if summary.get("stopped_by_budget"):
        log(f"STOPPED (budget): {summary['halt']}")
    elif summary["halt"]:
        log(f"HALTED: {summary['halt']}")
    for a, info in summary["paused"].items():
        log(f"PAUSED {a}: {info}")
    log(f"ledger: {json.dumps(summary['ledger'])}")
    return summary


async def preflight(args) -> int:
    """One number-only stateless call per model, one at a time; stop at the first anomaly."""
    from regime_shift.prompts import build_stateless_messages

    key_or_exit()
    human = pipeline.prepare_human("data/human", ladder.TRIAL_SET)
    trial = human["trials_by_subject"][1][0]
    base = ladder.baseline_spend_usd(ROOT)
    LEDGERS.mkdir(parents=True, exist_ok=True)
    ledger = ladder.Ledger(
        args.project_cap,
        base["baseline"],
        1.0,
        LEDGERS / f"preflight_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json",
    )
    client = make_client(4)
    report = []
    ok_all = True
    try:
        for alias in PREFLIGHT_MODELS:
            lm = ladder.LADDER[alias]
            kw = ladder.build_request(lm, build_stateless_messages(trial, 1), "number-only")
            est = ladder.estimate_cost_usd(lm, kw)
            await ledger.reserve(est)
            entry = {"alias": alias, "model_sent": lm.model_id}
            try:
                raw = await client.messages.with_raw_response.create(**kw)
                msg = raw.parse()
                if inspect.isawaitable(msg):
                    msg = await msg
                entry.update(
                    http_status=getattr(raw, "status_code", 200),
                    stop_reason=getattr(msg, "stop_reason", None),
                    model_returned=getattr(msg, "model", None),
                    request_id=getattr(raw, "request_id", None),
                )
                text = "".join(
                    getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text"
                )
                entry["text"] = text[:120]
                u_ = msg.usage
                usage = {
                    "input_tokens": u_.input_tokens,
                    "output_tokens": u_.output_tokens,
                    "cache_read_input_tokens": getattr(u_, "cache_read_input_tokens", 0) or 0,
                    "cache_creation_input_tokens": getattr(u_, "cache_creation_input_tokens", 0)
                    or 0,
                }
                cost = ladder.call_cost_usd(lm.price, usage)
                await ledger.commit(est, cost)
                entry["cost_usd"] = round(cost, 6)
                try:
                    entry["integer"] = int(ladder.number_from_json(text))
                except Exception as exc:  # noqa: BLE001
                    entry["integer"] = None
                    entry["parse_error"] = repr(exc)
                entry["clean"] = bool(
                    entry["stop_reason"] == "end_turn"
                    and entry["model_returned"] == lm.model_id
                    and entry["integer"] is not None
                    and 0 <= entry["integer"] <= 100
                )
            except Exception as exc:  # noqa: BLE001
                await ledger.release(est)
                entry.update(
                    http_status=getattr(exc, "status_code", None),
                    error_type=type(exc).__name__,
                    body=ladder._scrub(str(getattr(exc, "body", None) or exc))[:800],
                    clean=False,
                )
            report.append(entry)
            log(f"preflight {alias}: {json.dumps(entry)}")
            if not entry["clean"]:
                ok_all = False
                log("PREFLIGHT STOP: anomaly; no further calls")
                break
    finally:
        await client.close()
    p = write_report("preflight", {"ok": ok_all, "calls": report, "ledger": ledger.summary()})
    log(f"preflight report: {p.relative_to(ROOT)}; ok={ok_all}")
    return 0 if ok_all else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument(
        "--budget", type=float, default=None, help="stage budget in USD (required for API stages)"
    )
    ap.add_argument("--project-cap", type=float, default=185.0)
    ap.add_argument("--global-limit", type=int, default=96)
    ap.add_argument("--progress", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--probe-cap",
        type=float,
        default=30.0,
        help="sonnet-probe: stop and ask if the 5-subject projection exceeds this",
    )
    ap.add_argument(
        "--mode",
        default="main",
        choices=["main", "stateless"],
        help="protocol for canary, one-subject and full",
    )
    args = ap.parse_args()

    if args.stage == "catalogue":
        print(f"Catalogue read {ladder.CATALOGUE_DATE} (List Models + platform.claude.com docs)\n")
        print(ladder.catalogue_table())
        return 0
    if args.stage == "check":
        human = pipeline.prepare_human("data/human", ladder.TRIAL_SET)
        print(json.dumps(ladder.check_prompt_and_mapping(human["trials_by_subject"]), indent=1))
        print(json.dumps(ladder.baseline_spend_usd(ROOT), indent=1))
        return 0
    if args.budget is None and not args.dry_run:
        ap.error("--budget is required for API stages")
    args.budget = args.budget or 0.0

    if args.stage == "regression":
        aliases = args.models or ["claude-sonnet-5-5", "claude-haiku-5-5"]
        out = ROOT / "results" / "ladder" / "stage2" / "regression"
        summary = asyncio.run(run_stage(args, aliases, 1, out, stage="regression"))
        if summary is None:
            return 0
        for a in aliases:
            old = ROOT / "results" / ladder.make_spec(a, 30).filename
            summary["models"][a]["regression"] = ladder.compare_with_existing(
                runner_path(out, a), old
            )
            log(f"[{a}] regression vs {old.name}: {summary['models'][a]['regression']}")
    elif args.stage == "canary":
        aliases = args.models or list(DEFAULT_LADDER)
        out = ROOT / "results" / "ladder" / "stage2" / "canary"
        # 20 calls per model = the first two trials of subject 1, at most 2 concurrent trials
        summary = asyncio.run(
            run_stage(
                args,
                aliases,
                1,
                out,
                trial_limit=2,
                max_calls_per_model=20,
                start=2,
                ceiling=2,
                stage="canary",
                mode=args.mode,
            )
        )
    elif args.stage == "one-subject":
        aliases = args.models or list(DEFAULT_LADDER)
        out = ROOT / "results" / "ladder" / "stage2" / "one_subject"
        summary = asyncio.run(run_stage(args, aliases, 1, out, stage="one-subject", mode=args.mode))
    elif args.stage == "preflight":
        return asyncio.run(preflight(args))
    elif args.stage == "stateless-ladder":
        aliases = args.models or list(STATELESS_LADDER)
        summary = asyncio.run(
            run_stage(
                args,
                aliases,
                15,
                ROOT / "results" / "partial",
                stage="stateless-ladder",
                mode="stateless",
                subject_sequential=True,
            )
        )
    elif args.stage == "sonnet-stateless":
        aliases = args.models or ["claude-sonnet-5-5"]
        summary = asyncio.run(
            run_stage(
                args, aliases, 30, ROOT / "results", stage="sonnet-stateless", mode="stateless"
            )
        )
    elif args.stage == "sonnet-probe":
        # 20 reasoning-allowed calls on the first two trials of subject 1, concurrency 2
        aliases = args.models or ["claude-sonnet-5-5"]
        out = ROOT / "results" / "ladder" / "stage4" / "probe"
        summary = asyncio.run(
            run_stage(
                args,
                aliases,
                1,
                out,
                trial_limit=2,
                max_calls_per_model=20,
                start=2,
                ceiling=2,
                stage="sonnet-probe",
                condition="reasoning",
            )
        )
        if summary is not None:
            for a, m in summary["models"].items():
                if m["calls"]:
                    th = m["usage"]["thinking_tokens"] / m["calls"]
                    proj = m["cost_per_call_usd"] * 5 * 720
                    m["probe"] = {
                        "mean_thinking_tokens": round(th),
                        "projected_5_subjects_usd": round(proj, 2),
                        "cap_usd": args.probe_cap,
                        "over_cap": proj > args.probe_cap,
                    }
                    verdict = (
                        "STOP AND ASK (over the cap)" if proj > args.probe_cap else "within the cap"
                    )
                    log(
                        f"[{a}] PROBE: mean thinking tokens {th:.0f}/call, cost/call "
                        f"${m['cost_per_call_usd']:.5f}, projected 5 subjects ${proj:.2f} -> "
                        f"{verdict} ${args.probe_cap:.0f}"
                    )
    elif args.stage == "sonnet-reasoning":
        aliases = args.models or ["claude-sonnet-5-5"]
        summary = asyncio.run(
            run_stage(
                args,
                aliases,
                5,
                ROOT / "results" / "partial",
                stage="sonnet-reasoning",
                condition="reasoning",
            )
        )
    else:
        aliases = args.models or list(DEFAULT_LADDER)
        out = ROOT / "results"
        summary = asyncio.run(run_stage(args, aliases, 30, out, stage="full", mode=args.mode))
    if summary is None:
        return 0
    p = write_report(args.stage, summary)
    log(f"report: {p.relative_to(ROOT)}")
    for m in summary["models"].values():
        if m["calls"]:
            m["cache_reads_happened"] = m["usage"]["cache_read_input_tokens"] > 0
            m["projected_full_run_usd"] = round(m["cost_per_call_usd"] * 21600, 2)
    keys = (
        "calls",
        "errors",
        "n429",
        "cost_usd",
        "cost_per_call_usd",
        "projected_full_run_usd",
        "cache_reads_happened",
        "cached_input_share",
        "calls_per_second",
        "concurrency_final",
        "stop_reasons",
        "api_models",
        "completeness",
    )
    print(
        json.dumps(
            {a: {k: m.get(k) for k in keys} for a, m in summary["models"].items()},
            indent=1,
            default=str,
        )
    )
    print("ledger:", json.dumps(summary["ledger"]))
    ok = summary["halt"] is None and not summary["paused"] and not summary.get("failures")
    if args.stage == "sonnet-probe" and any(
        m.get("probe", {}).get("over_cap") for m in summary["models"].values()
    ):
        return 3  # projection over --probe-cap: stop and ask
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

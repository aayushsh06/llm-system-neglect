"""Run the main (multi-turn, v1 prompt) protocol on a Claude model through the Anthropic API.

    python scripts/collection/run_anthropic.py --model claude-sonnet-5-5 --condition number-only \
        --subjects 5 --max-dollars 10 --max-calls 4000 --parallel 8

Subjects < 30 are written as a PARTIAL file under results/partial (kept out of the main
tables); 30 subjects go to results/. Resume is automatic (rows on disk are skipped). The
API key is read by the SDK from ANTHROPIC_API_KEY and never written anywhere. A one-line
summary per run (no key) is appended to results/api_runs.csv. --dry-run prints the request
shape and a projected cost without calling the API. --promote-from copies an existing PARTIAL
file to the full-run file name (relabelled) before resuming to 30 subjects.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from regime_shift import pipeline  # noqa: E402
from regime_shift.collection import anthropic_backend as ab  # noqa: E402
from regime_shift.collection.runner import read_jsonl, run_spec  # noqa: E402
from regime_shift.config import PartialRunSpec, RunSpec  # noqa: E402
from regime_shift.prompts import build_main_messages  # noqa: E402

SEED = 20251120  # same seed constant as every local run (only labels/keys use it here)
TRIAL_SET = "behavioral"
API_TEMPERATURE = 1.0  # the API default; the parameter itself is not sent (400 on 5.x models)
CHARS_PER_TOKEN = 3.1  # conservative for the newer tokenizer (~30% more tokens than the older one)
API_RUN_FIELDS = [
    "finished",
    "spec_label",
    "file",
    "model",
    "condition",
    "thinking_param",
    "refusals",
    "refusal_log",
    "truncations",
    "subjects",
    "n_calls",
    "n_errors",
    "seconds",
    "stopped",
    "complete",
    "calls",
    "cost_usd",
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "thinking_tokens",
    "max_dollars",
    "max_calls",
    "stop_reasons",
]


def make_spec(model: str, condition: str, n_subjects: int) -> RunSpec:
    cls = PartialRunSpec if n_subjects < 30 else RunSpec
    return cls(
        model=model,
        mode="main",
        temperature=API_TEMPERATURE,
        seed=SEED,
        n_subjects=n_subjects,
        trial_set=TRIAL_SET,
        num_predict=16,
        condition=condition,
    )


def projected_cost(model: str, condition: str, n_subjects: int, trials_by_subject) -> dict:
    """Rough token count from characters; cached and uncached input, no thinking tokens."""
    price = ab.PRICES[model]
    subjects = sorted(trials_by_subject)[:n_subjects]
    uncached = cached = calls = 0.0
    for s in subjects:
        for t in trials_by_subject[s]:
            prev_tokens = 0
            for k in range(1, 11):
                msgs = build_main_messages(t, k, ["50"] * (k - 1))
                tokens = sum(len(m["content"]) for m in msgs) / CHARS_PER_TOKEN
                uncached += tokens
                # prefix beyond 512 tokens is read from cache, the rest written/paid in full
                if prev_tokens >= 512:
                    cached += prev_tokens * (price.cache_read / price.input)
                    cached += (tokens - prev_tokens) * (price.cache_write_5m / price.input)
                else:
                    cached += tokens
                prev_tokens = tokens
                calls += 1
    out_tokens = calls * (9 if condition == "number-only" else 300)  # reasoning tokens are a guess
    if condition == "number-only":
        uncached += calls * 190  # the JSON schema adds ~190 input tokens per request (measured)
        cached += 190 * (price.cache_read / price.input) * calls
    return {
        "calls": int(calls),
        "input_tokens_est": int(uncached),
        "cost_uncached_usd": round(
            uncached * price.input / 1e6 + out_tokens * price.output / 1e6, 2
        ),
        "cost_cached_usd": round(cached * price.input / 1e6 + out_tokens * price.output / 1e6, 2),
        "output_tokens_assumed": int(out_tokens),
    }


def promote(model: str, condition: str, from_n: int, out: Path) -> None:
    """Copy a PARTIAL file to the full-run file name, relabelling the ``source`` field."""
    part = make_spec(model, condition, from_n)
    full = make_spec(model, condition, 30)
    src, dst = out / "partial" / part.filename, out / full.filename
    if dst.exists() or not src.exists():
        print(
            f"promote: nothing to do ({dst.name} exists: {dst.exists()}, "
            f"{src.name} exists: {src.exists()})"
        )
        return
    rows = read_jsonl(src)
    with dst.open("w") as fh:
        for r in rows:
            r["source"] = full.source
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"promote: {len(rows)} rows copied into {dst.name} (partial kept)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ab.PRICES))
    ap.add_argument("--condition", default="number-only", choices=ab.CONDITIONS)
    ap.add_argument("--subjects", type=int, default=5)
    ap.add_argument("--max-dollars", type=float, required=True)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-refusals", type=int, default=1, help="stop the run at the Nth refusal")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--promote-from", type=int, default=None, help="PARTIAL subject count to promote"
    )
    args = ap.parse_args()
    out = ROOT / args.out
    human = pipeline.prepare_human("data/human", TRIAL_SET)
    tbs = human["trials_by_subject"]
    spec = make_spec(args.model, args.condition, args.subjects)
    dest = out / "partial" if args.subjects < 30 else out
    proj = projected_cost(args.model, args.condition, args.subjects, tbs)
    print(f"spec: {spec.label} -> {dest / spec.filename}")
    print(f"projected: {proj}")
    if args.dry_run:
        be = ab.AnthropicBackend(
            args.model,
            args.condition,
            max_dollars=0,
            max_calls=0,
            client=object(),
            max_tokens=args.max_tokens,
        )
        msgs = build_main_messages(tbs[1][0], 2, ["50"])
        kw = be.request_kwargs(msgs)
        kw["system"] = kw["system"][:80] + "..."
        print("request shape:", json.dumps(kw, indent=1)[:1500])
        return 0
    env_path = ROOT / ".env"
    loaded = ab.load_env_file(env_path)
    if env_path.exists() and (env_path.stat().st_mode & 0o077):
        print(f"note: {env_path} is readable by other users; consider chmod 600")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            f"ANTHROPIC_API_KEY is not set (checked the environment and {env_path}); "
            "nothing was run."
        )
        return 2
    source = "from .env" if "ANTHROPIC_API_KEY" in loaded else "from the environment"
    print(f"API key: found ({source})")
    if args.promote_from:
        promote(args.model, args.condition, args.promote_from, out)
    be = ab.AnthropicBackend(
        args.model,
        args.condition,
        max_dollars=args.max_dollars,
        max_calls=args.max_calls,
        max_tokens=args.max_tokens,
        cache=not args.no_cache,
        max_refusals=args.max_refusals,
    )
    print(
        f"thinking: {be.thinking_desc}; max_tokens {be.max_tokens}; cache {be.cache}; "
        f"caps ${be.spend.max_dollars:.2f} / {be.spend.max_calls} calls"
    )
    t0 = time.time()
    st = run_spec(spec, be, tbs, dest, parallel=args.parallel, progress=print, progress_every=20)
    summary = be.spend.summary()
    print("SPEND:", json.dumps(summary))
    print("REFUSALS:", json.dumps(be.refusal_log), "| truncations:", be.truncations)
    row = {
        "finished": time.strftime("%F %T"),
        "spec_label": spec.label,
        "file": str((dest / spec.filename).relative_to(ROOT)),
        "model": args.model,
        "condition": args.condition,
        "thinking_param": be.thinking_desc,
        "refusals": len(be.refusal_log),
        "refusal_log": json.dumps(be.refusal_log),
        "truncations": be.truncations,
        "subjects": args.subjects,
        "n_calls": st.n_calls,
        "n_errors": st.n_errors,
        "seconds": round(time.time() - t0),
        "stopped": st.stopped or "",
        "complete": st.complete,
        **{k: v for k, v in summary.items() if k != "stop_reasons"},
        "stop_reasons": json.dumps(summary["stop_reasons"]),
    }
    path = out / "api_runs.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=API_RUN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in API_RUN_FIELDS})
    return 0 if st.complete else 1


if __name__ == "__main__":
    sys.exit(main())

"""Run one open-weight model on the 30 subjects' sequences through a local Ollama server.

    python scripts/collection/run_local.py --model qwen2.5:7b  # multi-turn, 30 subjects
    python scripts/collection/run_local.py --model qwen2.5:7b --mode stateless
    python scripts/collection/run_local.py --model qwen2.5:7b --subjects 10

Uses the same seed, temperature, token limit and prompt as the runs in the paper. Rows are
written one per call to results/<label>.jsonl and a run resumes from the rows on disk. The
Ollama server must be running (``ollama serve``) and the model pulled (``ollama pull <tag>``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from regime_shift import pipeline  # noqa: E402
from regime_shift.collection.backends import OllamaBackend  # noqa: E402
from regime_shift.collection.runner import run_spec  # noqa: E402
from regime_shift.config import PartialRunSpec, RunSpec  # noqa: E402

SEED = 20251120
NUM_PREDICT = 8
TEMPERATURE = 0.7
TRIAL_SET = "behavioral"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Ollama tag, e.g. qwen2.5:7b")
    ap.add_argument("--mode", default="main", choices=["main", "stateless"])
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--data", default=str(ROOT / "data" / "human"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    human = pipeline.prepare_human(args.data, TRIAL_SET)
    print(pipeline.run_gate(human["human_all"], human["published"]).message)
    cls = RunSpec if args.subjects >= 30 else PartialRunSpec
    spec = cls(
        model=args.model,
        mode=args.mode,
        temperature=args.temperature,
        seed=SEED,
        n_subjects=args.subjects,
        trial_set=TRIAL_SET,
        num_predict=NUM_PREDICT,
    )
    out = Path(args.out) if args.subjects >= 30 else Path(args.out) / "partial"
    backend = OllamaBackend(args.model, host=args.host, think=False)
    trials = {s: t for s, t in human["trials_by_subject"].items() if s <= args.subjects}
    run_spec(spec, backend, trials, out, parallel=args.parallel)
    print(f"written: {out / spec.filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

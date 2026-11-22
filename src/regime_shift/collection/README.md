# Data collection

This folder is the record of how the result files were produced. Its logic is kept exactly as
it was when the runs were made; fixes and tidying stop at comments and names.

| file | role |
|---|---|
| `backends.py` | the Ollama chat backend (open-weight models), the JSON answer schema, and two mock observers used by the tests |
| `runner.py` | one JSON line per call, per-call seeds, resume from the rows already on disk |
| `anthropic_backend.py` | the Claude backend on the Anthropic SDK: thinking settings per model, prices, cost tracking, refusal stop |
| `ladder.py` | the parallel, budget-capped runner used for the five Claude models (ledger, breakers, adaptive concurrency) |

The scripts that drive them are in `scripts/collection/`: `run_local.py` (one open-weight
model through Ollama), `run_anthropic.py` (one Claude model, one run at a time) and
`run_ladder.py` (the Claude ladder stage by stage). The API key is read from `.env` or the
environment and is never written to any file.

# System neglect in large language models

Code and results for the paper

> **System Neglect and Sequential Bayesian Inference in Large Language Models: Trial-Matched
> Evidence from a Human Regime-Shift Detection Paradigm.** Aayush Sharma, 2026.

When people judge whether a hidden process has changed, they respond to the evidence in front
of them and neglect the system that produced it: they over-react when change is rare and
signals are noisy, and under-react when change is frequent and signals are precise. The paper
tests whether language models share this *system neglect* by running a published regime-shift
detection experiment (Wang, Wu and Wu, eLife, 2025; task from Massey and Wu, 2005) on eleven models,
six open-weight and five proprietary models (Anthropic's Claude family), with each simulated
subject given the exact ball sequences seen by one of the 30 original participants. Each source
is summarised by two sensitivities: how much of the stated switch probability and of the stated
signal diagnosticity it acts on, where a Bayesian observer scores 1 and complete neglect scores
0. Humans score 0.26 and 0.29. Small open-weight models score near zero. The Claude models match
or exceed humans, are all closer to the Bayesian posterior than the human average, and the most
recent scores 1 on both while still under-weighting the switch probability by a constant factor.
Rule-based agents with known sensitivities validate the measure and show that the parameter
pattern usually read as system neglect also appears in agents that ignore the task.

Every number, table and figure in the paper is generated from the result files in this
repository by one script; nothing is typed by hand.

## Layout

```
src/regime_shift/        the package
  task.py                conditions, trial generation, seeding
  bayes.py               forward filter and the system-neglect model
  fitting.py             per-subject nonlinear least squares, as in the authors' MATLAB code
  analysis.py            distance from the posterior, index of overreaction, correlations
  validation.py          the validation gate against the published estimates
  human_data.py          OSF download with md5 checks, MATLAB-table decoder, long-format loader
  prompts.py             instruction text; multi-turn and stateless prompts
  parsing.py             reply parser, one status per call
  config.py              run specifications, labels and file names
  results.py             result files -> long table
  pipeline.py            prepare_human -> run_gate -> simulate -> analyze
  sanity.py              model-free checks: copying, calibration, signal response
  paper_figures.py       the paper's figures, and the source names and order used by the tables
  collection/            the data-collection code, kept as it was used (see its README):
    backends.py          Ollama backend and two mock observers
    runner.py            one JSON line per call, resume after interruption
    anthropic_backend.py Claude backend (Anthropic SDK), prices, cost tracking
    ladder.py            parallel, budget-capped runner for the Claude models
scripts/
  fetch_human_data.py    download the human data from OSF and verify checksums
  build_paper_numbers.py regenerate every number, table and figure of the paper
  explore_sensitivity.py the system-sensitivity measure (imported by the builder)
  explore_fake_subjects.py the rule-based agents (writes results/fake/)
  collection/
    run_local.py         run an open-weight model through Ollama
    run_anthropic.py     run a Claude model, one run at a time
    run_ladder.py        run the Claude ladder stage by stage with budget caps
examples/quickstart.py   a one-minute tour: one posterior, one fit, one model's sensitivities
tests/                   pytest suite, including the validation gate
llm-system-neglect-results.tar.zst   every result file, raw and derived; SHA256SUMS.txt beside it
```

Two folders are created locally and never committed: `data/human/` (downloaded from OSF) and
`results/` (unpacked from the archive).

## Install

Python 3.10 or newer. From the repository root:

```
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,api]"
```

The `api` extra installs the Anthropic SDK and is needed only to rerun a Claude model. Running
an open-weight model needs [Ollama](https://ollama.com/).

## Human data

The human data are not included: the OSF project that holds them (<https://osf.io/xh7dy/>)
sets no licence, so they are not redistributed here. Download the three files the analysis
uses (the Experiment 1 behavioural data and the two published parameter tables, about 1.2 MB):

```
python scripts/fetch_human_data.py
```

The script writes them to `data/human/`, checks every file against the md5 recorded on OSF and
stops on any mismatch. The test suite and every script stop without them.

## Results

All result files are in the repository as one compressed archive,
`llm-system-neglect-results.tar.zst` (25 MB). Verify it and unpack it at the repository root:

```
shasum -a 256 -c SHA256SUMS.txt
tar -xf llm-system-neglect-results.tar.zst          # bsdtar, or GNU tar 1.31+ with zstd
# or: zstd -dc llm-system-neglect-results.tar.zst | tar -xf -
```

This creates `results/` (697 MB, 90 files):

| path | contents |
|---|---|
| `results/*.jsonl` | the complete 30-subject runs, one JSON line per model call: prompt state, reply, parse status, token counts, cost |
| `results/partial/` | the partial runs that appear in the appendix tables |
| `results/variants/`, `results/greedy/` | prompt-variant and greedy-decoding runs, counted in the parser audit only |
| `results/fake/` | the rule-based agents and their analysis |
| `results/analysis/` | derived CSVs of the main runs |
| `results/model_info.csv` | digest, size and quantisation of every open-weight model |
| `results/api_runs.csv` | one summary line per API run: calls, tokens, cost (no key) |

## Quick look

`results/summary.csv` (in the archive, and rewritten by the builder) has one row per source:
subjects, the two sensitivities and the distance from the posterior with their intervals,
alpha and beta by level, and the fit's R². For a one-minute tour of the pipeline on real data:

```
python examples/quickstart.py
```

It loads the human data, computes the Bayesian posterior for one trial, fits the
system-neglect model for one subject, and prints one Claude model's two sensitivities and
its distance from the posterior.

## Reproduce every number, table and figure

With the human data and the results in place:

```
python -m pytest                       # the suite, including the validation gate
python scripts/build_paper_numbers.py  # about four minutes
```

The builder recomputes every per-subject fit and bootstrap from the result files and writes,
into a `paper/` folder it creates, `numbers.tex` (one macro per number), `tables/*.tex`,
`figures/*.pdf` and `*.png`, and the intermediate CSVs under `data/`. The bootstrap is seeded,
so a rebuild reproduces the paper's values exactly. The rule-based agents can be regenerated
with `python scripts/explore_fake_subjects.py` (no model calls; seeded).

## Rerun a model

**Open-weight models** (Ollama). Start a server, pull the model, then:

```
python scripts/collection/run_local.py --model qwen2.5:7b        # multi-turn, 30 subjects
python scripts/collection/run_local.py --model qwen2.5:7b --mode stateless
```

Seed, temperature (0.7), token limit and prompt are those of the paper. Rows are appended to
`results/<label>.jsonl` and an interrupted run resumes from the file. Ollama does not pin model
weights; the digest of every model used is in `results/model_info.csv`.

**Claude models** (Anthropic API). Copy `.env.example` to `.env` and put your key in it; the
key is read by the SDK and never written to any file. Every run has a hard dollar cap and a
call cap, stops at the first refusal, and appends one line to `results/api_runs.csv`:

```
python scripts/collection/run_anthropic.py --model claude-sonnet-5-5 --condition number-only \
    --subjects 30 --max-dollars 40 --max-calls 22000 --dry-run     # cost projection, no calls
python scripts/collection/run_ladder.py catalogue                   # the ladder, no calls
python scripts/collection/run_ladder.py full --budget 150           # 30 subjects per model
```

`scripts/collection/run_ladder.py --help` lists the stages (prompt check, regression, canary, one subject, full,
stateless, reasoning). Sampling through the API is the provider's default and cannot be seeded,
so a rerun agrees with the published files in distribution, not call by call.

## Software versions used for the paper

| software | version |
|---|---|
| Python | 3.14.0 |
| numpy / scipy / pandas | 2.5.5 / 1.18.1 / 3.1.1 |
| matplotlib | 3.11.2 |
| anthropic (SDK) | 2.12.0 |
| pytest / ruff | 9.2.0 / 0.17.6 |
| Ollama | 0.40.17 (macOS, Metal) |

Open-weight models: Qwen2.5 3B, 7B, 14B and 32B, Llama 3.1 8B and Gemma 2 9B, 4-bit builds from
the Ollama library. Claude models: Haiku 4.5, Haiku 5.5, Sonnet 4.6, Sonnet 5 and Sonnet 5.5,
run in November 2026.

## Licence

The code is released under the MIT licence (`LICENSE`). The result files in the archive and the
paper are released under the Creative Commons Attribution 4.0 licence (CC BY 4.0). The human
data belong to Wang, Wu and Wu and are not redistributed.

## How to cite

```
@misc{sharma2026systemneglect,
  author = {Sharma, Aayush},
  title  = {System Neglect and Sequential Bayesian Inference in Large Language Models:
            Trial-Matched Evidence from a Human Regime-Shift Detection Paradigm},
  year   = {2026},
  url    = {https://github.com/aayushsh06/llm-system-neglect},
}
```

Massey, C. and Wu, G. (2005). Detecting regime shifts: the causes of under- and overreaction.
*Management Science* 51(6), 932–947. Wang, Wu and Wu (2025). *eLife* 14:RP104684, data and code at
<https://osf.io/xh7dy/>.

"""Run configuration: one :class:`RunSpec` per (model, mode, temperature) run.

The spec's ``label`` names the JSONL result file and the ``source`` column of
the analysis tables, so both are fixed here and nowhere else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_MODELS: tuple[str, ...] = (
    "qwen2.5:3b",
    "qwen2.5:7b",
    "phi3.5:3.8b",
    "llama3.1:8b",
    "gemma2:9b",
)
MODES: tuple[str, ...] = ("main", "stateless")


def sanitize(name: str) -> str:
    """Make a model tag or variant name safe for a file name (``qwen2.5:7b`` -> ``qwen2.5_7b``)."""
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", name)


def condition_name(condition: str) -> str:
    """Human-readable condition: 'number only' or 'reasoning allowed (effort low)'."""
    if not condition:
        return ""
    if condition == "number-only":
        return "number only"
    if condition == "reasoning":
        return "reasoning allowed"
    if condition.startswith("reasoning-"):
        return f"reasoning allowed (effort {condition.split('-', 1)[1]})"
    return condition


@dataclass(frozen=True)
class RunSpec:
    """One (model, mode, temperature) run over ``n_subjects`` simulated subjects.

    ``mode`` is ``"main"`` (multi-turn: the model's earlier replies are replayed) or
    ``"stateless"`` (one call per period with the full history in the prompt).
    ``prompt_variant`` selects one of the prompt variants; ``"v1"`` is the main prompt.
    ``condition`` is ``""`` for a free-text number (the local default), ``"number-only"``
    for the JSON answer schema, and ``"reasoning"`` / ``"reasoning-low"`` ... for reasoning
    allowed at that effort.
    """

    model: str
    mode: str
    temperature: float
    seed: int
    n_subjects: int
    trial_set: str
    num_predict: int
    synthetic: bool = False
    prompt_variant: str = "v1"
    condition: str = ""

    def __post_init__(self) -> None:
        from .prompts import VARIANTS

        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.prompt_variant not in VARIANTS:
            raise ValueError(f"prompt_variant must be one of {VARIANTS}")
        if self.trial_set not in ("behavioral", "all"):
            raise ValueError("trial_set must be 'behavioral' or 'all'")

    @property
    def label(self) -> str:
        tag = f"{sanitize(self.model)}__{self.mode}__T{self.temperature:g}__{self.trial_set}"
        if self.prompt_variant != "v1":
            tag += f"__{sanitize(self.prompt_variant)}"
        if self.condition:
            tag += f"__{sanitize(self.condition)}"
        return ("synthetic__" if self.synthetic else "") + tag

    @property
    def filename(self) -> str:
        return self.label + ".jsonl"

    @property
    def source(self) -> str:
        """Label used in analysis tables."""
        greedy = "greedy" if self.temperature == 0 else f"T{self.temperature:g}"
        variant = "" if self.prompt_variant == "v1" else f", {self.prompt_variant}"
        cond = f", {condition_name(self.condition)}" if self.condition else ""
        return f"{self.model} [{self.mode}, {greedy}{variant}{cond}]"


class PartialRunSpec(RunSpec):
    """Same keys and seeds as the full spec; different file name and source label."""

    @property
    def filename(self) -> str:  # type: ignore[override]
        return self.label + f"__PARTIAL_s1-{self.n_subjects}.jsonl"

    @property
    def source(self) -> str:  # type: ignore[override]
        return RunSpec.source.fget(self)[:-1] + f", PARTIAL s1-{self.n_subjects}]"


@dataclass
class ExperimentConfig:
    """A set of runs over several models, expanded by :meth:`runs`."""

    models: tuple[str, ...] = DEFAULT_MODELS
    n_subjects: int = 30
    trial_set: str = "behavioral"
    modes: tuple[str, ...] = ("main", "stateless")
    temperature: float = 0.7
    greedy_run: bool = True
    seed: int = 20251120
    num_predict: int = 8
    parallel: int = 4
    output_dir: Path = field(default_factory=lambda: Path("results"))
    pilot: bool = False
    synthetic: bool = False

    def effective(self) -> ExperimentConfig:
        """Apply the pilot flag: one model, five subjects."""
        if self.pilot:
            return replace(self, models=self.models[:1], n_subjects=min(self.n_subjects, 5))
        return self

    def runs(self) -> list[RunSpec]:
        """Run order: main mode for every model first, then greedy, then stateless."""
        cfg = self.effective()
        common = {
            "seed": cfg.seed,
            "n_subjects": cfg.n_subjects,
            "trial_set": cfg.trial_set,
            "num_predict": cfg.num_predict,
            "synthetic": cfg.synthetic,
        }
        specs: list[RunSpec] = []
        if "main" in cfg.modes:
            specs += [RunSpec(m, "main", cfg.temperature, **common) for m in cfg.models]
            if cfg.greedy_run:
                specs += [RunSpec(m, "main", 0.0, **common) for m in cfg.models]
        if "stateless" in cfg.modes:
            specs += [RunSpec(m, "stateless", cfg.temperature, **common) for m in cfg.models]
        return specs

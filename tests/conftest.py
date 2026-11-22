from __future__ import annotations

import os
from pathlib import Path

import pytest

from regime_shift.analysis import add_bayes
from regime_shift.human_data import load_human_long, load_published_parameters

DATA_DIR = Path(
    os.environ.get("REGIME_SHIFT_DATA", Path(__file__).resolve().parents[1] / "data/human")
)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA_DIR / "data_allsubj_expt1.mat").exists():
        pytest.skip(f"human data not found in {DATA_DIR}; download it from https://osf.io/xh7dy/")
    return DATA_DIR


@pytest.fixture(scope="session")
def human_all(data_dir: Path):
    return add_bayes(load_human_long(data_dir / "data_allsubj_expt1.mat", "all"))


@pytest.fixture(scope="session")
def published(data_dir: Path):
    return load_published_parameters(data_dir / "BETA_neglect_final.mat")

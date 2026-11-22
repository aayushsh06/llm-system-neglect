import numpy as np
import pytest

from regime_shift import analysis as an
from regime_shift.fitting import PARAM_NAMES, fit_all_subjects
from regime_shift.validation import run_validation_gate


def test_validation_gate_reproduces_published_estimates(human_all, published):
    gate = run_validation_gate(human_all, published)
    assert gate.passed, gate.message
    assert gate.max_abs_subject_diff < 1e-3
    np.testing.assert_allclose(gate.our_means, [3.69, 1.04, 0.65, 1.69, 0.77, 0.57], atol=0.005)
    np.testing.assert_allclose(gate.our_means, gate.archive_means, atol=1e-4)
    assert gate.fits.success.all() and len(gate.fits) == 30


def test_gate_requires_both_sessions(human_all, published):
    with pytest.raises(ValueError):
        run_validation_gate(human_all[human_all.session == 0], published)


def test_behavioral_only_fit_differs_from_published(human_all, published):
    fits = fit_all_subjects(human_all[human_all.session == 0])
    diff = np.abs(fits[list(PARAM_NAMES)].to_numpy() - published[list(PARAM_NAMES)].to_numpy())
    assert diff.max() > 0.1  # the published values come from all 99 trials


def test_human_io_grid_matches_figure_2c(human_all):
    io = an.grid_matrix(an.grid_summary(an.index_of_overreaction(human_all), "io"), "human")
    assert io.loc[1.5, 0.01] == pytest.approx(0.0165, abs=5e-4)
    assert io.loc[3.0, 0.10] == pytest.approx(-0.0097, abs=5e-4)
    assert io.loc[9.0, 0.05] == pytest.approx(-0.0014, abs=5e-4)
    # system-neglect pattern: overreaction at low q / low d, underreaction at high q
    assert io.loc[1.5, 0.01] > 0 > io.loc[9.0, 0.10]

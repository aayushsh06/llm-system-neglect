import numpy as np
import pandas as pd
import pytest

from regime_shift.collection.backends import BayesianMock, NeglectMock
from regime_shift.fitting import PARAM_NAMES, fit_all_subjects, fit_system_neglect
from regime_shift.prompts import build_stateless_messages
from regime_shift.task import generate_subject_trials


def simulate_long(backend, subjects=(1, 2), n_blocks=11, seed=7, temperature=0.7):
    rows = []
    for s in subjects:
        for t in generate_subject_trials(s, seed, n_blocks=n_blocks):
            for p in range(1, 11):
                msgs = build_stateless_messages(t, p)
                text = backend.chat(
                    msgs, temperature=temperature, seed=1000 * t.trial + p, num_predict=8
                ).text
                rows.append(
                    {
                        "source": backend.name,
                        "subject": s,
                        "trial": t.trial,
                        "condition": t.condition,
                        "q": t.q,
                        "d": t.d,
                        "period": p,
                        "signal": t.signals[p - 1],
                        "estimate": int(text) / 100,
                    }
                )
    return pd.DataFrame(rows)


def test_fit_recovers_bayes_from_bayesian_mock():
    df = simulate_long(BayesianMock())
    fits = fit_all_subjects(df)
    assert fits.success.all()
    np.testing.assert_allclose(fits[list(PARAM_NAMES)].to_numpy(), 1.0, atol=0.06)


def test_fit_recovers_known_neglect_parameters():
    al, be = (3.0, 1.0, 0.6), (1.6, 0.8, 0.5)
    df = simulate_long(NeglectMock(al, be, noise_sd=0.0))
    fits = fit_all_subjects(df)
    assert fits.success.all()
    got = fits[list(PARAM_NAMES)].mean().to_numpy()
    np.testing.assert_allclose(got, [*al, *be], rtol=0.08, atol=0.04)


def test_fit_with_noise_is_close_and_handles_nan():
    al, be = (3.0, 1.0, 0.6), (1.6, 0.8, 0.5)
    df = simulate_long(NeglectMock(al, be, noise_sd=0.05), subjects=(1,))
    df.loc[df.sample(frac=0.02, random_state=0).index, "estimate"] = np.nan
    fit = fit_system_neglect(df)
    assert fit.success and fit.n_obs == int(df.estimate.notna().sum())
    np.testing.assert_allclose(fit.params, [*al, *be], rtol=0.25, atol=0.15)


def test_unidentified_fit_is_flagged_not_dropped():
    df = simulate_long(BayesianMock(), subjects=(1,), n_blocks=1)
    df.loc[df.q == 0.01, "estimate"] = np.nan
    fit = fit_system_neglect(df)
    assert not fit.success and "no usable responses" in fit.message
    assert np.isnan(fit.params).all()
    row = fit_all_subjects(df)
    assert len(row) == 1 and not row.success.iloc[0]


def test_constant_responses_do_not_crash():
    df = simulate_long(BayesianMock(), subjects=(1,), n_blocks=2)
    df["estimate"] = 0.5
    fit = fit_system_neglect(df)
    assert fit.n_obs == len(df)
    assert isinstance(fit.message, str)


@pytest.mark.parametrize("bad", ["period", "length"])
def test_bad_design_rejected(bad):
    df = simulate_long(BayesianMock(), subjects=(1,), n_blocks=1)
    df = df.iloc[:-1] if bad == "length" else df.assign(period=df.period.where(df.period < 10, 9))
    fit = fit_system_neglect(df)
    assert not fit.success and "bad design" in fit.message

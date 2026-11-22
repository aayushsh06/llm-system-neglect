import numpy as np
import pandas as pd
import pytest

from regime_shift import analysis as an
from regime_shift.bayes import forward_filter
from regime_shift.collection.backends import BayesianMock
from regime_shift.prompts import build_stateless_messages
from regime_shift.task import generate_subject_trials


def bayes_long(source="human", subjects=(1, 2), noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in subjects:
        for t in generate_subject_trials(s, 3, n_blocks=2):
            p = forward_filter(t.q, t.d, t.signals)
            for k in range(10):
                rows.append(
                    {
                        "source": source,
                        "subject": s,
                        "trial": t.trial,
                        "condition": t.condition,
                        "q": t.q,
                        "d": t.d,
                        "period": k + 1,
                        "signal": t.signals[k],
                        "estimate": float(np.clip(p[k] + noise * rng.normal(), 0, 1.0)),
                    }
                )
    return pd.DataFrame(rows)


def test_bayes_answers_give_zero_io_and_distance():
    df = bayes_long()
    io = an.index_of_overreaction(df)
    assert io.io.abs().max() < 1e-12
    dist = an.distance_from_bayes(df)
    assert dist.abs_dev.max() < 1e-12
    summary = an.grid_summary(io, "io")
    assert len(summary) == 9 and set(summary.columns) >= {"mean", "sem", "n", "t", "p"}
    mat = an.grid_matrix(summary, "human")
    assert mat.shape == (3, 3) and list(mat.index) == [1.5, 3.0, 9.0]


def test_io_convention_matches_archive_code():
    # two trials, hand-built: estimate jumps 0 -> 0.5 at period 2 then flat; Bayes flat at 0.2
    rows = []
    for k in range(10):
        rows.append(
            {
                "source": "x",
                "subject": 1,
                "trial": 1,
                "condition": 1,
                "q": 0.01,
                "d": 1.5,
                "period": k + 1,
                "signal": 1,
                "estimate": 0.0 if k == 0 else 0.5,
            }
        )
    df = pd.DataFrame(rows)
    df = an.add_bayes(df)
    io_incl = an.index_of_overreaction(df, include_first_period=True).io.iloc[0]
    io_excl = an.index_of_overreaction(df, include_first_period=False).io.iloc[0]
    d_bayes = np.diff(df.p_bayes.to_numpy())
    expected_excl = np.mean(np.r_[0.5, np.zeros(8)] - d_bayes)
    assert io_excl == pytest.approx(expected_excl)
    assert io_incl == pytest.approx(expected_excl * 9 / 10)


def test_deviation_correlation_identity_and_nan_accounting():
    h = bayes_long("human", noise=0.1, seed=1)
    m = h.copy()
    m["source"] = "model"
    m.loc[m.index[:5], "estimate"] = np.nan
    c = an.deviation_correlation(h, m)
    assert c["overall"].r.iloc[0] == pytest.approx(1.0)
    assert c["overall"].n_excluded_nan.iloc[0] == 5
    assert len(c["by_condition"]) == 9 and len(c["by_period"]) == 10
    assert c["by_subject_summary"].mean_r.iloc[0] == pytest.approx(1.0)


def test_deviation_correlation_rejects_unshared_sequences():
    h = bayes_long("human")
    m = bayes_long("model")
    m["signal"] = 1 - m["signal"]
    with pytest.raises(ValueError):
        an.deviation_correlation(h, m)


def test_parameter_table_and_slopes():
    fits = pd.DataFrame(
        {
            "source": ["m"] * 3,
            "subject": [1, 2, 3],
            "alpha_1": [3, 4, 2],
            "alpha_2": [1, 1.2, 0.8],
            "alpha_3": [0.6, 0.7, 0.5],
            "beta_1": [1.6, 1.8, 1.4],
            "beta_2": [0.8, 0.9, 0.7],
            "beta_3": [0.5, 0.6, 0.4],
            "success": [True, True, False],
        }
    )
    pt = an.parameter_table(fits)
    assert len(pt) == 6 and (pt.n_fits == 2).all() and (pt.n_failed == 1).all()
    assert pt[pt.name == "alpha_1"]["mean"].iloc[0] == pytest.approx(3.5)
    sl = an.neglect_slopes(fits)
    assert len(sl) == 2 and (sl.alpha_slope_q < 0).all() and (sl.beta_slope_logd < 0).all()
    ss = an.slope_summary(sl)
    assert set(ss.statistic) == {
        "alpha_slope_q",
        "beta_slope_logd",
        "alpha_low_minus_high",
        "beta_low_minus_high",
    }


def test_parse_failure_table():
    df = pd.DataFrame(
        {
            "source": ["m"] * 4,
            "mode": ["main"] * 4,
            "subject": [1, 1, 2, 2],
            "parse_status": ["ok", "empty", "ok_percent", "ambiguous"],
        }
    )
    pf = an.parse_failure_table(df)
    assert pf.usable.iloc[0] == 0.5 and pf.failed.iloc[0] == 0.5 and pf.n_calls.iloc[0] == 4
    assert pf.share_subjects_over_10pct_failed.iloc[0] == 1.0


def test_mock_bayesian_answers_match_filter():
    t = generate_subject_trials(1, 2, n_blocks=1)[0]
    msgs = build_stateless_messages(t, 6)
    text = BayesianMock().chat(msgs, temperature=0.7, seed=1, num_predict=8).text
    assert int(text) == int(np.clip(np.rint(forward_filter(t.q, t.d, t.signals)[5] * 100), 0, 99))


def test_accuracy_overall_pairs_models_with_humans():
    human = an.add_bayes(bayes_long("human", subjects=(1, 2, 3, 4, 5), noise=0.02, seed=1))
    good = an.add_bayes(bayes_long("good", subjects=(1, 2, 3, 4, 5), noise=0.02, seed=2))
    bad = an.add_bayes(bayes_long("bad", subjects=(1, 2, 3, 4, 5), noise=0.02, seed=3))
    bad["estimate"] = (bad.estimate + 0.3).clip(0, 1)
    dist = an.distance_from_bayes(pd.concat([human, good, bad], ignore_index=True))
    acc = an.accuracy_overall(dist, n_boot=500, seed=0).set_index("source")
    assert list(acc.index) == ["human", "bad", "good"]
    assert (
        acc.loc["human", "ci_low"]
        <= acc.loc["human", "mean_abs_dev"]
        <= acc.loc["human", "ci_high"]
    )
    assert np.isnan(acc.loc["human"].get("diff_from_human", np.nan))
    assert acc.loc["bad", "diff_from_human"] > 0.2 and bool(acc.loc["bad", "distinguishable"])
    assert abs(acc.loc["good", "diff_from_human"]) < 0.05 and not bool(
        acc.loc["good", "distinguishable"]
    )
    assert acc.loc["bad", "n_pairs"] == 5 and acc.loc["bad", "p_paired"] < 0.01


def test_fit_quality_separates_bayesian_and_ramp_sources():
    from regime_shift.fitting import fit_all_subjects

    human = an.add_bayes(bayes_long("human", subjects=(1, 2, 3), noise=0.02, seed=1))
    ramp = an.add_bayes(bayes_long("ramp", subjects=(1, 2, 3), noise=0.0, seed=2))
    ramp["estimate"] = ramp.period / 10.0
    both = pd.concat([human, ramp], ignore_index=True)
    fits = fit_all_subjects(both)
    fq = an.fit_quality(both, fits)
    s = fq["summary"].set_index("source")
    assert s.loc["human", "r2_neglect_mean"] > 0.9 and s.loc["human", "r2_bayes_mean"] > 0.9
    assert not bool(s.loc["human", "do_not_interpret"])
    assert s.loc["ramp", "r2_bayes_mean"] < s.loc["ramp", "r2_neglect_mean"]
    assert s.loc["ramp", "rmse_neglect_mean"] >= 0
    per = fq["per_subject"]
    assert set(per.source) == {"human", "ramp"} and len(per) == 6
    assert (per.rmse_bayes >= 0).all() and per.fit_success.all()

import numpy as np
import pandas as pd
import pytest

from regime_shift import analysis as an
from regime_shift import sanity
from regime_shift.fitting import PARAM_NAMES
from regime_shift.task import D_LEVELS, Q_LEVELS
from test_analysis import bayes_long  # noqa: I001  (tests/ is on sys.path under pytest)


def _human(subjects=(1, 2, 3)):
    return an.add_bayes(bayes_long("human", subjects=subjects, noise=0.02))


def _model(source="m [main, T0.7]", mode="main", ramp=False, seed=1, subjects=(1, 2, 3)):
    df = an.add_bayes(bayes_long(source, subjects=subjects, noise=0.02, seed=seed))
    if ramp:  # counts upward each period, blind to the balls
        df["estimate"] = df.period / 10.0
    df["model_tag"] = "m"
    df["mode"] = mode
    df["temperature"] = 0.7
    return df


def _fits(sources, alpha=(1, 1, 1), beta=(1, 1, 1), subjects=(1, 2, 3)):
    rows = []
    for src in sources:
        for s in subjects:
            row = {"source": src, "subject": s, "success": True}
            row.update(dict(zip(PARAM_NAMES, [*alpha, *beta], strict=True)))
            rows.append(row)
    return pd.DataFrame(rows)


def test_effective_parameters_scale_with_alpha_and_beta():
    eff = sanity.effective_parameters(_fits(["x"], alpha=(3, 1, 0.5), beta=(2, 1, 0.5)))
    a = eff[eff.quantity.str.startswith("effective switch")]["mean"].tolist()
    b = eff[eff.quantity.str.startswith("effective diag")]["mean"].tolist()
    assert a == pytest.approx([0.03, 0.05, 0.05])
    assert b == pytest.approx([2.25, 3.0, 3.0])


def test_signal_response_sign_for_bayes_and_zero_for_a_ramp():
    both = pd.concat([_human(), _model(ramp=True)], ignore_index=True)
    sr = sanity.signal_response(both)
    h = sr[(sr.source == "human") & (sr.d == "all")].set_index("after").mean_delta
    assert h["blue"] > 0 > h["red"]
    sep = sr[(sr.source == "human") & (sr.after == "red_minus_blue")]
    assert (sep.mean_delta < 0).all() and (sep.p < 0.05).all()
    m = sr[(sr.source == "m [main, T0.7]") & (sr.d == "all")].set_index("after").mean_delta
    assert m["blue"] == pytest.approx(0.1) and m["red"] == pytest.approx(0.1)
    assert m["red_minus_blue"] == pytest.approx(0.0, abs=1e-12)


def test_period_and_marginal_means_shapes():
    both = pd.concat([_human(), sanity.bayes_as_source(_human())], ignore_index=True)
    pm = sanity.period_means(both)
    assert set(pm.source) == {"human", "Bayes"}
    assert len(pm) == 2 * 9 * 10
    mm = sanity.marginal_means(both)
    assert len(mm) == 2 * 6
    assert mm[(mm.source == "Bayes") & (mm.factor == "q")]["mean"].is_monotonic_increasing


def test_bayes_source_is_the_posterior():
    human = _human()
    b = sanity.bayes_as_source(human)
    assert (b.source == "Bayes").all()
    np.testing.assert_allclose(b.estimate, human.p_bayes)


def test_ratio_anchor_share_detects_copying_the_prompt_number():
    copier = _model(source="c [main, T0.7]")
    copier["estimate"] = np.rint(100 / (1 + copier.d)) / 100  # always answers 40, 25 or 10
    qcopier = _model(source="s [main, T0.7]")
    qcopier["estimate"] = qcopier.q  # always answers 1, 5 or 10
    human = _human()
    share = sanity.ratio_anchor_share(
        pd.concat([human, sanity.bayes_as_source(human), copier, qcopier], ignore_index=True)
    )
    c = share[(share.source == "c [main, T0.7]") & (share.periods == "all periods")]
    assert len(c) == 9 and c.condition.tolist() == list(range(1, 10))
    assert c.ratio_number.tolist() == [40, 25, 10, 40, 25, 10, 40, 25, 10]
    assert c.prior_number.tolist() == [1, 1, 1, 5, 5, 5, 10, 10, 10]
    assert c.ambiguous.tolist() == [False] * 8 + [True]
    assert c[~c.ambiguous].ratio_echo.tolist() == pytest.approx([1.0] * 8)
    assert c[c.ambiguous].both_numbers.tolist() == pytest.approx([1.0])
    assert (c.prior_anchoring.fillna(0) == 0).all() and (c.majority_number_share == 0).all()
    s = share[(share.source == "s [main, T0.7]") & (share.periods == "period 1")]
    s = s[~s.ambiguous & (s.detectable_prior_anchoring > 0)]
    assert len(s) >= 4 and s.prior_anchoring.tolist() == pytest.approx([1.0] * len(s))
    assert s.ratio_echo.fillna(0).tolist() == pytest.approx([0.0] * len(s))
    h = share[(share.source == "human") & (share.periods == "period 1")]
    assert len(h) == 9 and (h.ratio_echo.fillna(0) < 0.5).all()
    b = share[(share.source == "Bayes") & (share.periods == "period 1")]
    assert (b.prior_anchoring.fillna(0) == 0).all()  # Bayes cannot "copy" by construction
    assert (b.detectable_prior_anchoring < 1).any()  # some period-1 Bayes answers equal q x 100
    m = share.dropna(subset=["excess_ratio_echo", "ratio_echo"])
    assert (m.excess_ratio_echo <= m.ratio_echo + 1e-12).all()
    assert (share[share.source == "human"].excess_prior_anchoring.abs().fillna(0) < 1e-12).all()
    t = sanity.verdict_tables(_human(), pd.concat([copier, qcopier], ignore_index=True))
    assert sanity.verdict_words("c [main, T0.7]", **t)["copy"] == "yes"
    v = sanity.verdict_words("s [main, T0.7]", **t)
    assert v["copy"] == "yes" and v["prior_anchoring_max"] == 1.0
    assert v["excess_prior_anchoring"] > 0.5 and v["ratio_echo_max"] == 0.0


def test_last_period_jump_detects_an_end_of_trial_effect():
    jumper = _model(source="j [main, T0.7]")
    jumper["estimate"] = jumper.period / 20.0 + (jumper.period == 10) * 0.3
    lj = sanity.last_period_jump(pd.concat([_human(), jumper], ignore_index=True)).set_index(
        "source"
    )
    assert lj.loc["j [main, T0.7]", "jump"] == pytest.approx(0.3)
    assert lj.loc["j [main, T0.7]", "step_periods_2_to_9"] == pytest.approx(0.05)
    assert abs(lj.loc["human", "jump"]) < 0.1


def test_run_sanity_checks_writes_the_check_tables(tmp_path):
    human = _human()
    model = pd.concat([_model(), _model("m [stateless, T0.7]", "stateless", seed=2)])
    fits = pd.concat([_fits(["human"]), _fits(["m [main, T0.7]", "m [stateless, T0.7]"])])
    t = sanity.run_sanity_checks(human, model, fits, tmp_path)
    for name in (
        "effective_parameters",
        "signal_response",
        "period_means",
        "marginal_means",
        "raw_answer_counts",
        "ratio_anchor_share",
        "last_period_jump",
        "copying_tolerance_comparison",
        "calibration",
        "accuracy_by_state",
    ):
        assert name in t and (tmp_path / f"sanity_{name}.csv").exists()
    assert set(t["signal_response"].source) >= {"human", "Bayes", "m [main, T0.7]"}


def test_run_sanity_checks_without_models(tmp_path):
    human = _human()
    t = sanity.run_sanity_checks(human, human.iloc[0:0], _fits(["human"]), tmp_path)
    assert set(t["signal_response"].source) == {"human", "Bayes"}


def test_effective_parameters_of_bayes_are_q_and_d():
    eff = sanity.effective_parameters(_fits(["x"]))
    a = eff[eff.quantity.str.startswith("effective switch")]
    b = eff[eff.quantity.str.startswith("effective diag")]
    assert a["mean"].tolist() == pytest.approx(list(Q_LEVELS))
    assert b["mean"].tolist() == pytest.approx(list(D_LEVELS))


def test_verdict_words_for_bayes_like_and_ramp_sources():
    subjects = tuple(range(1, 13))  # enough period-1 answers per cell for the copying shares
    human = _human(subjects)
    both = pd.concat(
        [
            _model(subjects=subjects),
            _model("r [main, T0.7]", ramp=True, seed=3, subjects=subjects),
        ],
        ignore_index=True,
    )
    t = sanity.verdict_tables(human, both)
    v = sanity.verdict_words("m [main, T0.7]", **t)
    assert v["copy"] == "no" and v["q"] == "yes" and v["red"] == "yes"
    r = sanity.verdict_words("r [main, T0.7]", **t)
    assert r["red"] == "no" and r["d"] == "flat"


def test_copying_tolerance_calibration_and_accuracy_by_state():
    human = _human(tuple(range(1, 7)))
    both = pd.concat(
        [human, sanity.bayes_as_source(human), _model(subjects=tuple(range(1, 7)))],
        ignore_index=True,
    )
    cmp = sanity.copying_tolerance_comparison(both).set_index("source")
    assert "prior anchoring max | old rule" in cmp and "prior anchoring max | new rule" in cmp
    # the new rule can only count fewer detectable answers than the old one
    assert (
        cmp["detectable prior anchoring mean | new rule"]
        <= cmp["detectable prior anchoring mean | old rule"] + 1e-12
    ).all()
    assert cmp.loc["Bayes"].filter(like="max |").fillna(0).eq(0).all()
    cal = sanity.calibration(both)
    b = cal[cal.source == "Bayes"]
    assert np.allclose(b.mean_stated, b.mean_bayes) and set(cal.bin).issubset(
        {"[0, 0.05)", "[0.05, 0.2)", "[0.2, 0.5)", "[0.5, 0.8)", "[0.8, 1)"}
    )
    assert cal.groupby("source").n.sum().nunique() == 1  # every source has every observation
    st = sanity.accuracy_by_posterior_state(both)
    o = st["overall"]
    for src, g in o.groupby("source"):
        assert g.share_of_periods.sum() == pytest.approx(1.0)
        total = both[(both.source == src)].pipe(lambda x: (x.estimate - x.p_bayes).abs().mean())
        assert g.contribution_to_overall.sum() == pytest.approx(total)
    assert (o[o.source == "Bayes"].mean_abs_dev == 0).all()
    pc = st["per_condition"]
    assert set(pc.state) == {"posterior < 0.2", "posterior >= 0.2"}

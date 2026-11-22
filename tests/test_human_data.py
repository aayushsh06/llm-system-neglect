import numpy as np

from regime_shift.human_data import decode_tables, human_trials


def test_decode_shapes(data_dir):
    tables = decode_tables(data_dir / "data_allsubj_expt1.mat")
    assert len(tables) == 30
    assert all(t.shape == (990, 38) for t in tables)
    assert [int(t.subNO.iloc[0]) for t in tables] == list(range(1, 31))


def test_long_table_facts(human_all):
    assert human_all.shape[0] == 29700
    assert human_all.estimate.isna().sum() == 263
    assert human_all.groupby("subject").trial.nunique().eq(99).all()
    assert (human_all[human_all.session == 0].groupby(["subject", "condition"]).size() == 80).all()
    assert (human_all[human_all.session == 1].groupby(["subject", "condition"]).size() == 30).all()
    assert human_all.estimate.max() <= 0.99 and human_all.estimate.min() >= 0
    assert set(human_all.q.unique()) == {0.01, 0.05, 0.1} and set(human_all.d.unique()) == {
        1.5,
        3.0,
        9.0,
    }


def test_forward_filter_matches_archive_posterior(human_all):
    np.testing.assert_allclose(human_all.p_bayes, human_all.p_bayes_archive, atol=1e-12)


def test_sequences_are_subject_specific(human_all):
    seqs = {
        s: tuple(g.sort_values(["trial", "period"]).signal) for s, g in human_all.groupby("subject")
    }
    assert len(set(seqs.values())) == 30
    trials = human_trials(human_all[human_all.session == 0])
    assert len(trials) == 30 and all(len(v) == 72 for v in trials.values())
    assert trials[1][0].signals == tuple(
        human_all[(human_all.subject == 1) & (human_all.trial == 1)].signal
    )

"""A one-minute tour on real data: one posterior, one fit, one model's two sensitivities."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regime_shift.analysis import add_bayes  # noqa: E402
from regime_shift.bayes import forward_filter  # noqa: E402
from regime_shift.fitting import fit_all_subjects  # noqa: E402
from regime_shift.human_data import load_human_long  # noqa: E402
from regime_shift.results import complete_trials_only, load_results  # noqa: E402
from regime_shift.task import D_LEVELS, Q_LEVELS  # noqa: E402

LQ, LD = np.log(Q_LEVELS), np.log(D_LEVELS)


def sensitivities(fit) -> tuple[float, float]:
    """s_q: slope of log(alpha q) on log q; s_d: slope of beta log d on log d (Bayes = 1, 1)."""
    alpha = np.array([fit.alpha_1, fit.alpha_2, fit.alpha_3])
    beta = np.array([fit.beta_1, fit.beta_2, fit.beta_3])
    return float(np.polyfit(LQ, np.log(alpha * Q_LEVELS), 1)[0]), float(
        np.polyfit(LD, beta * LD, 1)[0]
    )


# the human data: 30 subjects, 72 behavioural trials each, with the Bayesian posterior added
humans = add_bayes(load_human_long(ROOT / "data/human/data_allsubj_expt1.mat", "behavioral"))
humans["source"] = "human"

# one trial: the balls, the posterior after each ball, and what subject 1 answered
trial = humans[(humans.subject == 1) & (humans.trial == 1)].sort_values("period")
posterior = forward_filter(trial.q.iloc[0], trial.d.iloc[0], trial.signal.tolist())
print(f"subject 1, trial 1: q = {trial.q.iloc[0]:g}, d = {trial.d.iloc[0]:g}")
print("  balls (1 = blue):  ", trial.signal.tolist())
print("  Bayesian posterior:", np.round(posterior, 3).tolist())
print("  stated probability:", np.round(trial.estimate.to_numpy(), 2).tolist())

# the system-neglect fit for subject 1 and the two sensitivities it implies
fit = fit_all_subjects(humans[humans.subject == 1]).iloc[0]
print(f"subject 1 fit: alpha = {fit.alpha_1:.2f}, {fit.alpha_2:.2f}, {fit.alpha_3:.2f};", end=" ")
print(f"beta = {fit.beta_1:.2f}, {fit.beta_2:.2f}, {fit.beta_3:.2f}")
print("  sensitivities (s_q, s_d) = {:.2f}, {:.2f}".format(*sensitivities(fit)))

# one model's result file: its mean sensitivities and its distance from the posterior
model_file = ROOT / "results" / "claude-sonnet-5-5__main__T1__behavioral__number-only.jsonl"
model = add_bayes(complete_trials_only(load_results(model_file))[0])
pairs = np.array([sensitivities(r) for r in fit_all_subjects(model).itertuples()])
# distance as in the paper: each subject's mean over the nine condition means of
# |stated - posterior|, then the mean over subjects (Table 2's number for this model)
abs_dev = (model.estimate - model.p_bayes).abs()
distance = abs_dev.groupby([model.subject, model.condition]).mean().groupby(level=0).mean().mean()
print(f"{model_file.name}: {model.subject.nunique()} subjects")
print(f"  mean s_q = {pairs[:, 0].mean():.2f}, mean s_d = {pairs[:, 1].mean():.2f},", end=" ")
print(f"distance from the posterior = {distance:.3f}")

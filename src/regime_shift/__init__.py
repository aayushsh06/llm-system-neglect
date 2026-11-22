"""Regime-shift detection replication on language models.

The task and its mathematics live in :mod:`task` and :mod:`bayes`; the human
data loader in :mod:`human_data`; prompt construction and reply parsing in
:mod:`prompts` and :mod:`parsing`. Model access goes through :mod:`backends`
(Ollama and two mock observers) and :mod:`anthropic_backend`; runs are
orchestrated by :mod:`runner` and, for the Claude models, :mod:`ladder`.
Fitting and analysis are in :mod:`fitting`, :mod:`analysis`, :mod:`sanity`
and :mod:`validation`.
"""

__version__ = "0.1.0"

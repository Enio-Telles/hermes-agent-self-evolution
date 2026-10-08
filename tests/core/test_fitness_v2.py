"""Fitness v2: semantic feedback drives GEPA instead of keyword overlap."""

import dspy
import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import FitnessScore, LLMJudge, make_skill_metric


def test_judge_metric_returns_score_and_explanatory_feedback(monkeypatch):
    calls = []

    def fake_score(self, **kwargs):
        calls.append(kwargs)
        return FitnessScore(
            correctness=0.8,
            procedure_following=0.7,
            conciseness=0.9,
            feedback="Explain which steps were verified.",
        )

    monkeypatch.setattr(LLMJudge, "score", fake_score)
    metric = make_skill_metric(EvolutionConfig(hermes_agent_path=None), "# Skill", mode="judge")
    result = metric(
        dspy.Example(task_input="Fix bug", expected_behavior="Tests pass"),
        dspy.Prediction(output="Fixed the bug"),
        None, "predictor", [],
    )
    assert result.score == pytest.approx(0.79)
    assert "verified" in result.feedback
    assert calls[0]["skill_text"] == "# Skill"


def test_empty_prediction_is_rejected_without_external_judge(monkeypatch):
    monkeypatch.setattr(LLMJudge, "score", lambda *a, **kw: pytest.fail("judge must not run"))
    metric = make_skill_metric(EvolutionConfig(hermes_agent_path=None), "baseline", mode="judge")
    result = metric(
        dspy.Example(task_input="Task", expected_behavior="Good answer"),
        dspy.Prediction(output=""),
    )
    assert result.score == 0
    assert "empty" in result.feedback.lower()


def test_judge_failure_is_not_silently_awarded_neutral_score(monkeypatch):
    def broken(self, **kwargs):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(LLMJudge, "score", broken)
    metric = make_skill_metric(EvolutionConfig(hermes_agent_path=None), "baseline", mode="judge")
    with pytest.raises(RuntimeError, match="judge unavailable"):
        metric(
            dspy.Example(task_input="Task", expected_behavior="Good answer"),
            dspy.Prediction(output="Answer"),
        )


def test_keyword_mode_is_explicit_exploration_only():
    metric = make_skill_metric(EvolutionConfig(hermes_agent_path=None), "baseline", mode="heuristic")
    result = metric(
        dspy.Example(task_input="Task", expected_behavior="alpha beta"),
        dspy.Prediction(output="alpha beta"),
    )
    assert 0 <= result.score <= 1
    assert "heuristic" in result.feedback.lower()


def test_unknown_metric_mode_rejected():
    with pytest.raises(ValueError, match="metric mode"):
        make_skill_metric(EvolutionConfig(hermes_agent_path=None), "baseline", mode="unknown")

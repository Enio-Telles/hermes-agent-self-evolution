"""Regression tests for model data-egress consent and evolved-skill judge context."""

import dspy
import pytest

from evolution.core.config import EvolutionConfig, require_approved_model_egress
from evolution.core.fitness import FitnessScore, LLMJudge, make_skill_metric
from evolution.skills.skill_module import SkillModule
from evolution.skills.evolve_skill import evolve


def test_remote_endpoint_requires_explicit_consent(monkeypatch):
    monkeypatch.setenv("HERMES_EVOLVE_API_BASE", "https://model.example.com/v1")
    with pytest.raises(ValueError, match="remote model"):
        require_approved_model_egress(allow_remote_data=False)
    require_approved_model_egress(allow_remote_data=True)


def test_implicit_provider_requires_explicit_consent(monkeypatch):
    monkeypatch.delenv("HERMES_EVOLVE_API_BASE", raising=False)
    with pytest.raises(ValueError, match="remote model"):
        require_approved_model_egress(allow_remote_data=False)


@pytest.mark.parametrize("endpoint", [
    "http://127.0.0.1:8000/v1",
    "http://localhost:8000/v1",
    "http://[::1]:8000/v1",
])
def test_loopback_endpoint_does_not_require_remote_consent(monkeypatch, endpoint):
    monkeypatch.setenv("HERMES_EVOLVE_API_BASE", endpoint)
    require_approved_model_egress(allow_remote_data=False)


@pytest.mark.parametrize("endpoint", [
    "http://localhost.attacker.example/v1",
    "https://model.example.com/v1",
    "file:///tmp/model",
    "http://127.0.0.1.attacker.example/v1",
    "http://user:pass@localhost:8000/v1",
])
def test_unsafe_or_remote_endpoints_blocked_without_consent(monkeypatch, endpoint):
    monkeypatch.setenv("HERMES_EVOLVE_API_BASE", endpoint)
    with pytest.raises(ValueError):
        require_approved_model_egress(allow_remote_data=False)


def test_evolve_checks_egress_policy_before_building_dataset(monkeypatch):
    monkeypatch.delenv("HERMES_EVOLVE_API_BASE", raising=False)
    with pytest.raises(ValueError, match="remote model"):
        evolve(skill_name="some-skill", hermes_repo="/this/does/not/exist")


def test_skill_prediction_exposes_active_evolved_instructions(monkeypatch):
    module = SkillModule("# ORIGINAL")
    monkeypatch.setattr(
        module.predictor, "forward",
        lambda **kwargs: dspy.Prediction(output="done"),
    )
    first = module.forward(task_input="task")
    assert first.skill_text == "# ORIGINAL"

    for _, predictor in module.predictor.named_predictors():
        predictor.signature = predictor.signature.with_instructions("# EVOLVED")

    second = module.forward(task_input="task")
    assert second.skill_text == "# EVOLVED"


def test_judge_receives_candidate_skill_instructions(monkeypatch):
    used = []

    def score(self, **kwargs):
        used.append(kwargs["skill_text"])
        return FitnessScore(
            correctness=1.0, procedure_following=1.0,
            conciseness=1.0, feedback="OK",
        )

    monkeypatch.setattr(LLMJudge, "score", score)
    metric = make_skill_metric(
        EvolutionConfig(hermes_agent_path=None), "# BASELINE", mode="judge",
    )
    example = dspy.Example(task_input="task", expected_behavior="rubric")
    metric(example, dspy.Prediction(output="baseline", skill_text="# BASELINE"))
    metric(example, dspy.Prediction(output="candidate", skill_text="# EVOLVED"))
    assert used == ["# BASELINE", "# EVOLVED"]

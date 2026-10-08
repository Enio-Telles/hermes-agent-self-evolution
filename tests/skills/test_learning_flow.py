"""End-to-end governance outcomes without external model calls."""

import dspy
import pytest

from evolution.core.constraints import ConstraintResult, ConstraintValidator
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.learning_registry import LearningRegistry
from evolution.skills import evolve_skill
from evolution.skills.skill_module import SkillModule


@pytest.mark.parametrize(
    "run_tests,test_success,expected_reason",
    [
        (True, True, "improvement_verified"),
        (True, False, "candidate_tests_failed"),
        (False, True, "tests_unverified"),
    ],
)
def test_skill_attempt_tracks_governance_and_preserves_original(
    monkeypatch, tmp_path, run_tests, test_success, expected_reason
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_EVOLVE_API_BASE", "http://127.0.0.1:8090/v1")
    repo = tmp_path / "hermes-agent"
    skill = repo / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    original = (
        "---\nname: demo\ndescription: Demo skill\n---\n"
        "# Procedure\nFollow steps and validate every operation.\n"
    )
    skill.write_text(original)
    examples = [
        EvalExample(task_input="task", expected_behavior="correct steps")
        for _ in range(3)
    ]
    dataset = EvalDataset(train=examples[:1], val=examples[1:2], holdout=examples[2:])
    monkeypatch.setattr(evolve_skill.GoldenDatasetLoader, "load", lambda _: dataset)
    monkeypatch.setattr(evolve_skill, "make_lm", lambda _: object())
    monkeypatch.setattr(evolve_skill.dspy, "configure", lambda **kw: None)

    def fake_forward(self, task_input):
        return dspy.Prediction(output="candidate" if "every result" in self.skill_text else "baseline")

    monkeypatch.setattr(SkillModule, "forward", fake_forward)
    monkeypatch.setattr(
        evolve_skill,
        "make_skill_metric",
        lambda *a, **k: lambda gold, pred, **kw: dspy.Prediction(
            score=0.8 if pred.output == "candidate" else 0.4, feedback="Evaluated"
        ),
    )

    class Optimizer:
        def compile(self, baseline, trainset, valset):
            return SkillModule("# Procedure\nFollow steps and validate every result.")

    monkeypatch.setattr(evolve_skill.dspy, "GEPA", lambda **kw: Optimizer())
    monkeypatch.setattr(
        ConstraintValidator, "run_test_suite",
        lambda *a, **kw: ConstraintResult(
            test_success, "test_suite", "candidate checked"
        ),
    )

    evolve_skill.evolve(
        skill_name="demo", eval_source="golden", dataset_path=str(tmp_path),
        hermes_repo=str(repo), run_tests=run_tests, metric_mode="judge",
    )
    records = LearningRegistry(tmp_path / "output" / "skill_learning.sqlite3").history("demo")
    assert len(records) == 1
    assert records[0]["reason"] == expected_reason
    assert records[0]["decision"] == (
        "review_required" if expected_reason == "improvement_verified" else "rejected"
    )
    assert skill.read_text() == original

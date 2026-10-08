"""Persistent, metadata-only record of skill evolution experiments."""

import sqlite3

import pytest

from evolution.core.learning_registry import LearningRegistry, decide_candidate


def test_governance_requires_semantic_improvement_and_tests():
    approved = decide_candidate(0.50, 0.56, constraints_passed=True, tests_passed=True, metric_mode="judge")
    assert approved == ("review_required", "improvement_verified")

    assert decide_candidate(
        0.50, 0.56, constraints_passed=True, tests_passed=False, metric_mode="judge"
    )[0] == "rejected"
    assert decide_candidate(
        0.50, 0.56, constraints_passed=True, tests_passed=True, metric_mode="heuristic"
    )[0] == "rejected"
    assert decide_candidate(
        0.50, 0.49, constraints_passed=True, tests_passed=True, metric_mode="judge"
    )[0] == "rejected"


def test_small_or_invalid_improvements_cannot_be_approved():
    assert decide_candidate(0.50, 0.51, True, True, "judge")[0] == "rejected"
    for score in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError):
            decide_candidate(0.5, score, True, True, "judge")


def test_learning_registry_is_durable_metadata_only(tmp_path):
    path = tmp_path / "learning.sqlite3"
    first = LearningRegistry(path)
    uid = first.record(
        skill_name="demo",
        baseline_text="TOP_SECRET_BASELINE",
        candidate_text="TOP_SECRET_CANDIDATE",
        dataset_fingerprint="abc123",
        baseline_score=0.4,
        candidate_score=0.6,
        decision="review_required",
        reason="improvement_verified",
        metric_mode="judge",
        optimizer_model="model-a",
        eval_model="model-b",
        tests_passed=True,
    )
    records = LearningRegistry(path).history("demo")
    assert len(records) == 1
    assert records[0]["run_id"] == uid
    assert records[0]["candidate_score"] == pytest.approx(0.6)
    assert records[0]["baseline_sha256"] != records[0]["candidate_sha256"]
    assert b"TOP_SECRET" not in path.read_bytes()


def test_registry_rejects_unrecognized_decision_and_invalid_score(tmp_path):
    registry = LearningRegistry(tmp_path / "evolution.sqlite3")
    values = dict(
        skill_name="demo", baseline_text="baseline", candidate_text="candidate",
        dataset_fingerprint="abc", baseline_score=0.4, candidate_score=0.6,
        decision="ship_to_prod", reason="bad", metric_mode="judge",
        optimizer_model="a", eval_model="b", tests_passed=True,
    )
    with pytest.raises(ValueError):
        registry.record(**values)
    values["decision"] = "rejected"
    values["candidate_score"] = float("nan")
    with pytest.raises(ValueError):
        registry.record(**values)

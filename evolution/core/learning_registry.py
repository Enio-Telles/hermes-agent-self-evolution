"""SQLite metadata-only ledger of skill evolution experiments.

Do not put prompts, session transcripts, model responses or credentials here.
The record is intentionally a proposal/audit log, not a deployment authority.
"""

import hashlib
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


DECISIONS = {"review_required", "rejected"}
MODES = {"judge", "heuristic"}


def _validate_score(value):
    if value is None:
        return
    if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Scores must be finite values between 0 and 1")


def decide_candidate(
    baseline_score: float,
    candidate_score: float,
    constraints_passed: bool,
    tests_passed: bool,
    metric_mode: str,
    min_improvement: float = 0.02,
) -> tuple[str, str]:
    """Review eligibility only, not permission to publish or deploy."""
    _validate_score(baseline_score)
    _validate_score(candidate_score)
    if not math.isfinite(min_improvement) or min_improvement <= 0:
        raise ValueError("min_improvement must be finite and positive")
    if metric_mode not in MODES:
        raise ValueError("Unknown metric mode")
    if not constraints_passed:
        return "rejected", "constraints_failed"
    if tests_passed is not True:
        return "rejected", "tests_unverified"
    if metric_mode != "judge":
        return "rejected", "semantic_evaluation_required"
    if candidate_score - baseline_score < min_improvement:
        return "rejected", "insufficient_holdout_improvement"
    return "review_required", "improvement_verified"


class LearningRegistry:
    """Durable metadata-only records; SQLite transactions handle concurrent writers."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS skill_runs (
                    run_id TEXT PRIMARY KEY,
                    recorded_at TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    baseline_sha256 TEXT NOT NULL,
                    candidate_sha256 TEXT NOT NULL,
                    dataset_fingerprint TEXT NOT NULL,
                    baseline_score REAL,
                    candidate_score REAL,
                    decision TEXT NOT NULL CHECK (decision IN ('rejected', 'review_required')),
                    reason TEXT NOT NULL,
                    metric_mode TEXT NOT NULL,
                    optimizer_model TEXT NOT NULL,
                    eval_model TEXT NOT NULL,
                    tests_passed INTEGER
                )
            """)

    def record(
        self, *, skill_name: str, baseline_text: str, candidate_text: str,
        dataset_fingerprint: str, baseline_score: float | None,
        candidate_score: float | None, decision: str, reason: str,
        metric_mode: str, optimizer_model: str, eval_model: str,
        tests_passed: bool | None,
    ) -> str:
        _validate_score(baseline_score)
        _validate_score(candidate_score)
        if decision not in DECISIONS or metric_mode not in MODES:
            raise ValueError("Invalid decision or metric mode")
        if not skill_name or not reason or not dataset_fingerprint:
            raise ValueError("Required audit metadata is missing")
        if tests_passed not in (None, True, False):
            raise ValueError("Invalid test result")
        run_id = uuid.uuid4().hex
        values = (
            run_id, datetime.now(timezone.utc).isoformat(), skill_name,
            hashlib.sha256(baseline_text.encode("utf-8")).hexdigest(),
            hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
            dataset_fingerprint, baseline_score, candidate_score,
            decision, reason, metric_mode, optimizer_model, eval_model,
            None if tests_passed is None else int(tests_passed),
        )
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute(
                """INSERT INTO skill_runs (
                    run_id, recorded_at, skill_name, baseline_sha256,
                    candidate_sha256, dataset_fingerprint, baseline_score,
                    candidate_score, decision, reason, metric_mode,
                    optimizer_model, eval_model, tests_passed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
        return run_id

    def history(self, skill_name: str) -> list[dict]:
        with sqlite3.connect(self.path, timeout=30) as db:
            db.row_factory = sqlite3.Row
            results = db.execute(
                "SELECT * FROM skill_runs WHERE skill_name=? ORDER BY recorded_at, run_id",
                (skill_name,),
            ).fetchall()
        return [dict(row) for row in results]

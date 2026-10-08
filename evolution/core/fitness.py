"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import dspy
import math
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig, make_lm


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        lm = make_lm(self.config.eval_model)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str = "",
    pred_trace=None,
) -> float:
    """DSPy-compatible metric function for skill optimization.

    This is what gets passed to dspy.GEPA(metric=...).
    Returns a float 0-1 score.
    Signature must accept five args: (gold, pred, trace, pred_name, pred_trace).
    """
    # The prediction should have an 'output' field with the agent's response
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""

    if not agent_output.strip():
        return 0.0

    # Quick heuristic scoring (for speed during optimization)
    # ponytail: keyword-overlap heuristic ignores stopwords, so GEPA can
    # game it with keyword stuffing. Upgrade path: LLMJudge.score (above)
    # as the GEPA metric when eval budget allows — reflections are only
    # as good as this signal.
    score = 0.5  # Base score for non-empty output

    # Check if key phrases from expected behavior appear
    expected_lower = expected.lower()
    output_lower = agent_output.lower()

    # Simple keyword overlap as a fast proxy
    expected_words = set(expected_lower.split())
    output_words = set(output_lower.split())
    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score = 0.3 + (0.7 * overlap)

    return min(1.0, max(0.0, score))


def make_skill_metric(config: EvolutionConfig, skill_text: str, mode: str = "judge"):
    """Build GEPA's score-with-feedback metric; judge is the default.

    Heuristic mode is for cost-limited debugging, never for promotion.
    """
    if mode not in {"judge", "heuristic"}:
        raise ValueError("Unknown metric mode: " + str(mode))

    judge = LLMJudge(config) if mode == "judge" else None

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        output = str(getattr(prediction, "output", "") or "")
        if not output.strip():
            return dspy.Prediction(score=0.0, feedback="Empty output: task was not completed.")

        if mode == "heuristic":
            score = skill_fitness_metric(example, prediction, trace, pred_name, pred_trace)
            return dspy.Prediction(
                score=score,
                feedback="Heuristic keyword overlap only: correctness was not semantically verified.",
            )

        result = judge.score(
            task_input=str(getattr(example, "task_input", "")),
            expected_behavior=str(getattr(example, "expected_behavior", "")),
            agent_output=output,
            skill_text=str(getattr(prediction, "skill_text", skill_text)),
        )
        feedback = result.feedback.strip() or (
            "Review correctness, procedure compliance and conciseness."
        )
        return dspy.Prediction(score=result.composite, feedback=feedback)

    return metric


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    try:
        score = float(str(value).strip())
    except (ValueError, TypeError) as error:
        raise ValueError("Judge returned a non-numeric score") from error
    if not math.isfinite(score):
        raise ValueError("Judge returned a non-finite score")
    return min(1.0, max(0.0, score))

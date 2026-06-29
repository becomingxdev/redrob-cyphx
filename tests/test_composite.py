"""Tests for the composite scoring engine."""

from __future__ import annotations

from src.scoring import (
    FinalScore,
    HoneypotResult,
    PenaltyResult,
    ScoreResult,
)
from src.scoring.composite import compose_score


def _make_score(score: float) -> ScoreResult:
    return ScoreResult(score=score, reasons=["test"], metadata={})


def _make_penalty(score: float) -> PenaltyResult:
    return PenaltyResult(penalty_score=score, reasons=["test penalty"], metadata={"breakdown": {}})


def _make_honeypot(score: float) -> HoneypotResult:
    return HoneypotResult(suspicion_score=score, reasons=["test suspicion"], metadata={"breakdown": {}})


class TestCompositeScore:
    """Tests for compose_score()."""

    def test_returns_final_score(self):
        """Result must be a FinalScore."""
        result = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(70.0),
            experience_score=_make_score(60.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
        )
        assert isinstance(result, FinalScore)
        assert isinstance(result.score, float)
        assert isinstance(result.candidate_id, str)
        assert isinstance(result.component_scores, dict)
        assert isinstance(result.reasons, list)

    def test_score_in_valid_range(self):
        """Final score must be in [0, 100]."""
        result = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(100.0),
            skill_score=_make_score(100.0),
            experience_score=_make_score(100.0),
            education_score=_make_score(100.0),
            behavior_score=_make_score(100.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 1.0},
            consistency_result={"consistency_score": 1.0},
            honeypot=_make_honeypot(0.0),
        )
        assert 0.0 <= result.score <= 100.0

    def test_candidate_id_preserved(self):
        """Candidate ID must be preserved in the result."""
        result = compose_score(
            candidate_id="CAND_1234567",
            title_score=_make_score(50.0),
            skill_score=_make_score(50.0),
            experience_score=_make_score(50.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(10.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.8},
            consistency_result={"consistency_score": 0.8},
            honeypot=_make_honeypot(0.0),
        )
        assert result.candidate_id == "CAND_1234567"

    def test_component_scores_recorded(self):
        """All component scores should be in the component_scores dict."""
        result = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(70.0),
            experience_score=_make_score(60.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
        )
        assert result.component_scores["title"] == 80.0
        assert result.component_scores["skills"] == 70.0
        assert result.component_scores["experience"] == 60.0
        assert result.component_scores["education"] == 50.0
        assert result.component_scores["behavior"] == 20.0
        assert result.component_scores["confidence"] == 0.9
        assert result.component_scores["consistency"] == 0.9

    def test_penalty_reduces_score(self):
        """Non-zero penalty should reduce the final score."""
        weights = {k: v for k, v in {
            "title": 0.25, "skills": 0.30, "experience": 0.20,
            "education": 0.10, "behavior": 0.05, "confidence": 0.05,
            "consistency": 0.05, "penalty_multiplier": 1.0, "honeypot_multiplier": 1.0,
        }.items()}

        no_penalty = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(80.0),
            experience_score=_make_score(80.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
            weights=weights,
        )
        with_penalty = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(80.0),
            experience_score=_make_score(80.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(20.0),
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
            weights=weights,
        )
        assert with_penalty.score < no_penalty.score

    def test_honeypot_reduces_score(self):
        """Non-zero honeypot suspicion should reduce the final score."""
        weights = {k: v for k, v in {
            "title": 0.25, "skills": 0.30, "experience": 0.20,
            "education": 0.10, "behavior": 0.05, "confidence": 0.05,
            "consistency": 0.05, "penalty_multiplier": 1.0, "honeypot_multiplier": 1.0,
        }.items()}

        no_suspicion = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(80.0),
            experience_score=_make_score(80.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
            weights=weights,
        )
        with_suspicion = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(80.0),
            experience_score=_make_score(80.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(50.0),
            weights=weights,
        )
        assert with_suspicion.score < no_suspicion.score

    def test_low_confidence_reduces_score(self):
        """Low confidence should reduce the final score."""
        weights = {k: v for k, v in {
            "title": 0.25, "skills": 0.30, "experience": 0.20,
            "education": 0.10, "behavior": 0.05, "confidence": 0.05,
            "consistency": 0.05, "penalty_multiplier": 1.0, "honeypot_multiplier": 1.0,
        }.items()}

        high_conf = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(80.0),
            experience_score=_make_score(80.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 1.0},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
            weights=weights,
        )
        low_conf = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(80.0),
            experience_score=_make_score(80.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.1},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
            weights=weights,
        )
        assert low_conf.score < high_conf.score

    def test_zero_scores_produces_zero(self):
        """All zero component scores should produce near-zero final score."""
        result = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(0.0),
            skill_score=_make_score(0.0),
            experience_score=_make_score(0.0),
            education_score=_make_score(0.0),
            behavior_score=_make_score(0.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.0},
            consistency_result={"consistency_score": 0.0},
            honeypot=_make_honeypot(0.0),
        )
        assert result.score < 1.0

    def test_deterministic(self):
        """Same inputs must produce the same output."""
        kwargs = {
            "candidate_id": "CAND_0000001",
            "title_score": _make_score(75.0),
            "skill_score": _make_score(65.0),
            "experience_score": _make_score(55.0),
            "education_score": _make_score(45.0),
            "behavior_score": _make_score(15.0),
            "penalty": _make_penalty(3.0),
            "confidence_result": {"confidence": 0.8},
            "consistency_result": {"consistency_score": 0.85},
            "honeypot": _make_honeypot(2.0),
        }
        result1 = compose_score(**kwargs)
        result2 = compose_score(**kwargs)
        assert result1.score == result2.score
        assert result1.reasons == result2.reasons

    def test_no_ranking_or_sorting(self):
        """Composite engine should not perform any ranking."""
        result = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(70.0),
            experience_score=_make_score(60.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
        )
        # No rank field should exist.
        assert not hasattr(result, "rank")

    def test_penalty_and_honeypot_preserved(self):
        """Penalty and honeypot results must be accessible on the FinalScore."""
        penalty = _make_penalty(5.0)
        honeypot = _make_honeypot(10.0)
        result = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(80.0),
            skill_score=_make_score(70.0),
            experience_score=_make_score(60.0),
            education_score=_make_score(50.0),
            behavior_score=_make_score(20.0),
            penalty=penalty,
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=honeypot,
        )
        assert result.penalty_result is penalty
        assert result.honeypot_result is honeypot

    def test_custom_weights(self):
        """Custom weights should influence the final score."""
        # Give skills 100% weight, all others 0.
        weights = {
            "title": 0.0, "skills": 1.0, "experience": 0.0,
            "education": 0.0, "behavior": 0.0, "confidence": 0.0,
            "consistency": 0.0, "penalty_multiplier": 1.0, "honeypot_multiplier": 1.0,
        }
        result = compose_score(
            candidate_id="CAND_0000001",
            title_score=_make_score(0.0),
            skill_score=_make_score(80.0),
            experience_score=_make_score(0.0),
            education_score=_make_score(0.0),
            behavior_score=_make_score(0.0),
            penalty=_make_penalty(0.0),
            confidence_result={"confidence": 0.9},
            consistency_result={"consistency_score": 0.9},
            honeypot=_make_honeypot(0.0),
            weights=weights,
        )
        # Score should be dominated by skills (around 80 after adjustments).
        assert 70.0 < result.score < 85.0

"""Tests for the reason generator."""

from __future__ import annotations

from src.scoring import FinalScore, ScoreResult, PenaltyResult, HoneypotResult
from src.reasoning.generator import (
    Reason,
    ReasonGenerator,
    generate_reason,
    MAX_REASONS,
    MAX_REASON_LENGTH,
)


def _make_final(
    cid: str = "CAND_0000001",
    score: float = 75.0,
    title_score: float = 80.0,
    skill_score: float = 70.0,
    exp_score: float = 60.0,
    edu_score: float = 50.0,
    confidence: float = 0.85,
    consistency: float = 0.90,
    penalty: PenaltyResult | None = None,
    honeypot: HoneypotResult | None = None,
) -> FinalScore:
    return FinalScore(
        candidate_id=cid,
        score=score,
        component_scores={
            "title": title_score,
            "skills": skill_score,
            "experience": exp_score,
            "education": edu_score,
            "behavior": 20.0,
            "confidence": confidence,
            "consistency": consistency,
        },
        penalty_result=penalty,
        honeypot_result=honeypot,
    )


class TestGenerateReason:
    """Tests for generate_reason()."""

    def test_returns_reason_object(self):
        """Result must be a Reason with candidate_id and reasons list."""
        final = _make_final()
        result = generate_reason(final)
        assert isinstance(result, Reason)
        assert result.candidate_id == "CAND_0000001"
        assert isinstance(result.reasons, list)

    def test_max_reasons_limit(self):
        """Should never exceed MAX_REASONS (5)."""
        final = _make_final(
            penalty=PenaltyResult(penalty_score=5.0, reasons=["test"], metadata={
                "breakdown": {"weak_evidence": 5.0}
            }),
            honeypot=HoneypotResult(suspicion_score=40.0, metadata={"breakdown": {}}),
        )
        result = generate_reason(final)
        assert len(result.reasons) <= MAX_REASONS

    def test_max_reason_length(self):
        """Each reason string must not exceed MAX_REASON_LENGTH (120 chars)."""
        final = _make_final()
        result = generate_reason(final)
        for reason in result.reasons:
            assert len(reason) <= MAX_REASON_LENGTH

    def test_empty_final(self):
        """Zero scores should produce minimal or empty reasons."""
        final = _make_final(
            score=0.0, title_score=0.0, skill_score=0.0,
            exp_score=0.0, edu_score=0.0, confidence=0.0, consistency=0.0,
        )
        result = generate_reason(final)
        # Low scores should not trigger "Strong/Solid" labels.
        for r in result.reasons:
            assert "Strong" not in r and "Solid" not in r

    def test_strong_title_reason(self):
        """High title score should mention title."""
        final = _make_final(title_score=85.0)
        result = generate_reason(final)
        assert any("title" in r.lower() for r in result.reasons)

    def test_strong_skill_reason(self):
        """High skill score should mention skills."""
        final = _make_final(skill_score=90.0)
        result = generate_reason(final)
        assert any("skill" in r.lower() for r in result.reasons)

    def test_penalty_included(self):
        """Penalty should produce a reason mentioning 'penalty'."""
        final = _make_final(
            penalty=PenaltyResult(penalty_score=8.0, reasons=["test"], metadata={
                "breakdown": {"job_hopping": 8.0}
            }),
        )
        result = generate_reason(final)
        assert any("penalty" in r.lower() for r in result.reasons)

    def test_honeypot_included(self):
        """High honeypot suspicion should produce a reason."""
        final = _make_final(
            honeypot=HoneypotResult(suspicion_score=50.0, metadata={"breakdown": {}}),
        )
        result = generate_reason(final)
        assert any("suspicion" in r.lower() for r in result.reasons)

    def test_no_penalty_no_honeypot_clean(self):
        """Clean profile should not mention penalty or suspicion."""
        final = _make_final()
        result = generate_reason(final)
        for r in result.reasons:
            assert "penalty" not in r.lower()
            assert "suspicion" not in r.lower()

    def test_confidence_reason(self):
        """High confidence+consistency should produce corroboration reason."""
        final = _make_final(confidence=0.9, consistency=0.95)
        result = generate_reason(final)
        assert any("corroborated" in r.lower() or "consistent" in r.lower()
                    for r in result.reasons)

    def test_low_confidence_reason(self):
        """Low confidence should produce limited-evidence reason."""
        final = _make_final(confidence=0.1, consistency=0.9)
        result = generate_reason(final)
        assert any("evidence" in r.lower() for r in result.reasons)

    def test_with_component_results(self):
        """Passing component_results should produce richer skill/experience reasons."""
        final = _make_final(skill_score=80.0, exp_score=75.0)
        comp = {
            "skills": ScoreResult(
                score=80.0,
                reasons=["test"],
                metadata={"match_count": 5, "synergy": 9.0},
            ),
            "experience": ScoreResult(
                score=75.0,
                reasons=["test"],
                metadata={"progression": 0.7, "yoe": 8.0},
            ),
        }
        result = generate_reason(final, comp)
        assert any("expertise" in r.lower() for r in result.reasons)

    def test_deterministic(self):
        """Same input must produce same output."""
        final = _make_final(
            penalty=PenaltyResult(penalty_score=3.0, metadata={
                "breakdown": {"missing_information": 3.0}
            }),
        )
        r1 = generate_reason(final)
        r2 = generate_reason(final)
        assert r1.reasons == r2.reasons

    def test_reason_generator_wrapper(self):
        """ReasonGenerator.generate() should produce identical results."""
        gen = ReasonGenerator()
        final = _make_final()
        result = gen.generate(final)
        assert isinstance(result, Reason)
        assert result.candidate_id == "CAND_0000001"

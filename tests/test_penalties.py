"""Tests for the penalty engine."""

from __future__ import annotations

from src.features.title import TitleExtractor
from src.features.experience import ExperienceExtractor
from src.features.education import EducationExtractor
from src.features.career import CareerExtractor
from src.scoring import PenaltyResult
from src.scoring.penalties import apply_penalties
from tests.conftest import make_candidate, make_empty_candidate, _make_career


def _extract_all_features(candidate):
    return {
        "title": TitleExtractor().extract(candidate),
        "experience": ExperienceExtractor().extract(candidate),
        "education": EducationExtractor().extract(candidate),
        "career": CareerExtractor().extract(candidate),
    }


class TestPenalties:
    """Tests for apply_penalties()."""

    def test_returns_penalty_result(self):
        """Result must be a PenaltyResult."""
        candidate = make_candidate()
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert isinstance(result, PenaltyResult)
        assert isinstance(result.penalty_score, float)
        assert isinstance(result.reasons, list)
        assert isinstance(result.metadata, dict)

    def test_penalty_non_negative(self):
        """Penalty must be non-negative."""
        candidate = make_candidate()
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert result.penalty_score >= 0.0

    def test_clean_profile_low_penalty(self):
        """A clean, consistent profile should have minimal or zero penalty."""
        candidate = make_candidate()
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={"python": {"source_support_count": 3}},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert result.penalty_score < 5.0

    def test_job_hopping_penalty(self):
        """Frequent short jobs should trigger a job hopping penalty."""
        candidate = make_candidate(
            career_history=[
                _make_career(
                    company=f"Co{i}", title="Dev",
                    start_date=f"202{i}-01-01", end_date=f"202{i}-07-01",
                    duration_months=6, is_current=(i == 4),
                )
                for i in range(5)
            ]
        )
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert any("hopping" in r.lower() for r in result.reasons)

    def test_inflated_title_penalty(self):
        """Senior title with very little experience should trigger penalty."""
        candidate = make_candidate(
            profile={"years_of_experience": 0.5, "current_title": "Principal Architect"},
            career_history=[],
        )
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert any("inflated" in r.lower() or "experience" in r.lower() for r in result.reasons)

    def test_weak_evidence_penalty(self):
        """Skills with only single-source support should trigger penalty."""
        evidence = {
            "python": {"source_support_count": 1},
            "sql": {"source_support_count": 1},
            "docker": {"source_support_count": 1},
        }
        candidate = make_candidate()
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result=evidence,
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert any("weak" in r.lower() for r in result.reasons)

    def test_inconsistent_profile_penalty(self):
        """Low consistency score should trigger a penalty."""
        candidate = make_candidate()
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={},
            consistency_result={"consistency_score": 0.2, "conflicts": ["a", "b", "c"]},
        )
        assert any("consistency" in r.lower() for r in result.reasons)

    def test_missing_information_penalty(self):
        """Missing profile sections should trigger a penalty."""
        candidate = make_empty_candidate()
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert any("missing" in r.lower() for r in result.reasons)

    def test_penalty_capped(self):
        """Total penalty must be capped at 30."""
        candidate = make_empty_candidate()
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={},
            consistency_result={"consistency_score": 0.0, "conflicts": ["a", "b", "c", "d"]},
        )
        assert result.penalty_score <= 30.0

    def test_deterministic(self):
        """Same input must produce the same output."""
        candidate = make_candidate()
        features = _extract_all_features(candidate)
        evidence = {"python": {"source_support_count": 2}}
        consistency = {"consistency_score": 0.8, "conflicts": []}
        result1 = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result=evidence,
            consistency_result=consistency,
        )
        result2 = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result=evidence,
            consistency_result=consistency,
        )
        assert result1.penalty_score == result2.penalty_score

    def test_metadata_has_breakdown(self):
        """Metadata must have a 'breakdown' dict."""
        candidate = make_candidate()
        features = _extract_all_features(candidate)
        result = apply_penalties(
            candidate=candidate,
            title_features=features["title"],
            experience_features=features["experience"],
            education_features=features["education"],
            career_features=features["career"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert "breakdown" in result.metadata
        for key in ("job_hopping", "inflated_title", "weak_evidence",
                     "inconsistent_profile", "missing_information"):
            assert key in result.metadata["breakdown"]

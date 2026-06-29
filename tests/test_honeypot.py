"""Tests for the honeypot detection engine."""

from __future__ import annotations

from src.features.title import TitleExtractor
from src.features.experience import ExperienceExtractor
from src.features.education import EducationExtractor
from src.scoring import HoneypotResult
from src.scoring.honeypot import detect_honeypot
from tests.conftest import make_candidate, make_empty_candidate, _make_career, _make_education


def _extract_features(candidate):
    return {
        "title": TitleExtractor().extract(candidate),
        "experience": ExperienceExtractor().extract(candidate),
        "education": EducationExtractor().extract(candidate),
    }


class TestHoneypot:
    """Tests for detect_honeypot()."""

    def test_returns_honeypot_result(self):
        """Result must be a HoneypotResult."""
        candidate = make_candidate()
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert isinstance(result, HoneypotResult)
        assert isinstance(result.suspicion_score, float)
        assert isinstance(result.reasons, list)
        assert isinstance(result.metadata, dict)

    def test_suspicion_non_negative(self):
        """Suspicion score must be non-negative."""
        candidate = make_candidate()
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert result.suspicion_score >= 0.0

    def test_clean_profile_low_suspicion(self):
        """A clean, consistent profile should have low or zero suspicion."""
        candidate = make_candidate()
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={"python": {"source_support_count": 3, "sources": ["skills", "career_history"]}},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert result.suspicion_score < 5.0

    def test_impossible_timeline(self):
        """Career starting long before education ends should trigger suspicion."""
        candidate = make_candidate(
            career_history=[_make_career(start_date="2012-01-01")],
            education=[_make_education(end_year=2019)],
        )
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert result.suspicion_score > 0.0
        assert any("timeline" in r.lower() or "career started" in r.lower() for r in result.reasons)

    def test_contradictory_experience(self):
        """High YoE with very few positions should trigger suspicion."""
        candidate = make_candidate(
            profile={"years_of_experience": 20.0},
            career_history=[_make_career()],
        )
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert any("contradictory" in r.lower() or "position" in r.lower() for r in result.reasons)

    def test_senior_title_no_history(self):
        """Senior title with empty career history should trigger suspicion."""
        candidate = make_candidate(
            profile={"current_title": "Senior Principal Engineer"},
            career_history=[],
        )
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert any("senior" in r.lower() or "no career" in r.lower() for r in result.reasons)

    def test_fake_expertise_self_reported_only(self):
        """All skills self-reported only should trigger suspicion."""
        evidence = {
            "python": {"source_support_count": 1, "sources": ["skills"]},
            "sql": {"source_support_count": 1, "sources": ["skills"]},
            "docker": {"source_support_count": 1, "sources": ["skills"]},
        }
        candidate = make_candidate()
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result=evidence,
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert any("self-reported" in r.lower() for r in result.reasons)

    def test_suspicious_consistency(self):
        """Very low consistency with many conflicts should trigger suspicion."""
        candidate = make_candidate()
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={},
            consistency_result={"consistency_score": 0.1, "conflicts": ["a", "b", "c", "d"]},
        )
        assert any("low consistency" in r.lower() for r in result.reasons)

    def test_suspicion_capped(self):
        """Suspicion score must be capped at 100."""
        candidate = make_empty_candidate()
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={},
            consistency_result={"consistency_score": 0.0, "conflicts": ["a"]},
        )
        assert result.suspicion_score <= 100.0

    def test_deterministic(self):
        """Same input must produce the same output."""
        candidate = make_candidate()
        features = _extract_features(candidate)
        evidence = {"python": {"source_support_count": 2}}
        consistency = {"consistency_score": 0.8, "conflicts": []}
        result1 = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result=evidence,
            consistency_result=consistency,
        )
        result2 = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result=evidence,
            consistency_result=consistency,
        )
        assert result1.suspicion_score == result2.suspicion_score

    def test_metadata_has_breakdown(self):
        """Metadata must have a 'breakdown' dict."""
        candidate = make_candidate()
        features = _extract_features(candidate)
        result = detect_honeypot(
            candidate=candidate,
            experience_features=features["experience"],
            education_features=features["education"],
            title_features=features["title"],
            evidence_result={},
            consistency_result={"consistency_score": 1.0, "conflicts": []},
        )
        assert "breakdown" in result.metadata
        for key in ("impossible_timelines", "contradictory_experience",
                     "fake_expertise", "suspicious_claims"):
            assert key in result.metadata["breakdown"]

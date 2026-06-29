"""Tests for the title scoring engine."""

from __future__ import annotations

from src.features.title import TitleExtractor
from src.scoring import ScoreResult
from src.scoring.title_score import score_title
from tests.conftest import make_candidate, make_empty_candidate


def _extract_title_features(candidate):
    return TitleExtractor().extract(candidate)


class TestTitleScore:
    """Tests for score_title()."""

    def test_returns_score_result(self):
        """Result must be a ScoreResult with score, reasons, metadata."""
        candidate = make_candidate()
        features = _extract_title_features(candidate)
        result = score_title(candidate, features)
        assert isinstance(result, ScoreResult)
        assert isinstance(result.score, float)
        assert isinstance(result.reasons, list)
        assert isinstance(result.metadata, dict)

    def test_score_in_valid_range(self):
        """Score must be in [0, 100]."""
        candidate = make_candidate()
        features = _extract_title_features(candidate)
        result = score_title(candidate, features)
        assert 0.0 <= result.score <= 100.0

    def test_no_title_returns_zero(self):
        """Missing title should produce score 0."""
        candidate = make_empty_candidate()
        features = _extract_title_features(candidate)
        result = score_title(candidate, features)
        assert result.score == 0.0

    def test_exact_match_bonus(self):
        """Exact title match with target should yield a significant bonus."""
        candidate = make_candidate(
            profile={"current_title": "Senior Backend Engineer"}
        )
        features = _extract_title_features(candidate)
        result = score_title(candidate, features, target_title="Senior Backend Engineer")
        assert result.score > 40.0  # Base + seniority + exact match
        assert any("Exact title match" in r for r in result.reasons)

    def test_ai_related_bonus(self):
        """AI-related title should earn the AI bonus."""
        candidate = make_candidate(
            profile={"current_title": "ML Engineer"}
        )
        features = _extract_title_features(candidate)
        result = score_title(candidate, features)
        assert result.metadata["is_ai_related"] is True
        assert any("AI/ML related" in r for r in result.reasons)

    def test_seniority_scales_score(self):
        """Higher seniority should generally produce a higher score."""
        junior = make_candidate(profile={"current_title": "Junior Developer"})
        senior = make_candidate(profile={"current_title": "Senior Developer"})
        principal = make_candidate(profile={"current_title": "Principal Engineer"})

        junior_features = _extract_title_features(junior)
        senior_features = _extract_title_features(senior)
        principal_features = _extract_title_features(principal)

        junior_score = score_title(junior, junior_features)
        senior_score = score_title(senior, senior_features)
        principal_score = score_title(principal, principal_features)

        assert senior_score.score > junior_score.score
        assert principal_score.score >= senior_score.score

    def test_management_bonus(self):
        """Management titles should earn a management bonus."""
        candidate = make_candidate(profile={"current_title": "Engineering Manager"})
        features = _extract_title_features(candidate)
        result = score_title(candidate, features)
        assert any("Management" in r for r in result.reasons)

    def test_target_mismatch_penalty(self):
        """No token overlap with target should incur a small penalty."""
        candidate = make_candidate(profile={"current_title": "Chef"})
        features = _extract_title_features(candidate)
        result = score_title(candidate, features, target_title="Backend Engineer")
        assert any("No token overlap" in r for r in result.reasons)

    def test_deterministic(self):
        """Same input must produce the same output."""
        candidate = make_candidate()
        features = _extract_title_features(candidate)
        result1 = score_title(candidate, features)
        result2 = score_title(candidate, features)
        assert result1.score == result2.score
        assert result1.reasons == result2.reasons

    def test_metadata_contains_expected_keys(self):
        """Metadata must contain title, seniority, is_ai_related, is_manager."""
        candidate = make_candidate()
        features = _extract_title_features(candidate)
        result = score_title(candidate, features)
        for key in ("title", "seniority", "is_ai_related", "is_manager"):
            assert key in result.metadata

"""Tests for the education scoring engine."""

from __future__ import annotations

from src.features.education import EducationExtractor
from src.scoring import ScoreResult
from src.scoring.education_score import score_education
from tests.conftest import make_candidate, make_empty_candidate, _make_education


def _extract_education_features(candidate):
    return EducationExtractor().extract(candidate)


class TestEducationScore:
    """Tests for score_education()."""

    def test_returns_score_result(self):
        """Result must be a ScoreResult."""
        candidate = make_candidate()
        features = _extract_education_features(candidate)
        result = score_education(candidate, features)
        assert isinstance(result, ScoreResult)
        assert isinstance(result.score, float)
        assert isinstance(result.reasons, list)
        assert isinstance(result.metadata, dict)

    def test_score_in_valid_range(self):
        """Score must be in [0, 100]."""
        candidate = make_candidate()
        features = _extract_education_features(candidate)
        result = score_education(candidate, features)
        assert 0.0 <= result.score <= 100.0

    def test_no_education_returns_zero(self):
        """Candidate with no education should score 0."""
        candidate = make_empty_candidate()
        features = _extract_education_features(candidate)
        result = score_education(candidate, features)
        assert result.score == 0.0

    def test_natural_cap(self):
        """Education score should never dominate (capped at 50)."""
        candidate = make_candidate(
            education=[
                _make_education(degree="Ph.D", field_of_study="Computer Science", tier="tier_1"),
                _make_education(degree="M.Tech", field_of_study="Computer Science", tier="tier_1"),
            ]
        )
        features = _extract_education_features(candidate)
        result = score_education(candidate, features)
        assert result.score <= 50.0

    def test_phd_scores_higher_than_bachelors(self):
        """PhD should score higher than a Bachelor's degree."""
        phd_candidate = make_candidate(education=[_make_education(degree="Ph.D", tier="tier_1")])
        bsc_candidate = make_candidate(education=[_make_education(degree="B.Sc", tier="tier_3")])
        phd_features = _extract_education_features(phd_candidate)
        bsc_features = _extract_education_features(bsc_candidate)
        phd_result = score_education(phd_candidate, phd_features)
        bsc_result = score_education(bsc_candidate, bsc_features)
        assert phd_result.score > bsc_result.score

    def test_tier_1_scores_higher_than_tier_3(self):
        """Tier 1 institution should score higher than tier 3."""
        t1 = make_candidate(education=[_make_education(tier="tier_1")])
        t3 = make_candidate(education=[_make_education(tier="tier_3")])
        t1_features = _extract_education_features(t1)
        t3_features = _extract_education_features(t3)
        t1_result = score_education(t1, t1_features)
        t3_result = score_education(t3, t3_features)
        assert t1_result.score > t3_result.score

    def test_relevant_field_bonus(self):
        """Computer Science field should be recognized as relevant."""
        candidate = make_candidate(education=[_make_education(field_of_study="Computer Science")])
        features = _extract_education_features(candidate)
        result = score_education(candidate, features, candidate_skills={"python", "sql"})
        assert any("relevant" in r.lower() for r in result.reasons)

    def test_multiple_degrees_bonus(self):
        """Multiple degrees should earn a small bonus."""
        single = make_candidate(education=[_make_education()])
        multi = make_candidate(
            education=[
                _make_education(degree="B.Tech"),
                _make_education(degree="M.Tech"),
            ]
        )
        single_features = _extract_education_features(single)
        multi_features = _extract_education_features(multi)
        single_result = score_education(single, single_features)
        multi_result = score_education(multi, multi_features)
        assert multi_result.score > single_result.score

    def test_deterministic(self):
        """Same input must produce the same output."""
        candidate = make_candidate()
        features = _extract_education_features(candidate)
        result1 = score_education(candidate, features)
        result2 = score_education(candidate, features)
        assert result1.score == result2.score
        assert result1.reasons == result2.reasons

    def test_metadata_contains_expected_keys(self):
        """Metadata must contain degree, degree_bucket, tier, relevance."""
        candidate = make_candidate()
        features = _extract_education_features(candidate)
        result = score_education(candidate, features)
        for key in ("degree", "degree_bucket", "tier", "relevance"):
            assert key in result.metadata

    def test_missing_field_of_study(self):
        """Missing field_of_study should not crash."""
        candidate = make_candidate(education=[_make_education(field_of_study="")])
        features = _extract_education_features(candidate)
        result = score_education(candidate, features)
        assert 0.0 <= result.score <= 100.0

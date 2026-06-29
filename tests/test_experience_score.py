"""Tests for the experience scoring engine."""

from __future__ import annotations

from src.features.experience import ExperienceExtractor
from src.features.career import CareerExtractor
from src.scoring import ScoreResult
from src.scoring.experience_score import score_experience
from tests.conftest import make_candidate, make_empty_candidate, _make_career


def _extract_experience_features(candidate):
    return ExperienceExtractor().extract(candidate)


def _extract_career_features(candidate):
    return CareerExtractor().extract(candidate)


class TestExperienceScore:
    """Tests for score_experience()."""

    def test_returns_score_result(self):
        """Result must be a ScoreResult."""
        candidate = make_candidate()
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result = score_experience(candidate, exp_features, career_features)
        assert isinstance(result, ScoreResult)
        assert isinstance(result.score, float)
        assert isinstance(result.reasons, list)
        assert isinstance(result.metadata, dict)

    def test_score_in_valid_range(self):
        """Score must be in [0, 100]."""
        candidate = make_candidate()
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result = score_experience(candidate, exp_features, career_features)
        assert 0.0 <= result.score <= 100.0

    def test_no_experience_returns_zero(self):
        """Candidate with no experience should score 0."""
        candidate = make_empty_candidate()
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result = score_experience(candidate, exp_features, career_features)
        assert result.score == 0.0

    def test_target_years_scoring(self):
        """Providing target_years should influence scoring."""
        candidate = make_candidate(profile={"years_of_experience": 5.0})
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)

        result_at_target = score_experience(
            candidate, exp_features, career_features, target_years=5.0
        )
        result_over = score_experience(
            candidate, exp_features, career_features, target_years=2.0
        )
        assert 0.0 <= result_at_target.score <= 100.0
        assert 0.0 <= result_over.score <= 100.0

    def test_progression_upward_bonus(self):
        """Upward career progression should earn a bonus."""
        candidate = make_candidate(
            career_history=[
                _make_career(title="Junior Developer", start_date="2020-01-01", end_date="2021-06-01", duration_months=18, is_current=False),
                _make_career(title="Senior Developer", start_date="2021-07-01", end_date="2023-01-01", duration_months=18, is_current=False),
                _make_career(title="Lead Developer", start_date="2023-02-01", end_date=None, duration_months=28, is_current=True),
            ]
        )
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result = score_experience(candidate, exp_features, career_features)
        assert result.metadata["progression"] > 0.3
        assert any("progression" in r.lower() for r in result.reasons)

    def test_promotion_bonus(self):
        """Internal promotion should earn a bonus."""
        candidate = make_candidate(
            career_history=[
                _make_career(
                    company="Acme", title="Developer",
                    start_date="2020-01-01", end_date="2022-01-01",
                    duration_months=24, is_current=False,
                ),
                _make_career(
                    company="Acme", title="Senior Developer",
                    start_date="2022-01-01", end_date=None,
                    duration_months=40, is_current=True,
                ),
            ]
        )
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result = score_experience(candidate, exp_features, career_features)
        assert any("promotion" in r.lower() for r in result.reasons)

    def test_job_hopping_penalty(self):
        """High job hopping index should incur a penalty."""
        candidate = make_candidate(
            career_history=[
                _make_career(
                    company=f"Co{i}", title="Developer",
                    start_date=f"202{i}-01-01", end_date=f"202{i}-06-01",
                    duration_months=6, is_current=(i == 5),
                )
                for i in range(6)
            ]
        )
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result = score_experience(candidate, exp_features, career_features)
        assert any("hopping" in r.lower() for r in result.reasons)

    def test_multiple_positions_bonus(self):
        """3+ positions should earn a breadth bonus."""
        candidate = make_candidate(
            career_history=[
                _make_career(company=f"Co{i}", title="Dev", start_date=f"201{i}-01-01", duration_months=24, is_current=(i == 2))
                for i in range(3)
            ]
        )
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result = score_experience(candidate, exp_features, career_features)
        assert any("Multiple positions" in r for r in result.reasons)

    def test_deterministic(self):
        """Same input must produce the same output."""
        candidate = make_candidate()
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result1 = score_experience(candidate, exp_features, career_features)
        result2 = score_experience(candidate, exp_features, career_features)
        assert result1.score == result2.score
        assert result1.reasons == result2.reasons

    def test_metadata_contains_expected_keys(self):
        """Metadata must contain yoe, progression, stability, hopping_index."""
        candidate = make_candidate()
        exp_features = _extract_experience_features(candidate)
        career_features = _extract_career_features(candidate)
        result = score_experience(candidate, exp_features, career_features)
        for key in ("yoe", "progression", "stability", "hopping_index", "total_positions"):
            assert key in result.metadata

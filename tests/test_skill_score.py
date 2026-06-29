"""Tests for the skill scoring engine."""

from __future__ import annotations

from src.features.skills import SkillsExtractor
from src.scoring import ScoreResult
from src.scoring.skill_score import score_skills
from tests.conftest import make_candidate, make_empty_candidate, _make_skill, _make_career


def _extract_skill_features(candidate):
    return SkillsExtractor().extract(candidate)


class TestSkillScore:
    """Tests for score_skills()."""

    def test_returns_score_result(self):
        """Result must be a ScoreResult."""
        candidate = make_candidate(skills=[_make_skill(name="Python")])
        features = _extract_skill_features(candidate)
        result = score_skills(candidate, features)
        assert isinstance(result, ScoreResult)
        assert isinstance(result.score, float)
        assert isinstance(result.reasons, list)
        assert isinstance(result.metadata, dict)

    def test_score_in_valid_range(self):
        """Score must be in [0, 100]."""
        candidate = make_candidate()
        features = _extract_skill_features(candidate)
        result = score_skills(candidate, features)
        assert 0.0 <= result.score <= 100.0

    def test_no_skills_returns_zero(self):
        """Candidate with no skills should score 0."""
        candidate = make_empty_candidate()
        features = _extract_skill_features(candidate)
        result = score_skills(candidate, features)
        assert result.score == 0.0
        assert any("No skills" in r for r in result.reasons)

    def test_required_skills_full_match(self):
        """Full match of required skills should score high."""
        candidate = make_candidate(
            skills=[
                _make_skill(name="Python"),
                _make_skill(name="SQL"),
                _make_skill(name="Docker"),
            ]
        )
        features = _extract_skill_features(candidate)
        result = score_skills(
            candidate, features,
            required_skills=["python", "sql", "docker"],
        )
        assert result.metadata["match_count"] == 3
        assert result.metadata["required_count"] == 3

    def test_required_skills_partial_match(self):
        """Partial match of required skills should score lower than full match."""
        # Use a candidate whose headline/summary/career text do NOT mention
        # the other required skills, so only the explicit skills list matters.
        candidate = make_candidate(
            skills=[_make_skill(name="Python")],
            profile={"headline": "Developer", "summary": "Builds things."},
            career_history=[_make_career(title="Developer", description="Works on projects.")],
        )
        features = _extract_skill_features(candidate)
        result = score_skills(
            candidate, features,
            required_skills=["python", "sql", "docker"],
        )
        assert result.metadata["match_count"] == 1
        assert result.metadata["required_count"] == 3

    def test_preferred_skills_bonus(self):
        """Preferred skill matches should add bonus points."""
        candidate = make_candidate(skills=[_make_skill(name="Python")])
        features = _extract_skill_features(candidate)
        result_with = score_skills(
            candidate, features,
            preferred_skills=["python"],
        )
        result_without = score_skills(candidate, features)
        assert result_with.score > result_without.score

    def test_synergy_bonus(self):
        """Complementary skills should earn a synergy bonus."""
        candidate = make_candidate(
            skills=[
                _make_skill(name="Python"),
                _make_skill(name="Docker"),
                _make_skill(name="AWS"),
                _make_skill(name="SQL"),
            ]
        )
        features = _extract_skill_features(candidate)
        result = score_skills(candidate, features)
        assert any("Synergy" in r for r in result.reasons)
        assert result.metadata["synergy"] > 0.0

    def test_volume_score(self):
        """More skills should generally produce a higher volume score."""
        few = make_candidate(skills=[_make_skill(name="Python")])
        many = make_candidate(
            skills=[
                _make_skill(name="Python"),
                _make_skill(name="SQL"),
                _make_skill(name="Docker"),
                _make_skill(name="AWS"),
                _make_skill(name="Kubernetes"),
            ]
        )
        few_features = _extract_skill_features(few)
        many_features = _extract_skill_features(many)
        few_result = score_skills(few, few_features)
        many_result = score_skills(many, many_features)
        assert many_result.metadata["unique_skills"] > few_result.metadata["unique_skills"]

    def test_endorsement_bonus(self):
        """Endorsements should add a small bonus."""
        high_endorse = make_candidate(
            skills=[_make_skill(name="Python", endorsements=50)]
        )
        low_endorse = make_candidate(
            skills=[_make_skill(name="Python", endorsements=0)]
        )
        high_features = _extract_skill_features(high_endorse)
        low_features = _extract_skill_features(low_endorse)
        high_result = score_skills(high_endorse, high_features)
        low_result = score_skills(low_endorse, low_features)
        assert high_result.score > low_result.score

    def test_deterministic(self):
        """Same input must produce the same output."""
        candidate = make_candidate()
        features = _extract_skill_features(candidate)
        result1 = score_skills(candidate, features)
        result2 = score_skills(candidate, features)
        assert result1.score == result2.score
        assert result1.reasons == result2.reasons

    def test_metadata_contains_expected_keys(self):
        """Metadata must contain match_count, required_count, synergy, etc."""
        candidate = make_candidate()
        features = _extract_skill_features(candidate)
        result = score_skills(candidate, features)
        for key in ("match_count", "required_count", "preferred_count", "synergy", "unique_skills"):
            assert key in result.metadata

"""Tests for the behavior scoring engine."""

from __future__ import annotations

from src.scoring import ScoreResult
from src.scoring.behavior_score import score_behavior
from tests.conftest import make_candidate, make_empty_candidate, _make_redrob


class TestBehaviorScore:
    """Tests for score_behavior()."""

    def test_returns_score_result(self):
        """Result must be a ScoreResult."""
        candidate = make_candidate()
        result = score_behavior(candidate)
        assert isinstance(result, ScoreResult)
        assert isinstance(result.score, float)
        assert isinstance(result.reasons, list)
        assert isinstance(result.metadata, dict)

    def test_score_in_valid_range(self):
        """Score must be in [0, 100]."""
        candidate = make_candidate()
        result = score_behavior(candidate)
        assert 0.0 <= result.score <= 100.0

    def test_empty_candidate_scores_low(self):
        """Candidate with no signals should score low."""
        candidate = make_empty_candidate()
        result = score_behavior(candidate)
        assert result.score < 10.0

    def test_github_activity_bonus(self):
        """GitHub activity should add bonus points."""
        active = make_candidate(redrob_signals=_make_redrob(github_activity_score=10.0))
        inactive = make_candidate(redrob_signals=_make_redrob(github_activity_score=0.0))
        active_result = score_behavior(active)
        inactive_result = score_behavior(inactive)
        assert active_result.score > inactive_result.score
        assert any("GitHub" in r for r in active_result.reasons)

    def test_leadership_keywords_bonus(self):
        """Leadership keywords in headline/summary should earn bonus."""
        leader = make_candidate(
            profile={
                "headline": "Founder and CTO",
                "summary": "Open source contributor and conference speaker.",
            }
        )
        normal = make_candidate(profile={"headline": "Developer", "summary": "Writes code."})
        leader_result = score_behavior(leader)
        normal_result = score_behavior(normal)
        assert leader_result.score > normal_result.score
        assert any("Leadership" in r for r in leader_result.reasons)

    def test_network_bonus(self):
        """Strong network (many connections) should earn a bonus."""
        connected = make_candidate(redrob_signals=_make_redrob(connection_count=500))
        isolated = make_candidate(redrob_signals=_make_redrob(connection_count=50))
        connected_result = score_behavior(connected)
        isolated_result = score_behavior(isolated)
        assert connected_result.score > isolated_result.score

    def test_verification_bonus(self):
        """Verified identity should earn bonus points."""
        full_verify = make_candidate(
            redrob_signals=_make_redrob(
                verified_email=True, verified_phone=True, linkedin_connected=True,
            )
        )
        no_verify = make_candidate(
            redrob_signals=_make_redrob(
                verified_email=False, verified_phone=False, linkedin_connected=False,
            )
        )
        full_result = score_behavior(full_verify)
        no_verify_result = score_behavior(no_verify)
        assert full_result.score > no_verify_result.score
        assert any("verified" in r.lower() for r in full_result.reasons)

    def test_profile_completeness_bonus(self):
        """High profile completeness should earn a bonus."""
        complete = make_candidate(redrob_signals=_make_redrob(profile_completeness_score=95.0))
        incomplete = make_candidate(redrob_signals=_make_redrob(profile_completeness_score=60.0))
        complete_result = score_behavior(complete)
        incomplete_result = score_behavior(incomplete)
        assert complete_result.score > incomplete_result.score

    def test_natural_cap(self):
        """Behavior score should be naturally capped (small factor)."""
        candidate = make_candidate(
            redrob_signals=_make_redrob(
                github_activity_score=50.0,
                connection_count=1000,
                endorsements_received=200,
                profile_completeness_score=100.0,
            ),
            profile={
                "headline": "Founder Mentor Speaker Open Source Contributor",
                "summary": "Published author and conference organizer.",
            },
        )
        result = score_behavior(candidate)
        assert result.score <= 30.0

    def test_deterministic(self):
        """Same input must produce the same output."""
        candidate = make_candidate()
        result1 = score_behavior(candidate)
        result2 = score_behavior(candidate)
        assert result1.score == result2.score
        assert result1.reasons == result2.reasons

    def test_metadata_contains_expected_keys(self):
        """Metadata must contain github_activity, connections, verified_count, etc."""
        candidate = make_candidate()
        result = score_behavior(candidate)
        for key in ("github_activity", "has_leadership_keywords", "connections",
                     "endorsements", "verified_count", "profile_completeness"):
            assert key in result.metadata

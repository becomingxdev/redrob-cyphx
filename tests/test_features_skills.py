import pytest
from src.models.factory import CandidateFactory
from src.features.skills import SkillsExtractor


@pytest.fixture
def base_candidate_dict() -> dict:
    return {
        "candidate_id": "CAND_TEST_01",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "",
            "summary": "",
            "location": "",
            "country": "",
            "years_of_experience": 5.0,
            "current_title": "",
            "current_company": "",
            "current_company_size": "",
            "current_industry": "",
        },
        "career_history": [],
        "education": [],
        "skills": [],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 85.0,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-01-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 5,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {},
            "connection_count": 100,
            "endorsements_received": 10,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 10.0, "max": 20.0},
            "preferred_work_mode": "remote",
            "willing_to_relocate": True,
            "github_activity_score": 80.0,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 2,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


class TestSkillsExtraction:
    """Original tests — basic extraction and empty handling."""

    def test_skills_extraction(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("SQL", "expert", 25, 72),
            _make_skill("Git", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = SkillsExtractor()
        features = extractor.extract(candidate)

        assert features["unique_skills"] == 3
        assert features["skills"] == ["git", "python", "sql"]
        assert features["endorsement_total"] == 40
        assert features["expert_count"] == 1
        assert features["advanced_count"] == 1
        assert features["intermediate_count"] == 1
        assert features["duration_statistics"]["min"] == 36
        assert features["duration_statistics"]["max"] == 72
        assert features["duration_statistics"]["avg"] == 56.0

    def test_skills_extraction_empty(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = []
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = SkillsExtractor()
        features = extractor.extract(candidate)

        assert features["unique_skills"] == 0
        assert features["skills"] == []
        assert features["endorsement_total"] == 0
        assert features["expert_count"] == 0
        assert features["duration_statistics"]["min"] == 0
        assert features["duration_statistics"]["avg"] == 0.0


class TestCaseInsensitiveDedup:
    """Verify that duplicate detection is case-insensitive."""

    def test_identical_case_duplicates(self, base_candidate_dict) -> None:
        """Two entries with the exact same name count as one unique skill."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("Python", "expert", 20, 72),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["unique_skills"] == 1
        assert features["skills"] == ["python"]
        # Endorsements should still sum both entries (before dedup)
        assert features["endorsement_total"] == 30

    def test_mixed_case_python_python_upper(self, base_candidate_dict) -> None:
        """Python, python, PYTHON — all three should deduplicate to one."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("python", "expert", 20, 48),
            _make_skill("PYTHON", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["unique_skills"] == 1
        assert features["skills"] == ["python"]

    def test_mixed_case_various_skills(self, base_candidate_dict) -> None:
        """Mixed-case variants of different skills should dedup each group."""
        base_candidate_dict["skills"] = [
            _make_skill("Java", "advanced", 15, 48),
            _make_skill("java", "beginner", 3, 12),
            _make_skill("JAVA", "expert", 30, 72),
            _make_skill("Docker", "intermediate", 8, 24),
            _make_skill("docker", "advanced", 12, 36),
            _make_skill("Kubernetes", "beginner", 2, 6),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["unique_skills"] == 3
        assert sorted(features["skills"]) == ["docker", "java", "kubernetes"]

    def test_whitespace_trimming_in_dedup(self, base_candidate_dict) -> None:
        """Whitespace around skill names should not prevent dedup."""
        base_candidate_dict["skills"] = [
            _make_skill("  Python  ", "advanced", 10, 60),
            _make_skill("python", "expert", 20, 48),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["unique_skills"] == 1
        assert features["skills"] == ["python"]


class TestSkillsEndorsements:
    """Verify endorsement totaling and zero handling."""

    def test_endorsements_sum_all_entries(self, base_candidate_dict) -> None:
        """endorsement_total must sum across all skill entries including duplicates."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("SQL", "expert", 25, 72),
            _make_skill("Git", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["endorsement_total"] == 10 + 25 + 5

    def test_zero_endorsements(self, base_candidate_dict) -> None:
        """Skills with 0 endorsements should not break the total."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 0, 60),
            _make_skill("SQL", "expert", 0, 72),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["endorsement_total"] == 0
        assert features["unique_skills"] == 2


class TestSkillsDurationStatistics:
    """Verify duration min/max/avg/total calculations."""

    def test_duration_with_none_values(self, base_candidate_dict) -> None:
        """Skills with duration_months=None should be excluded from duration stats."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("SQL", "expert", 25, None),
            _make_skill("Git", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        # Only Python (60) and Git (36) have durations
        assert features["duration_statistics"]["min"] == 36
        assert features["duration_statistics"]["max"] == 60
        assert features["duration_statistics"]["avg"] == 48.0  # (60+36)/2
        assert features["duration_statistics"]["total"] == 96
        # SQL should still count as a unique skill
        assert features["unique_skills"] == 3

    def test_duration_all_none(self, base_candidate_dict) -> None:
        """When all skills have None duration, stats should be zeroed."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, None),
            _make_skill("SQL", "expert", 25, None),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["duration_statistics"] == {"min": 0, "max": 0, "avg": 0.0, "total": 0}


class TestSkillsProficiencyBuckets:
    """Verify proficiency bucket mapping and distribution counts."""

    def test_unknown_proficiency_mapped(self, base_candidate_dict) -> None:
        """A proficiency string not in any bucket should be preserved as-is."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "wizard", 10, 60),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        # "wizard" is not in any bucket, so it stays as "wizard"
        assert features["proficiency_distribution"].get("wizard", 0) == 1

    def test_empty_proficiency_mapped_to_unknown(self, base_candidate_dict) -> None:
        """Empty proficiency should map to 'unknown'."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "", 10, 60),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["proficiency_distribution"].get("unknown", 0) == 1

    def test_bucket_count_keys_present(self, base_candidate_dict) -> None:
        """The result must contain explicit {bucket}_count keys for all config buckets."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "expert", 10, 60),
            _make_skill("SQL", "advanced", 20, 48),
            _make_skill("Git", "beginner", 5, 24),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["expert_count"] == 1
        assert features["advanced_count"] == 1
        assert features["beginner_count"] == 1
        assert features["intermediate_count"] == 0


class TestSkillsDeterminism:
    """Verify the extractor is deterministic."""

    def test_same_input_same_output(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("python", "expert", 20, 48),
            _make_skill("SQL", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = SkillsExtractor()

        assert extractor.extract(candidate) == extractor.extract(candidate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(
    name: str,
    proficiency: str,
    endorsements: int,
    duration_months: int | None,
) -> dict:
    """Create a minimal skill entry dict."""
    return {
        "name": name,
        "proficiency": proficiency,
        "endorsements": endorsements,
        "duration_months": duration_months,
    }

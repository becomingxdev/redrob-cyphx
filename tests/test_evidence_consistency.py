import pytest

from src.evidence.consistency import ConsistencyAnalyzer, DEFAULT_WEIGHTS
from src.models.factory import CandidateFactory


@pytest.fixture
def base_candidate_dict() -> dict:
    return {
        "candidate_id": "CAND_EVID_02",
        "profile": {
            "anonymized_name": "Consistency Tester",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(name: str) -> dict:
    return {"name": name, "proficiency": "intermediate", "endorsements": 1, "duration_months": 12}


def _make_role(
    title: str,
    duration_months: int = 12,
    company: str = "Acme",
    industry: str = "Tech",
) -> dict:
    return {
        "company": company,
        "title": title,
        "start_date": "2020-01-01",
        "end_date": "2021-01-01",
        "duration_months": duration_months,
        "is_current": False,
        "industry": industry,
        "company_size": "100-500",
        "description": "",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConsistencyScoreShape:
    """Output shape and range guarantees."""

    def test_score_in_unit_interval(self, base_candidate_dict) -> None:
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert 0.0 <= result["consistency_score"] <= 1.0

    def test_result_has_required_keys(self, base_candidate_dict) -> None:
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        for key in ("consistency_score", "checks", "agreements", "conflicts"):
            assert key in result

    def test_each_check_has_metadata(self, base_candidate_dict) -> None:
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        for name, check in result["checks"].items():
            assert "agreed" in check and isinstance(check["agreed"], bool)
            assert "weight" in check and isinstance(check["weight"], float)
            assert "detail" in check and isinstance(check["detail"], str)
            assert name in DEFAULT_WEIGHTS


class TestConsistencyAgreements:
    """Specific agreement scenarios."""

    def test_headline_title_seniority_match(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["headline"] = "Senior Backend Engineer"
        base_candidate_dict["profile"]["current_title"] = "Senior Engineer"
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["headline_title"]["agreed"] is True

    def test_headline_title_seniority_mismatch(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["headline"] = "Junior Developer"
        base_candidate_dict["profile"]["current_title"] = "Principal Engineer"
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["headline_title"]["agreed"] is False

    def test_title_skills_domain_overlap(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["current_title"] = "Backend Engineer"
        base_candidate_dict["skills"] = [_make_skill("django"), _make_skill("python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["title_skills_domain"]["agreed"] is True

    def test_title_skills_domain_no_overlap(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["current_title"] = "Backend Engineer"
        base_candidate_dict["skills"] = [_make_skill("react"), _make_skill("vue")]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["title_skills_domain"]["agreed"] is False

    def test_experience_yoe_consistent(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["years_of_experience"] = 5.0
        base_candidate_dict["career_history"] = [
            _make_role("Engineer", duration_months=24),
            _make_role("Engineer", duration_months=24),
        ]  # 48 months = 4 years; within tolerance of 5.
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["experience_yoe"]["agreed"] is True

    def test_experience_yoe_large_mismatch(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["years_of_experience"] = 20.0
        base_candidate_dict["career_history"] = [
            _make_role("Engineer", duration_months=12),
        ]  # 1 year vs 20 declared.
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["experience_yoe"]["agreed"] is False

    def test_title_career_title_above_history(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["current_title"] = "Director of Engineering"
        base_candidate_dict["career_history"] = [
            _make_role("Junior Engineer"), _make_role("Engineer"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["title_career"]["agreed"] is False

    def test_assessment_skills_overlap(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("python")]
        base_candidate_dict["redrob_signals"]["skill_assessment_scores"] = {"python": 0.9}
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["assessment_skills"]["agreed"] is True

    def test_title_career_senior_current_lead_past(self, base_candidate_dict) -> None:
        """Senior current title should be consistent if they were a Lead in the past."""
        base_candidate_dict["profile"]["current_title"] = "Senior Engineer"
        base_candidate_dict["career_history"] = [
            _make_role("Lead Engineer"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["title_career"]["agreed"] is True

    def test_title_skills_domain_no_false_overlap_on_substring(self, base_candidate_dict) -> None:
        """Skill 'html' should not falsely match the 'ml' domain due to substring match."""
        base_candidate_dict["profile"]["current_title"] = "ML Engineer"
        base_candidate_dict["skills"] = [_make_skill("html")]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["checks"]["title_skills_domain"]["agreed"] is False


class TestConsistencyNeutral:
    """Missing signals are treated as neutral (no conflict), not conflicts."""

    def test_empty_candidate_neutral(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["years_of_experience"] = 0.0
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConsistencyAnalyzer().analyze(candidate)
        # All checks should be neutral-agreement on an empty candidate.
        assert result["consistency_score"] == 1.0
        assert result["conflicts"] == []


class TestConsistencyConfig:
    """Configurable weights via evidence_rules.yaml."""

    def test_default_weights_used_when_no_config(self) -> None:
        analyzer = ConsistencyAnalyzer(config_path="config/__does_not_exist__.yaml")
        assert analyzer.weights == DEFAULT_WEIGHTS

    def test_full_agreement_yields_one(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["headline"] = "Senior Backend Engineer"
        base_candidate_dict["profile"]["current_title"] = "Senior Backend Engineer"
        base_candidate_dict["profile"]["years_of_experience"] = 4.0
        base_candidate_dict["skills"] = [_make_skill("python"), _make_skill("django")]
        base_candidate_dict["career_history"] = [
            _make_role("Senior Backend Engineer", duration_months=24),
            _make_role("Backend Engineer", duration_months=24),
        ]
        base_candidate_dict["education"] = [
            {"institution": "X", "degree": "B.E.", "field_of_study": "Computer Science",
             "start_year": 2014, "end_year": 2018, "grade": None, "tier": "tier_2"},
        ]
        base_candidate_dict["redrob_signals"]["skill_assessment_scores"] = {"python": 0.9}
        candidate = CandidateFactory.create(base_candidate_dict)

        result = ConsistencyAnalyzer().analyze(candidate)
        assert result["consistency_score"] == 1.0


class TestConsistencyDeterminism:
    def test_same_input_same_output(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["headline"] = "Senior Engineer"
        base_candidate_dict["profile"]["current_title"] = "Senior Engineer"
        base_candidate_dict["skills"] = [_make_skill("python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        analyzer = ConsistencyAnalyzer()
        assert analyzer.analyze(candidate) == analyzer.analyze(candidate)

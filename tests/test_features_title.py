import pytest
from src.models.factory import CandidateFactory
from src.features.title import TitleExtractor


@pytest.fixture
def base_candidate_dict() -> dict:
    return {
        "candidate_id": "CAND_TEST_01",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "Software Engineer",
            "summary": "Summary",
            "location": "San Francisco",
            "country": "USA",
            "years_of_experience": 5.0,
            "current_title": "Senior Software Engineer",
            "current_company": "Google",
            "current_company_size": "10001+",
            "current_industry": "Tech",
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


def test_title_extraction_senior(base_candidate_dict) -> None:
    candidate = CandidateFactory.create(base_candidate_dict)
    extractor = TitleExtractor()
    features = extractor.extract(candidate)

    assert features["title"] == "Senior Software Engineer"
    assert features["normalized_title"] == "senior software engineer"
    assert features["seniority"] == "senior"
    assert features["is_manager"] is False
    assert features["is_ai_related"] is False
    assert "senior" in features["tokens"]


def test_title_extraction_ai_manager(base_candidate_dict) -> None:
    base_candidate_dict["profile"]["current_title"] = "Principal AI Research Manager"
    candidate = CandidateFactory.create(base_candidate_dict)
    extractor = TitleExtractor()
    features = extractor.extract(candidate)

    assert features["seniority"] == "principal"
    assert features["is_manager"] is True
    assert features["is_ai_related"] is True


def test_title_extraction_empty_or_missing(base_candidate_dict) -> None:
    base_candidate_dict["profile"]["current_title"] = ""
    candidate = CandidateFactory.create(base_candidate_dict)
    extractor = TitleExtractor()
    features = extractor.extract(candidate)

    assert features["title"] == ""
    assert features["normalized_title"] == ""
    assert features["seniority"] == "mid"
    assert features["is_manager"] is False
    assert features["is_ai_related"] is False
    assert features["tokens"] == []

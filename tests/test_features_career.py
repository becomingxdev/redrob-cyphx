import pytest
from src.models.factory import CandidateFactory
from src.features.career import CareerExtractor


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
        "career_history": [
            {
                "company": "Google",
                "title": "Software Engineer",
                "start_date": "2020-01-01",
                "end_date": "2022-01-01",
                "duration_months": 24,
                "is_current": False,
                "industry": "Tech",
                "company_size": "10000+",
                "description": "",
            },
            {
                "company": "Google",
                "title": "Senior Software Engineer",
                "start_date": "2022-01-01",
                "end_date": "2024-01-01",
                "duration_months": 24,
                "is_current": False,
                "industry": "Tech",
                "company_size": "10000+",
                "description": "",
            },
        ],
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


def test_career_extraction(base_candidate_dict) -> None:
    candidate = CandidateFactory.create(base_candidate_dict)
    extractor = CareerExtractor()
    features = extractor.extract(candidate)

    assert features["industries_worked_in"] == ["Tech"]
    assert features["companies_worked_for"] == ["Google"]
    assert features["company_size_history"] == ["10000+"]
    assert features["title_progression"] == ["Software Engineer", "Senior Software Engineer"]
    assert features["role_transitions"] == [
        {"from_title": "Software Engineer", "to_title": "Senior Software Engineer"}
    ]
    assert features["promotion_indicators"] is True
    assert features["career_stability_metrics"]["average_tenure_months"] == 24.0


def test_career_extraction_empty(base_candidate_dict) -> None:
    base_candidate_dict["career_history"] = []
    candidate = CandidateFactory.create(base_candidate_dict)
    extractor = CareerExtractor()
    features = extractor.extract(candidate)

    assert features["industries_worked_in"] == []
    assert features["companies_worked_for"] == []
    assert features["company_size_history"] == []
    assert features["title_progression"] == []
    assert features["role_transitions"] == []
    assert features["promotion_indicators"] is False
    assert features["career_stability_metrics"]["average_tenure_months"] == 0.0

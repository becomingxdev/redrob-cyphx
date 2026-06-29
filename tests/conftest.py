"""Shared test fixtures for scoring engine tests."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from src.models.candidate import Candidate
from src.models.career import CareerHistory
from src.models.certification import Certification
from src.models.education import Education
from src.models.language import Language
from src.models.profile import Profile
from src.models.redrob import RedrobSignals
from src.models.salary import SalaryRange
from src.models.skill import Skill


def _make_salary(min_val: float = 10.0, max_val: float = 20.0) -> SalaryRange:
    return SalaryRange(min=min_val, max=max_val)


def _make_redrob(**overrides) -> RedrobSignals:
    defaults = {
        "profile_completeness_score": 85.0,
        "signup_date": "2024-01-01",
        "last_active_date": "2025-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 20,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.5,
        "avg_response_time_hours": 48.0,
        "skill_assessment_scores": {"python": 85.0},
        "connection_count": 200,
        "endorsements_received": 30,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": _make_salary(),
        "preferred_work_mode": "remote",
        "willing_to_relocate": True,
        "github_activity_score": 5.0,
        "search_appearance_30d": 100,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.7,
        "offer_acceptance_rate": 0.6,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    defaults.update(overrides)
    return RedrobSignals(**defaults)


def _make_profile(**overrides) -> Profile:
    defaults = {
        "anonymized_name": "Test Candidate",
        "headline": "Senior Backend Engineer",
        "summary": "Experienced backend developer with Python and SQL expertise.",
        "location": "Bangalore",
        "country": "India",
        "years_of_experience": 5.0,
        "current_title": "Senior Backend Engineer",
        "current_company": "TechCorp",
        "current_company_size": "5001-10000",
        "current_industry": "Technology",
    }
    defaults.update(overrides)
    return Profile(**defaults)


def _make_career(**overrides) -> CareerHistory:
    defaults = {
        "company": "TechCorp",
        "title": "Senior Backend Engineer",
        "start_date": "2022-01-15",
        "end_date": None,
        "duration_months": 30,
        "is_current": True,
        "industry": "Technology",
        "company_size": "5001-10000",
        "description": "Built scalable microservices with Python and Docker.",
    }
    defaults.update(overrides)
    return CareerHistory(**defaults)


def _make_education(**overrides) -> Education:
    defaults = {
        "institution": "Indian Institute of Technology",
        "degree": "B.Tech",
        "field_of_study": "Computer Science",
        "start_year": 2015,
        "end_year": 2019,
        "grade": "8.5 CGPA",
        "tier": "tier_1",
    }
    defaults.update(overrides)
    return Education(**defaults)


def _make_skill(**overrides) -> Skill:
    defaults = {
        "name": "Python",
        "proficiency": "advanced",
        "endorsements": 15,
        "duration_months": 48,
    }
    defaults.update(overrides)
    return Skill(**defaults)


def _make_cert(**overrides) -> Certification:
    defaults = {
        "name": "AWS Solutions Architect",
        "issuer": "Amazon",
        "year": 2023,
    }
    defaults.update(overrides)
    return Certification(**defaults)


def _make_language(**overrides) -> Language:
    defaults = {
        "language": "English",
        "proficiency": "professional",
    }
    defaults.update(overrides)
    return Language(**defaults)


def _merge_model(model_obj, overrides):
    """Merge a dict of partial overrides into an existing dataclass instance.

    Returns a new instance with the overridden fields applied. This lets
    tests pass e.g. ``profile={"current_title": "X"}`` without having to
    rebuild the whole Profile.
    """
    if not isinstance(overrides, dict):
        return overrides  # Already a model object; pass through.
    current = {f.name: getattr(model_obj, f.name) for f in fields(model_obj)}
    current.update(overrides)
    return type(model_obj)(**current)


def make_candidate(**overrides) -> Candidate:
    """Create a fully populated test candidate with sensible defaults.

    Any field can be overridden via keyword arguments. For ``profile`` and
    ``redrob_signals``, a partial dict is merged into the default model
    instance; pass a full model object to replace it entirely.
    """
    profile = overrides.pop("profile", _make_profile())
    if isinstance(profile, dict):
        profile = _merge_model(_make_profile(), profile)

    redrob = overrides.pop("redrob_signals", _make_redrob())
    if isinstance(redrob, dict):
        redrob = _merge_model(_make_redrob(), redrob)

    defaults = {
        "candidate_id": "CAND_9999999",
        "profile": profile,
        "career_history": [_make_career()],
        "education": [_make_education()],
        "skills": [_make_skill()],
        "certifications": [_make_cert()],
        "languages": [_make_language()],
        "redrob_signals": redrob,
        "raw": {},
    }
    defaults.update(overrides)
    return Candidate(**defaults)


def make_empty_candidate() -> Candidate:
    """Create a minimally populated candidate with empty/null fields."""
    return Candidate(
        candidate_id="CAND_0000000",
        profile=Profile(
            anonymized_name="",
            headline="",
            summary="",
            location="",
            country="",
            years_of_experience=0.0,
            current_title="",
            current_company="",
            current_company_size="",
            current_industry="",
        ),
        career_history=[],
        education=[],
        skills=[],
        certifications=[],
        languages=[],
        redrob_signals=RedrobSignals(
            profile_completeness_score=0.0,
            signup_date="",
            last_active_date="",
            open_to_work_flag=False,
            profile_views_received_30d=0,
            applications_submitted_30d=0,
            recruiter_response_rate=0.0,
            avg_response_time_hours=0.0,
            skill_assessment_scores={},
            connection_count=0,
            endorsements_received=0,
            notice_period_days=0,
            expected_salary_range_inr_lpa=SalaryRange(min=0.0, max=0.0),
            preferred_work_mode="",
            willing_to_relocate=False,
            github_activity_score=0.0,
            search_appearance_30d=0,
            saved_by_recruiters_30d=0,
            interview_completion_rate=0.0,
            offer_acceptance_rate=0.0,
            verified_email=False,
            verified_phone=False,
            linkedin_connected=False,
        ),
        raw={},
    )

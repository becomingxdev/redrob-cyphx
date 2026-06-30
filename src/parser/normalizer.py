"""Normalizes candidate records by cleaning data without inferring information."""

from __future__ import annotations

from typing import Any


def _strip_str(value: Any) -> str:
    """Strip whitespace if the value is a string; return empty string for None."""
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return value


def _normalize_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize all string fields in a profile dict."""
    string_keys = {
        "anonymized_name",
        "headline",
        "summary",
        "location",
        "country",
        "current_title",
        "current_company",
        "current_company_size",
        "current_industry",
    }
    result: dict[str, Any] = {}
    for key in string_keys:
        result[key] = _strip_str(data.get(key, ""))
    result["years_of_experience"] = data.get("years_of_experience", 0.0) or 0.0
    return result


def _normalize_career(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single career history entry."""
    string_keys = {"company", "title", "start_date", "industry", "company_size", "description"}
    result: dict[str, Any] = {}
    for key in string_keys:
        result[key] = _strip_str(data.get(key, ""))
    result["end_date"] = _strip_str(data.get("end_date"))
    result["duration_months"] = data.get("duration_months", 0) or 0
    result["is_current"] = data.get("is_current", False) or False
    return result


def _normalize_education(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single education entry."""
    string_keys = {"institution", "degree", "field_of_study", "tier"}
    result: dict[str, Any] = {}
    for key in string_keys:
        result[key] = _strip_str(data.get(key, ""))
    result["start_year"] = data.get("start_year", 0) or 0
    result["end_year"] = data.get("end_year", 0) or 0
    result["grade"] = _strip_str(data.get("grade"))
    return result


def _normalize_skill(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single skill entry."""
    return {
        "name": _strip_str(data.get("name", "")),
        "proficiency": _strip_str(data.get("proficiency", "")),
        "endorsements": data.get("endorsements", 0) or 0,
        "duration_months": data.get("duration_months"),
    }


def _normalize_certification(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single certification entry."""
    return {
        "name": _strip_str(data.get("name", "")),
        "issuer": _strip_str(data.get("issuer", "")),
        "year": data.get("year", 0) or 0,
    }


def _normalize_language(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single language entry."""
    return {
        "language": _strip_str(data.get("language", "")),
        "proficiency": _strip_str(data.get("proficiency", "")),
    }


def _normalize_salary(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize the salary range sub-object."""
    return {
        "min": data.get("min", 0.0) or 0.0,
        "max": data.get("max", 0.0) or 0.0,
    }


def _normalize_redrob_signals(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize the redrob_signals sub-object."""
    result: dict[str, Any] = {
        "profile_completeness_score": data.get("profile_completeness_score", 0.0) or 0.0,
        "signup_date": _strip_str(data.get("signup_date", "")),
        "last_active_date": _strip_str(data.get("last_active_date", "")),
        "open_to_work_flag": data.get("open_to_work_flag", False) or False,
        "profile_views_received_30d": data.get("profile_views_received_30d", 0) or 0,
        "applications_submitted_30d": data.get("applications_submitted_30d", 0) or 0,
        "recruiter_response_rate": data.get("recruiter_response_rate", 0.0) or 0.0,
        "avg_response_time_hours": data.get("avg_response_time_hours", 0.0) or 0.0,
        "skill_assessment_scores": data.get("skill_assessment_scores") or {},
        "connection_count": data.get("connection_count", 0) or 0,
        "endorsements_received": data.get("endorsements_received", 0) or 0,
        "notice_period_days": data.get("notice_period_days", 0) or 0,
        "expected_salary_range_inr_lpa": _normalize_salary(
            data.get("expected_salary_range_inr_lpa") or {}
        ),
        "preferred_work_mode": _strip_str(data.get("preferred_work_mode", "")),
        "willing_to_relocate": data.get("willing_to_relocate", False) or False,
        "github_activity_score": data.get("github_activity_score", 0.0) or 0.0,
        "search_appearance_30d": data.get("search_appearance_30d", 0) or 0,
        "saved_by_recruiters_30d": data.get("saved_by_recruiters_30d", 0) or 0,
        "interview_completion_rate": data.get("interview_completion_rate", 0.0) or 0.0,
        "offer_acceptance_rate": data.get("offer_acceptance_rate", 0.0) or 0.0,
        "verified_email": data.get("verified_email", False) or False,
        "verified_phone": data.get("verified_phone", False) or False,
        "linkedin_connected": data.get("linkedin_connected", False) or False,
    }
    return result


def _deduplicate_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate skills by name, keeping the first occurrence."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for skill in skills:
        name = skill.get("name", "")
        if name not in seen:
            seen.add(name)
            result.append(skill)
    return result


def normalize_candidate(record: dict) -> dict:
    """Clean and normalize a candidate record without inferring information.

    - Strips whitespace from all string fields.
    - Replaces None with sensible defaults.
    - Removes duplicate skills (by name, first occurrence wins).
    - Ensures optional list fields default to empty lists.
    - Returns a new dict; the original is not mutated.
    """
    # PERF 3: Build a fresh dict instead of deepcopy — every key is
    # explicitly overwritten below, so deepcopy is pure overhead.
    # All reads come from the original `record`; writes go to `data`.
    data: dict[str, Any] = {}

    data["candidate_id"] = _strip_str(record.get("candidate_id", ""))
    data["profile"] = _normalize_profile(record.get("profile") or {})

    # career_history — list of dicts
    raw_career = record.get("career_history") or []
    data["career_history"] = [_normalize_career(item) for item in raw_career]

    # education — list of dicts
    raw_edu = record.get("education") or []
    data["education"] = [_normalize_education(item) for item in raw_edu]

    # skills — list of dicts, deduplicated by name
    raw_skills = record.get("skills") or []
    normalized_skills = [_normalize_skill(item) for item in raw_skills]
    data["skills"] = _deduplicate_skills(normalized_skills)

    # certifications — optional list of dicts
    raw_certs = record.get("certifications") or []
    data["certifications"] = [_normalize_certification(item) for item in raw_certs]

    # languages — optional list of dicts
    raw_langs = record.get("languages") or []
    data["languages"] = [_normalize_language(item) for item in raw_langs]

    # redrob_signals
    data["redrob_signals"] = _normalize_redrob_signals(record.get("redrob_signals") or {})

    return data

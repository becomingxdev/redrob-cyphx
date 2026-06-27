"""Validates candidate records against the expected schema structure."""

from __future__ import annotations

from typing import Any


# ---------- top-level required keys ----------
_TOP_LEVEL_REQUIRED: set[str] = {
    "candidate_id",
    "profile",
    "career_history",
    "education",
    "skills",
    "redrob_signals",
}

# ---------- profile ----------
_PROFILE_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "anonymized_name": str,
    "headline": str,
    "summary": str,
    "location": str,
    "country": str,
    "years_of_experience": (int, float),
    "current_title": str,
    "current_company": str,
    "current_company_size": str,
    "current_industry": str,
}

# ---------- career_history item ----------
_CAREER_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "company": str,
    "title": str,
    "start_date": str,
    "end_date": (str, type(None)),
    "duration_months": int,
    "is_current": bool,
    "industry": str,
    "company_size": str,
    "description": str,
}

# ---------- education item ----------
_EDUCATION_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "institution": str,
    "degree": str,
    "field_of_study": str,
    "start_year": int,
    "end_year": int,
    "grade": (str, type(None)),
    "tier": str,
}

# ---------- skill item ----------
_SKILL_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "name": str,
    "proficiency": str,
    "endorsements": int,
    "duration_months": (int, type(None)),
}

# ---------- certification item ----------
_CERT_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "name": str,
    "issuer": str,
    "year": int,
}

# ---------- language item ----------
_LANGUAGE_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "language": str,
    "proficiency": str,
}

# ---------- redrob_signals ----------
_REDROB_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "profile_completeness_score": (int, float),
    "signup_date": str,
    "last_active_date": str,
    "open_to_work_flag": bool,
    "profile_views_received_30d": int,
    "applications_submitted_30d": int,
    "recruiter_response_rate": (int, float),
    "avg_response_time_hours": (int, float),
    "skill_assessment_scores": dict,
    "connection_count": int,
    "endorsements_received": int,
    "notice_period_days": int,
    "expected_salary_range_inr_lpa": dict,
    "preferred_work_mode": str,
    "willing_to_relocate": bool,
    "github_activity_score": (int, float),
    "search_appearance_30d": int,
    "saved_by_recruiters_30d": int,
    "interview_completion_rate": (int, float),
    "offer_acceptance_rate": (int, float),
    "verified_email": bool,
    "verified_phone": bool,
    "linkedin_connected": bool,
}


def _check_keys_and_types(
    data: dict[str, Any],
    required: dict[str, type | tuple[type, ...]],
    prefix: str,
) -> list[str]:
    """Return a list of error strings for missing / mistyped fields."""
    errors: list[str] = []
    for key, expected_type in required.items():
        if key not in data:
            errors.append(f"{prefix}: missing required field '{key}'")
            continue
        if not isinstance(data[key], expected_type):
            errors.append(
                f"{prefix}: field '{key}' expected {expected_type}, "
                f"got {type(data[key]).__name__}"
            )
    return errors


def validate_candidate(record: dict) -> tuple[bool, list[str]]:
    """Validate a candidate record against the expected schema structure.

    Checks required fields, expected data types, list/dict types, and
    nested structures.  Does NOT enforce enum values or numeric ranges.

    Returns:
        A tuple of (is_valid, error_messages).
    """
    if not isinstance(record, dict):
        return False, ["record is not a dict"]

    errors: list[str] = []

    # --- top-level keys ---
    for key in _TOP_LEVEL_REQUIRED:
        if key not in record:
            errors.append(f"missing required field '{key}'")

    if errors:
        return False, errors

    # --- candidate_id ---
    if not isinstance(record["candidate_id"], str):
        errors.append("candidate_id must be a string")

    # --- profile ---
    profile = record["profile"]
    if not isinstance(profile, dict):
        errors.append("profile must be a dict")
    else:
        errors.extend(_check_keys_and_types(profile, _PROFILE_REQUIRED, "profile"))

    # --- career_history ---
    ch = record["career_history"]
    if not isinstance(ch, list):
        errors.append("career_history must be a list")
    else:
        for idx, item in enumerate(ch):
            prefix = f"career_history[{idx}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix}: must be a dict, got {type(item).__name__}")
                continue
            errors.extend(_check_keys_and_types(item, _CAREER_REQUIRED, prefix))

    # --- education ---
    edu = record["education"]
    if not isinstance(edu, list):
        errors.append("education must be a list")
    else:
        for idx, item in enumerate(edu):
            prefix = f"education[{idx}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix}: must be a dict, got {type(item).__name__}")
                continue
            errors.extend(_check_keys_and_types(item, _EDUCATION_REQUIRED, prefix))

    # --- skills ---
    skills = record["skills"]
    if not isinstance(skills, list):
        errors.append("skills must be a list")
    else:
        for idx, item in enumerate(skills):
            prefix = f"skills[{idx}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix}: must be a dict, got {type(item).__name__}")
                continue
            errors.extend(_check_keys_and_types(item, _SKILL_REQUIRED, prefix))

    # --- certifications (optional) ---
    certs = record.get("certifications")
    if certs is not None:
        if not isinstance(certs, list):
            errors.append("certifications must be a list")
        else:
            for idx, item in enumerate(certs):
                prefix = f"certifications[{idx}]"
                if not isinstance(item, dict):
                    errors.append(f"{prefix}: must be a dict, got {type(item).__name__}")
                    continue
                errors.extend(_check_keys_and_types(item, _CERT_REQUIRED, prefix))

    # --- languages (optional) ---
    langs = record.get("languages")
    if langs is not None:
        if not isinstance(langs, list):
            errors.append("languages must be a list")
        else:
            for idx, item in enumerate(langs):
                prefix = f"languages[{idx}]"
                if not isinstance(item, dict):
                    errors.append(f"{prefix}: must be a dict, got {type(item).__name__}")
                    continue
                errors.extend(_check_keys_and_types(item, _LANGUAGE_REQUIRED, prefix))

    # --- redrob_signals ---
    signals = record["redrob_signals"]
    if not isinstance(signals, dict):
        errors.append("redrob_signals must be a dict")
    else:
        errors.extend(_check_keys_and_types(signals, _REDROB_REQUIRED, "redrob_signals"))
        # Nested salary range
        salary = signals.get("expected_salary_range_inr_lpa")
        if isinstance(salary, dict):
            if "min" not in salary:
                errors.append("redrob_signals.expected_salary_range_inr_lpa: missing 'min'")
            elif not isinstance(salary["min"], (int, float)):
                errors.append(
                    "redrob_signals.expected_salary_range_inr_lpa.min: expected number"
                )
            if "max" not in salary:
                errors.append("redrob_signals.expected_salary_range_inr_lpa: missing 'max'")
            elif not isinstance(salary["max"], (int, float)):
                errors.append(
                    "redrob_signals.expected_salary_range_inr_lpa.max: expected number"
                )

    return (len(errors) == 0, errors)

"""Tests for src.parser.validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.parser.validator import validate_candidate

FIXTURES = Path(__file__).parent / "fixtures"


def _load_valid_candidate() -> dict:
    """Load the first valid candidate from the fixture."""
    with open(FIXTURES / "valid_candidates.jsonl", encoding="utf-8") as f:
        return json.loads(f.readline())


# ------------------------------------------------------------------ #
# Valid candidate
# ------------------------------------------------------------------ #

class TestValidCandidate:
    def test_valid_candidate_passes(self) -> None:
        record = _load_valid_candidate()
        is_valid, errors = validate_candidate(record)
        assert is_valid is True
        assert errors == []

    def test_valid_candidate_with_empty_certifications(self) -> None:
        record = _load_valid_candidate()
        record["certifications"] = []
        is_valid, errors = validate_candidate(record)
        assert is_valid is True
        assert errors == []

    def test_valid_candidate_with_empty_languages(self) -> None:
        record = _load_valid_candidate()
        record["languages"] = []
        is_valid, errors = validate_candidate(record)
        assert is_valid is True
        assert errors == []

    def test_valid_candidate_with_certifications_present(self) -> None:
        record = _load_valid_candidate()
        record["certifications"] = [
            {"name": "AWS", "issuer": "Amazon", "year": 2023}
        ]
        is_valid, errors = validate_candidate(record)
        assert is_valid is True
        assert errors == []


# ------------------------------------------------------------------ #
# Non-dict input
# ------------------------------------------------------------------ #

class TestNonDictInput:
    def test_list_input_fails(self) -> None:
        is_valid, errors = validate_candidate([1, 2, 3])
        assert is_valid is False
        assert any("not a dict" in e for e in errors)

    def test_string_input_fails(self) -> None:
        is_valid, errors = validate_candidate("not a dict")
        assert is_valid is False

    def test_none_input_fails(self) -> None:
        is_valid, errors = validate_candidate(None)
        assert is_valid is False


# ------------------------------------------------------------------ #
# Missing top-level required fields
# ------------------------------------------------------------------ #

class TestMissingTopLevelFields:
    def test_missing_candidate_id(self) -> None:
        record = _load_valid_candidate()
        del record["candidate_id"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("candidate_id" in e for e in errors)

    def test_missing_profile(self) -> None:
        record = _load_valid_candidate()
        del record["profile"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("profile" in e for e in errors)

    def test_missing_career_history(self) -> None:
        record = _load_valid_candidate()
        del record["career_history"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("career_history" in e for e in errors)

    def test_missing_education(self) -> None:
        record = _load_valid_candidate()
        del record["education"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("education" in e for e in errors)

    def test_missing_skills(self) -> None:
        record = _load_valid_candidate()
        del record["skills"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("skills" in e for e in errors)

    def test_missing_redrob_signals(self) -> None:
        record = _load_valid_candidate()
        del record["redrob_signals"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("redrob_signals" in e for e in errors)


# ------------------------------------------------------------------ #
# Wrong types
# ------------------------------------------------------------------ #

class TestWrongTypes:
    def test_candidate_id_not_string(self) -> None:
        record = _load_valid_candidate()
        record["candidate_id"] = 12345
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("candidate_id" in e for e in errors)

    def test_profile_not_dict(self) -> None:
        record = _load_valid_candidate()
        record["profile"] = "I am a string"
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("profile must be a dict" in e for e in errors)

    def test_career_history_not_list(self) -> None:
        record = _load_valid_candidate()
        record["career_history"] = {"company": "Acme"}
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("career_history must be a list" in e for e in errors)

    def test_education_not_list(self) -> None:
        record = _load_valid_candidate()
        record["education"] = "not a list"
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("education must be a list" in e for e in errors)

    def test_skills_not_list(self) -> None:
        record = _load_valid_candidate()
        record["skills"] = "not a list"
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("skills must be a list" in e for e in errors)

    def test_certifications_not_list(self) -> None:
        record = _load_valid_candidate()
        record["certifications"] = "not a list"
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("certifications must be a list" in e for e in errors)

    def test_languages_not_list(self) -> None:
        record = _load_valid_candidate()
        record["languages"] = "not a list"
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("languages must be a list" in e for e in errors)

    def test_redrob_signals_not_dict(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"] = [1, 2, 3]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("redrob_signals must be a dict" in e for e in errors)


# ------------------------------------------------------------------ #
# Profile field validation
# ------------------------------------------------------------------ #

class TestProfileFields:
    def test_missing_profile_field(self) -> None:
        record = _load_valid_candidate()
        del record["profile"]["headline"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("headline" in e for e in errors)

    def test_profile_wrong_field_type(self) -> None:
        record = _load_valid_candidate()
        record["profile"]["years_of_experience"] = "six"
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("years_of_experience" in e for e in errors)


# ------------------------------------------------------------------ #
# Nested structure validation (career, education, skills)
# ------------------------------------------------------------------ #

class TestNestedStructures:
    def test_career_item_not_dict(self) -> None:
        record = _load_valid_candidate()
        record["career_history"] = [42]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("career_history[0]" in e for e in errors)

    def test_career_item_missing_field(self) -> None:
        record = _load_valid_candidate()
        del record["career_history"][0]["title"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("title" in e for e in errors)

    def test_education_item_missing_field(self) -> None:
        record = _load_valid_candidate()
        del record["education"][0]["degree"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("degree" in e for e in errors)

    def test_skill_item_missing_field(self) -> None:
        record = _load_valid_candidate()
        del record["skills"][0]["proficiency"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("proficiency" in e for e in errors)

    def test_certification_item_missing_field(self) -> None:
        record = _load_valid_candidate()
        record["certifications"] = [{"name": "AWS"}]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("issuer" in e for e in errors)

    def test_language_item_missing_field(self) -> None:
        record = _load_valid_candidate()
        record["languages"] = [{"language": "French"}]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("proficiency" in e for e in errors)


# ------------------------------------------------------------------ #
# Redrob signals validation
# ------------------------------------------------------------------ #

class TestRedrobSignals:
    def test_missing_signal_field(self) -> None:
        record = _load_valid_candidate()
        del record["redrob_signals"]["preferred_work_mode"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("preferred_work_mode" in e for e in errors)

    def test_salary_range_missing_min(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"]["expected_salary_range_inr_lpa"] = {"max": 30.0}
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("min" in e for e in errors)

    def test_salary_range_missing_max(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 10.0}
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("max" in e for e in errors)

    def test_salary_range_wrong_type(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"]["expected_salary_range_inr_lpa"] = {
            "min": "ten", "max": 30.0
        }
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("min" in e for e in errors)

    def test_skill_assessment_scores_not_dict(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"]["skill_assessment_scores"] = [1, 2, 3]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("skill_assessment_scores" in e for e in errors)


# ------------------------------------------------------------------ #
# Error messages contain useful path info
# ------------------------------------------------------------------ #

class TestErrorMessages:
    def test_error_includes_field_path(self) -> None:
        record = _load_valid_candidate()
        del record["profile"]["current_title"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("profile:" in e for e in errors)

    def test_error_includes_indexed_path(self) -> None:
        record = _load_valid_candidate()
        del record["career_history"][0]["company"]
        is_valid, errors = validate_candidate(record)
        assert is_valid is False
        assert any("career_history[0]" in e for e in errors)

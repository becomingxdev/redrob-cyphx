"""Tests for src.parser.normalizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.parser.normalizer import normalize_candidate

FIXTURES = Path(__file__).parent / "fixtures"


def _load_valid_candidate() -> dict:
    """Load the first valid candidate from the fixture."""
    with open(FIXTURES / "valid_candidates.jsonl", encoding="utf-8") as f:
        return json.loads(f.readline())


# ------------------------------------------------------------------ #
# Whitespace stripping
# ------------------------------------------------------------------ #

class TestWhitespaceStripping:
    def test_strips_profile_strings(self) -> None:
        record = _load_valid_candidate()
        record["profile"]["headline"] = "  Backend Engineer  "
        record["profile"]["location"] = "  Toronto  "
        result = normalize_candidate(record)
        assert result["profile"]["headline"] == "Backend Engineer"
        assert result["profile"]["location"] == "Toronto"

    def test_strips_candidate_id(self) -> None:
        record = _load_valid_candidate()
        record["candidate_id"] = "  CAND_0000001  "
        result = normalize_candidate(record)
        assert result["candidate_id"] == "CAND_0000001"

    def test_strips_career_strings(self) -> None:
        record = _load_valid_candidate()
        record["career_history"][0]["company"] = "  Mindtree  "
        result = normalize_candidate(record)
        assert result["career_history"][0]["company"] == "Mindtree"

    def test_strips_skill_strings(self) -> None:
        record = _load_valid_candidate()
        record["skills"][0]["name"] = "  Python  "
        result = normalize_candidate(record)
        assert result["skills"][0]["name"] == "Python"

    def test_strips_language_strings(self) -> None:
        record = _load_valid_candidate()
        record["languages"][0]["language"] = "  English  "
        result = normalize_candidate(record)
        assert result["languages"][0]["language"] == "English"

    def test_strips_redrob_strings(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"]["preferred_work_mode"] = "  onsite  "
        result = normalize_candidate(record)
        assert result["redrob_signals"]["preferred_work_mode"] == "onsite"


# ------------------------------------------------------------------ #
# None replacement with defaults
# ------------------------------------------------------------------ #

class TestNoneReplacement:
    def test_none_candidate_id_becomes_empty_string(self) -> None:
        record = _load_valid_candidate()
        record["candidate_id"] = None
        result = normalize_candidate(record)
        assert result["candidate_id"] == ""

    def test_none_profile_field_becomes_empty_string(self) -> None:
        record = _load_valid_candidate()
        record["profile"]["headline"] = None
        result = normalize_candidate(record)
        assert result["profile"]["headline"] == ""

    def test_none_years_of_experience_becomes_zero(self) -> None:
        record = _load_valid_candidate()
        record["profile"]["years_of_experience"] = None
        result = normalize_candidate(record)
        assert result["profile"]["years_of_experience"] == 0.0

    def test_none_numeric_field_becomes_zero(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"]["connection_count"] = None
        result = normalize_candidate(record)
        assert result["redrob_signals"]["connection_count"] == 0

    def test_none_boolean_field_becomes_false(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"]["verified_email"] = None
        result = normalize_candidate(record)
        assert result["redrob_signals"]["verified_email"] is False

    def test_none_dict_field_becomes_empty_dict(self) -> None:
        record = _load_valid_candidate()
        record["redrob_signals"]["skill_assessment_scores"] = None
        result = normalize_candidate(record)
        assert result["redrob_signals"]["skill_assessment_scores"] == {}


# ------------------------------------------------------------------ #
# Empty string normalization
# ------------------------------------------------------------------ #

class TestEmptyStringNormalization:
    def test_empty_string_remains_empty(self) -> None:
        record = _load_valid_candidate()
        record["profile"]["headline"] = ""
        result = normalize_candidate(record)
        assert result["profile"]["headline"] == ""

    def test_whitespace_only_becomes_empty(self) -> None:
        record = _load_valid_candidate()
        record["profile"]["summary"] = "   "
        result = normalize_candidate(record)
        assert result["profile"]["summary"] == ""


# ------------------------------------------------------------------ #
# Missing / None list defaults
# ------------------------------------------------------------------ #

class TestListDefaults:
    def test_missing_certifications_defaults_to_empty_list(self) -> None:
        record = _load_valid_candidate()
        del record["certifications"]
        result = normalize_candidate(record)
        assert result["certifications"] == []

    def test_none_certifications_defaults_to_empty_list(self) -> None:
        record = _load_valid_candidate()
        record["certifications"] = None
        result = normalize_candidate(record)
        assert result["certifications"] == []

    def test_missing_languages_defaults_to_empty_list(self) -> None:
        record = _load_valid_candidate()
        del record["languages"]
        result = normalize_candidate(record)
        assert result["languages"] == []

    def test_none_languages_defaults_to_empty_list(self) -> None:
        record = _load_valid_candidate()
        record["languages"] = None
        result = normalize_candidate(record)
        assert result["languages"] == []


# ------------------------------------------------------------------ #
# Duplicate skill removal
# ------------------------------------------------------------------ #

class TestDuplicateSkillRemoval:
    def test_removes_duplicate_skills_by_name(self) -> None:
        record = _load_valid_candidate()
        record["skills"] = [
            {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 60},
            {"name": "Python", "proficiency": "beginner", "endorsements": 2, "duration_months": 6},
            {"name": "SQL", "proficiency": "expert", "endorsements": 5, "duration_months": 48},
        ]
        result = normalize_candidate(record)
        assert len(result["skills"]) == 2
        assert result["skills"][0]["name"] == "Python"
        assert result["skills"][0]["proficiency"] == "advanced"
        assert result["skills"][1]["name"] == "SQL"

    def test_no_duplicates_keeps_all(self) -> None:
        record = _load_valid_candidate()
        original_count = len(record["skills"])
        result = normalize_candidate(record)
        assert len(result["skills"]) == original_count


# ------------------------------------------------------------------ #
# Deterministic output
# ------------------------------------------------------------------ #

class TestDeterminism:
    def test_same_input_produces_same_output(self) -> None:
        record = _load_valid_candidate()
        result1 = normalize_candidate(record)
        result2 = normalize_candidate(record)
        assert result1 == result2

    def test_original_is_not_mutated(self) -> None:
        record = _load_valid_candidate()
        original_headline = record["profile"]["headline"]
        normalize_candidate(record)
        assert record["profile"]["headline"] == original_headline

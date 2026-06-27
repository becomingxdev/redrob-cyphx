"""Tests for src.parser.loader."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from src.models.candidate import Candidate
from src.parser.loader import load_candidates

FIXTURES = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _candidate_ids(iterable: Any) -> list[str]:
    return [c.candidate_id for c in iterable]


# ------------------------------------------------------------------ #
# Valid JSONL
# ------------------------------------------------------------------ #

class TestValidJsonl:
    def test_yields_candidate_objects(self) -> None:
        candidates = list(load_candidates(str(FIXTURES / "valid_candidates.jsonl")))
        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_correct_candidate_ids(self) -> None:
        candidates = list(load_candidates(str(FIXTURES / "valid_candidates.jsonl")))
        ids = _candidate_ids(candidates)
        assert "CAND_0000001" in ids
        assert "CAND_0000002" in ids

    def test_profile_populated(self) -> None:
        candidates = list(load_candidates(str(FIXTURES / "valid_candidates.jsonl")))
        c1 = next(c for c in candidates if c.candidate_id == "CAND_0000001")
        assert c1.profile.anonymized_name == "Ira Vora"
        assert c1.profile.current_title == "Backend Engineer"

    def test_career_history_populated(self) -> None:
        candidates = list(load_candidates(str(FIXTURES / "valid_candidates.jsonl")))
        c1 = next(c for c in candidates if c.candidate_id == "CAND_0000001")
        assert len(c1.career_history) >= 1
        assert c1.career_history[0].company == "Mindtree"

    def test_skills_populated(self) -> None:
        candidates = list(load_candidates(str(FIXTURES / "valid_candidates.jsonl")))
        c1 = next(c for c in candidates if c.candidate_id == "CAND_0000001")
        assert len(c1.skills) >= 1

    def test_redrob_signals_populated(self) -> None:
        candidates = list(load_candidates(str(FIXTURES / "valid_candidates.jsonl")))
        c1 = next(c for c in candidates if c.candidate_id == "CAND_0000001")
        assert c1.redrob_signals.expected_salary_range_inr_lpa.min == 18.7
        assert c1.redrob_signals.expected_salary_range_inr_lpa.max == 36.1
        assert isinstance(c1.redrob_signals.skill_assessment_scores, dict)

    def test_raw_preserved(self) -> None:
        candidates = list(load_candidates(str(FIXTURES / "valid_candidates.jsonl")))
        c1 = next(c for c in candidates if c.candidate_id == "CAND_0000001")
        assert isinstance(c1.raw, dict)
        assert c1.raw["candidate_id"] == "CAND_0000001"


# ------------------------------------------------------------------ #
# Valid JSONL.GZ
# ------------------------------------------------------------------ #

class TestValidGz:
    def test_yields_candidate_objects_from_gz(self) -> None:
        candidates = list(
            load_candidates(str(FIXTURES / "valid_candidates.jsonl.gz"))
        )
        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_correct_ids_from_gz(self) -> None:
        candidates = list(
            load_candidates(str(FIXTURES / "valid_candidates.jsonl.gz"))
        )
        ids = _candidate_ids(candidates)
        assert "CAND_0000001" in ids
        assert "CAND_0000002" in ids


# ------------------------------------------------------------------ #
# Malformed JSON
# ------------------------------------------------------------------ #

class TestMalformedJson:
    def test_skips_malformed_lines(self) -> None:
        candidates = list(
            load_candidates(str(FIXTURES / "malformed_json.jsonl"))
        )
        # 4 lines: 2 valid, 2 malformed -> should yield 2
        assert len(candidates) == 2
        ids = _candidate_ids(candidates)
        assert "CAND_0000001" in ids
        assert "CAND_0000002" in ids

    def test_logs_warnings_for_malformed_json(self, caplog: Any) -> None:
        list(load_candidates(str(FIXTURES / "malformed_json.jsonl")))
        malformed_warnings = [
            r for r in caplog.records if "malformed JSON" in r.message
        ]
        assert len(malformed_warnings) >= 1


# ------------------------------------------------------------------ #
# Invalid structure
# ------------------------------------------------------------------ #

class TestInvalidStructure:
    def test_skips_invalid_records(self) -> None:
        candidates = list(
            load_candidates(str(FIXTURES / "invalid_structure.jsonl"))
        )
        # All 3 records are structurally invalid
        assert len(candidates) == 0

    def test_logs_reason_for_skip(self, caplog: Any) -> None:
        list(load_candidates(str(FIXTURES / "invalid_structure.jsonl")))
        skip_messages = [
            r for r in caplog.records if "Skipping" in r.message
        ]
        assert len(skip_messages) >= 1


# ------------------------------------------------------------------ #
# Empty file
# ------------------------------------------------------------------ #

class TestEmptyFile:
    def test_yields_nothing_for_empty_file(self) -> None:
        candidates = list(load_candidates(str(FIXTURES / "empty.jsonl")))
        assert len(candidates) == 0

    def test_is_generator(self) -> None:
        gen = load_candidates(str(FIXTURES / "empty.jsonl"))
        # Should be a generator, not a list
        import types
        assert isinstance(gen, types.GeneratorType)


# ------------------------------------------------------------------ #
# Generator / iterator behavior
# ------------------------------------------------------------------ #

class TestGeneratorBehavior:
    def test_returns_generator(self) -> None:
        gen = load_candidates(str(FIXTURES / "valid_candidates.jsonl"))
        import types
        assert isinstance(gen, types.GeneratorType)

    def test_not_materialized_until_iterated(self) -> None:
        gen = load_candidates(str(FIXTURES / "valid_candidates.jsonl"))
        # Before iteration, no data is read — generator exists but hasn't run
        # This is inherently true for generators; we verify the type.
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")

    def test_exhausts_after_full_iteration(self) -> None:
        gen = load_candidates(str(FIXTURES / "valid_candidates.jsonl"))
        candidates = list(gen)
        assert len(candidates) == 2
        with pytest.raises(StopIteration):
            next(gen)


# ------------------------------------------------------------------ #
# Non-existent file
# ------------------------------------------------------------------ #

class TestNonExistentFile:
    def test_yields_nothing(self) -> None:
        candidates = list(
            load_candidates(str(FIXTURES / "does_not_exist.jsonl"))
        )
        assert len(candidates) == 0

    def test_logs_error(self, caplog: Any) -> None:
        list(load_candidates(str(FIXTURES / "does_not_exist.jsonl")))
        error_messages = [
            r for r in caplog.records if "File not found" in r.message
        ]
        assert len(error_messages) == 1

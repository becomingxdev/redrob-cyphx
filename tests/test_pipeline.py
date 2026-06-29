"""End-to-end pipeline tests.

Tests the full flow from Candidate → FinalScore → RankedEntry → Reason → CSV
without touching the real dataset (uses fixtures).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.features.title import TitleExtractor
from src.features.skills import SkillsExtractor
from src.features.experience import ExperienceExtractor
from src.features.education import EducationExtractor
from src.features.career import CareerExtractor

from src.evidence.verifier import EvidenceVerifier
from src.evidence.consistency import ConsistencyAnalyzer
from src.evidence.confidence import ConfidenceCalculator

from src.scoring.title_score import score_title
from src.scoring.skill_score import score_skills
from src.scoring.experience_score import score_experience
from src.scoring.education_score import score_education
from src.scoring.behavior_score import score_behavior
from src.scoring.penalties import apply_penalties
from src.scoring.honeypot import detect_honeypot
from src.scoring.composite import compose_score

from src.output.ranking import rank_candidates
from src.reasoning.generator import generate_reason
from src.output.csv_writer import write_submission_csv

from tests.conftest import make_candidate, make_empty_candidate, _make_skill, _make_career


def _run_full_pipeline(candidate) -> tuple:
    """Run the entire per-candidate pipeline and return ranked + reasons.

    Returns:
        (RankedEntry, Reason) for the single candidate.
    """
    # Feature Extraction
    extracted = {
        "title": TitleExtractor().extract(candidate),
        "skills": SkillsExtractor().extract(candidate),
        "experience": ExperienceExtractor().extract(candidate),
        "education": EducationExtractor().extract(candidate),
        "career": CareerExtractor().extract(candidate),
    }

    # Evidence
    verifier = EvidenceVerifier()
    consistency_analyzer = ConsistencyAnalyzer()
    confidence_calc = ConfidenceCalculator()

    evidence = verifier.verify(candidate)
    consistency = consistency_analyzer.analyze(candidate)
    confidence = confidence_calc.calculate(
        candidate=candidate, evidence=evidence, consistency=consistency,
    )

    # Scores
    title = score_title(candidate, extracted["title"])
    skills = score_skills(candidate, extracted["skills"])
    exp = score_experience(candidate, extracted["experience"], extracted["career"])
    edu = score_education(
        candidate, extracted["education"],
        candidate_skills=set(extracted["skills"].get("skills", [])),
    )
    beh = score_behavior(candidate)
    penalty = apply_penalties(
        candidate=candidate,
        title_features=extracted["title"],
        experience_features=extracted["experience"],
        education_features=extracted["education"],
        career_features=extracted["career"],
        evidence_result=evidence,
        consistency_result=consistency,
    )
    honeypot = detect_honeypot(
        candidate=candidate,
        experience_features=extracted["experience"],
        education_features=extracted["education"],
        title_features=extracted["title"],
        evidence_result=evidence,
        consistency_result=consistency,
    )

    # Composite
    final = compose_score(
        candidate_id=candidate.candidate_id,
        title_score=title,
        skill_score=skills,
        experience_score=exp,
        education_score=edu,
        behavior_score=beh,
        penalty=penalty,
        confidence_result=confidence,
        consistency_result=consistency,
        honeypot=honeypot,
    )

    comp_results = {
        "title": title, "skills": skills, "experience": exp,
        "education": edu, "behavior": beh,
    }

    # Rank
    ranked = rank_candidates([final])

    # Reason
    reason = generate_reason(final, comp_results)

    return ranked[0], reason, final


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_single_candidate_pipeline(self):
        """A single normal candidate should produce valid ranked entry + reason."""
        candidate = make_candidate()
        entry, reason, final = _run_full_pipeline(candidate)

        assert entry.rank == 1
        assert 0.0 <= final.score <= 100.0
        assert len(reason.reasons) <= 5
        assert reason.candidate_id == candidate.candidate_id

    def test_empty_candidate_pipeline(self):
        """Empty candidate should produce a low but valid score."""
        candidate = make_empty_candidate()
        entry, reason, final = _run_full_pipeline(candidate)

        assert entry.rank == 1
        assert 0.0 <= final.score <= 100.0

    def test_csv_output_single(self, tmp_path):
        """Single candidate should produce a valid 2-row CSV (header + 1 data)."""
        candidate = make_candidate()
        entry, reason, _ = _run_full_pipeline(candidate)

        out = tmp_path / "sub.csv"
        write_submission_csv(out, [entry], {candidate.candidate_id: reason})

        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 2  # header + 1 data row
        assert rows[0][0] == "candidate_id"
        assert rows[1][0] == candidate.candidate_id
        assert rows[1][1] == "1"

    def test_csv_output_multiple(self, tmp_path):
        """Three candidates should produce a 4-row CSV with correct ordering."""
        candidates = [
            make_candidate(candidate_id="CAND_A"),
            make_candidate(candidate_id="CAND_B"),
            make_candidate(candidate_id="CAND_C"),
        ]

        finals = []
        comp_all = {}
        for c in candidates:
            _, _, final, comp = _run_full_pipeline_detailed(c)
            finals.append(final)
            comp_all[c.candidate_id] = comp

        ranked = rank_candidates(finals)
        reasons = {
            entry.candidate_id: generate_reason(entry.final, comp_all[entry.candidate_id])
            for entry in ranked
        }

        out = tmp_path / "sub.csv"
        write_submission_csv(out, ranked, reasons)

        with open(out, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 4
        # Verify ranks are 1, 2, 3.
        ranks = [int(rows[i + 1][1]) for i in range(3)]
        assert sorted(ranks) == [1, 2, 3]

    def test_deterministic_pipeline(self):
        """Running the pipeline twice on the same candidate must be identical."""
        candidate = make_candidate()

        _, reason1, final1 = _run_full_pipeline(candidate)
        _, reason2, final2 = _run_full_pipeline(candidate)

        assert final1.score == final2.score
        assert reason1.reasons == reason2.reasons

    def test_multiple_candidates_ranks_unique(self):
        """Ranking multiple candidates should assign unique ranks."""
        candidates = [
            make_candidate(candidate_id=f"CAND_{i:07d}",
                          profile={"years_of_experience": float(i)})
            for i in range(5)
        ]

        finals = []
        for c in candidates:
            _, _, final, _ = _run_full_pipeline_detailed(c)
            finals.append(final)

        ranked = rank_candidates(finals)
        ranks = [e.rank for e in ranked]
        assert sorted(ranks) == [1, 2, 3, 4, 5]

    def test_score_valid_range(self):
        """Pipeline should never produce scores outside [0, 100]."""
        candidates = [
            make_candidate(candidate_id=f"CAND_{i:07d}")
            for i in range(10)
        ]

        finals = []
        for c in candidates:
            _, _, final, _ = _run_full_pipeline_detailed(c)
            finals.append(final)

        for f in finals:
            assert 0.0 <= f.score <= 100.0

    def test_large_sample_performance(self):
        """Pipeline should handle 100 synthetic candidates in reasonable time."""
        import time

        candidates = [
            make_candidate(
                candidate_id=f"CAND_{i:07d}",
                profile={"years_of_experience": float(i % 15)},
            )
            for i in range(100)
        ]

        start = time.time()
        finals = []
        comp_all = {}
        for c in candidates:
            _, _, final, comp = _run_full_pipeline_detailed(c)
            finals.append(final)
            comp_all[c.candidate_id] = comp

        ranked = rank_candidates(finals)
        for entry in ranked:
            generate_reason(entry.final, comp_all[entry.candidate_id])

        elapsed = time.time() - start
        assert len(ranked) == 100
        # Must complete 100 candidates in under 30 seconds (generous).
        assert elapsed < 30.0


def _run_full_pipeline_detailed(candidate) -> tuple:
    """Run the full pipeline returning all intermediate results.

    Returns:
        (RankedEntry, Reason, FinalScore, component_results)
    """
    extracted = {
        "title": TitleExtractor().extract(candidate),
        "skills": SkillsExtractor().extract(candidate),
        "experience": ExperienceExtractor().extract(candidate),
        "education": EducationExtractor().extract(candidate),
        "career": CareerExtractor().extract(candidate),
    }

    verifier = EvidenceVerifier()
    consistency_analyzer = ConsistencyAnalyzer()
    confidence_calc = ConfidenceCalculator()

    evidence = verifier.verify(candidate)
    consistency = consistency_analyzer.analyze(candidate)
    confidence = confidence_calc.calculate(
        candidate=candidate, evidence=evidence, consistency=consistency,
    )

    title = score_title(candidate, extracted["title"])
    skills = score_skills(candidate, extracted["skills"])
    exp = score_experience(candidate, extracted["experience"], extracted["career"])
    edu = score_education(
        candidate, extracted["education"],
        candidate_skills=set(extracted["skills"].get("skills", [])),
    )
    beh = score_behavior(candidate)
    penalty = apply_penalties(
        candidate=candidate,
        title_features=extracted["title"],
        experience_features=extracted["experience"],
        education_features=extracted["education"],
        career_features=extracted["career"],
        evidence_result=evidence,
        consistency_result=consistency,
    )
    honeypot = detect_honeypot(
        candidate=candidate,
        experience_features=extracted["experience"],
        education_features=extracted["education"],
        title_features=extracted["title"],
        evidence_result=evidence,
        consistency_result=consistency,
    )

    final = compose_score(
        candidate_id=candidate.candidate_id,
        title_score=title, skill_score=skills, experience_score=exp,
        education_score=edu, behavior_score=beh,
        penalty=penalty, confidence_result=confidence,
        consistency_result=consistency, honeypot=honeypot,
    )

    comp_results = {
        "title": title, "skills": skills, "experience": exp,
        "education": edu, "behavior": beh,
    }

    ranked = rank_candidates([final])
    reason = generate_reason(final, comp_results)

    return ranked[0], reason, final, comp_results

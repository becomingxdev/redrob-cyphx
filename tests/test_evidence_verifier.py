import copy

import pytest

from src.evidence.verifier import EvidenceVerifier, SOURCE_NAMES
from src.models.factory import CandidateFactory


@pytest.fixture
def base_candidate_dict() -> dict:
    """Minimal but fully-populated candidate dict (mirrors feature-test fixtures)."""
    return {
        "candidate_id": "CAND_EVID_01",
        "profile": {
            "anonymized_name": "Evidence Tester",
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

def _make_skill(name: str, proficiency: str = "intermediate") -> dict:
    return {
        "name": name,
        "proficiency": proficiency,
        "endorsements": 1,
        "duration_months": 12,
    }


def _make_role(
    company: str,
    title: str,
    description: str = "",
    industry: str = "Tech",
) -> dict:
    return {
        "company": company,
        "title": title,
        "start_date": "2020-01-01",
        "end_date": "2021-01-01",
        "duration_months": 12,
        "is_current": False,
        "industry": industry,
        "company_size": "100-500",
        "description": description,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVerifierStructure:
    """Output shape and basic structural guarantees."""

    def test_source_names_constant(self) -> None:
        """SOURCE_NAMES must be the documented, ordered set of sources."""
        assert SOURCE_NAMES == ("skills", "experience", "career_history", "summary", "projects")

    def test_empty_candidate_returns_empty_evidence(self, base_candidate_dict) -> None:
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence == {}

    def test_each_feature_has_all_source_keys(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)

        assert "python" in evidence
        for source in SOURCE_NAMES:
            assert source in evidence["python"]
            assert isinstance(evidence["python"][source], bool)
        assert "source_support_count" in evidence["python"]


class TestVerifierSkillsSource:
    """The 'skills' source reflects the explicit skills list."""

    def test_skill_always_supported_by_skills_source(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)

        assert evidence["python"]["skills"] is True
        # No text evidence provided, so other sources should be False.
        assert evidence["python"]["experience"] is False
        assert evidence["python"]["summary"] is False

    def test_skill_names_are_normalized_lowercased(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("  PyThOn  ")]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert "python" in evidence

    def test_duplicate_skills_deduplicated(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [
            _make_skill("Python"), _make_skill("python"), _make_skill("PYTHON"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert list(evidence.keys()) == ["python"]


class TestVerifierTextSources:
    """Experience, career_history, and summary free-text matching."""

    def test_skill_in_experience_description(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        base_candidate_dict["career_history"] = [
            _make_role("Acme", "Engineer", description="Built services in Python and Go.")
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)

        assert evidence["python"]["experience"] is True

    def test_skill_in_experience_via_title(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Spark")]
        base_candidate_dict["career_history"] = [
            _make_role("Acme", "Spark Engineer", description="")
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["spark"]["experience"] is True

    def test_skill_in_career_history_company(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Acme")]
        base_candidate_dict["career_history"] = [_make_role("Acme", "Engineer")]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["acme"]["career_history"] is True

    def test_skill_in_summary(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        base_candidate_dict["profile"]["summary"] = "I love coding in Python."
        base_candidate_dict["profile"]["headline"] = "Python developer"
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["python"]["summary"] is True

    def test_summary_checks_both_headline_and_summary(self, base_candidate_dict) -> None:
        """Skill in only the headline (not summary) should still match 'summary' source."""
        base_candidate_dict["skills"] = [_make_skill("Rust")]
        base_candidate_dict["profile"]["headline"] = "Rust enthusiast"
        base_candidate_dict["profile"]["summary"] = "Nothing relevant here."
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["rust"]["summary"] is True

    def test_no_false_positive_substring(self, base_candidate_dict) -> None:
        """'sql' must NOT match inside 'mysql' (whole-word matching)."""
        base_candidate_dict["skills"] = [_make_skill("SQL")]
        base_candidate_dict["profile"]["summary"] = "I worked with mysql databases."
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["sql"]["summary"] is False

    def test_multi_token_skill_matches_as_phrase(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Machine Learning")]
        base_candidate_dict["profile"]["summary"] = "Experience in machine learning pipelines."
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert "machine learning" in evidence
        assert evidence["machine learning"]["summary"] is True

    def test_multi_token_skill_newline_resilience(self, base_candidate_dict) -> None:
        """A multi-token feature separated by a newline in the blob must still match."""
        base_candidate_dict["skills"] = [_make_skill("Machine Learning")]
        base_candidate_dict["profile"]["summary"] = "Worked on machine\nlearning models."
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["machine learning"]["summary"] is True

    def test_special_characters_word_boundary(self, base_candidate_dict) -> None:
        """'c#' and 'c++' must match correctly as whole words and not fail due to trailing non-alphanumeric chars."""
        base_candidate_dict["skills"] = [_make_skill("C#"), _make_skill("C++")]
        base_candidate_dict["profile"]["summary"] = "Experienced in C# development and C++ systems."
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["c#"]["summary"] is True
        assert evidence["c++"]["summary"] is True


class TestVerifierProjectsSource:
    """The 'projects' source is read defensively from candidate.raw."""

    def test_projects_absent_defaults_false(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["python"]["projects"] is False

    def test_projects_as_string_list(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        base_candidate_dict["projects"] = ["A Python web scraper"]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["python"]["projects"] is True

    def test_projects_as_dict_list(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Docker")]
        base_candidate_dict["projects"] = [
            {"name": "CI tool", "description": "Containers with docker", "tech": ["docker"]}
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        assert evidence["docker"]["projects"] is True


class TestVerifierSupportCount:
    """source_support_count reflects the number of True sources."""

    def test_support_count_counts_independent_sources(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        base_candidate_dict["profile"]["summary"] = "Python developer"
        base_candidate_dict["profile"]["headline"] = "Python pro"
        base_candidate_dict["career_history"] = [
            _make_role("Acme", "Python Engineer", description="Python services.")
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)

        # skills + experience + career_history + summary = 4 sources
        assert evidence["python"]["source_support_count"] == 4

    def test_support_count_zero_when_only_skills(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)
        # Only the skills source supports it => 1 supporting source.
        assert evidence["python"]["source_support_count"] == 1


class TestVerifierDeterminism:
    """The verifier must be deterministic."""

    def test_same_input_same_output(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [
            _make_skill("Python"), _make_skill("SQL"), _make_skill("Spark"),
        ]
        base_candidate_dict["profile"]["summary"] = "Python and SQL data work."
        base_candidate_dict["career_history"] = [
            _make_role("Acme", "Engineer", description="Spark pipelines."),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        verifier = EvidenceVerifier()
        assert verifier.verify(candidate) == verifier.verify(candidate)

    def test_does_not_mutate_candidate(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        base_candidate_dict["projects"] = ["Python tool"]
        candidate = CandidateFactory.create(base_candidate_dict)
        raw_before = copy.deepcopy(candidate.raw)
        EvidenceVerifier().verify(candidate)
        assert candidate.raw == raw_before

import pytest
from src.models.factory import CandidateFactory
from src.features.education import EducationExtractor


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


class TestEducationExtraction:
    """Original tests — highest degree selection and empty handling."""

    def test_education_extraction(self, base_candidate_dict) -> None:
        base_candidate_dict["education"] = [
            _make_edu("State University", "B.S.", "Computer Science", 2015, 2019, "tier_2"),
            _make_edu("Elite Tech", "M.S.", "Machine Learning", 2019, 2021, "tier_1"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = EducationExtractor()
        features = extractor.extract(candidate)

        assert features["highest_degree"] == "M.S."
        assert features["field_of_study"] == "Machine Learning"
        assert features["education_tier"] == "tier_1"
        assert features["graduation_year"] == 2021
        assert features["education_count"] == 2

    def test_education_extraction_empty(self, base_candidate_dict) -> None:
        base_candidate_dict["education"] = []
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = EducationExtractor()
        features = extractor.extract(candidate)

        assert features["highest_degree"] == ""
        assert features["field_of_study"] == ""
        assert features["education_tier"] == ""
        assert features["graduation_year"] == 0
        assert features["education_count"] == 0


class TestDegreeHierarchy:
    """Verify that _degree_score correctly classifies degree levels
    and that the extractor returns factual information — no scoring weights."""

    def test_phd_selected_over_masters(self, base_candidate_dict) -> None:
        base_candidate_dict["education"] = [
            _make_edu("State U", "M.S.", "Physics", 2010, 2012, "tier_2"),
            _make_edu("Elite U", "PhD", "AI", 2012, 2016, "tier_1"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        assert features["highest_degree"] == "PhD"
        assert features["education_tier"] == "tier_1"
        assert features["graduation_year"] == 2016

    def test_masters_selected_over_bachelors(self, base_candidate_dict) -> None:
        base_candidate_dict["education"] = [
            _make_edu("State U", "B.S.", "CS", 2012, 2016, "tier_2"),
            _make_edu("Elite U", "MBA", "Business", 2016, 2018, "tier_1"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        assert features["highest_degree"] == "MBA"
        assert features["field_of_study"] == "Business"

    def test_bachelors_selected_over_diploma(self, base_candidate_dict) -> None:
        base_candidate_dict["education"] = [
            _make_edu("College", "Diploma", "IT", 2010, 2012, "tier_3"),
            _make_edu("University", "B.Sc", "Mathematics", 2012, 2016, "tier_2"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        assert features["highest_degree"] == "B.Sc"
        assert features["education_count"] == 2

    def test_single_education(self, base_candidate_dict) -> None:
        base_candidate_dict["education"] = [
            _make_edu("Uni", "B.Tech", "ECE", 2014, 2018, "tier_2"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        assert features["highest_degree"] == "B.Tech"
        assert features["field_of_study"] == "ECE"
        assert features["education_count"] == 1

    def test_school_level(self, base_candidate_dict) -> None:
        base_candidate_dict["education"] = [
            _make_edu("High School", "High School", "General", 2010, 2012, "unknown"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        assert features["highest_degree"] == "High School"
        assert features["education_tier"] == "unknown"
        assert features["graduation_year"] == 2012


class TestDegreeTiebreaker:
    """When two degrees have the same level, the later graduation year wins."""

    def test_tie_break_by_graduation_year(self, base_candidate_dict) -> None:
        """Two M.S. degrees — the one with the later end_year should be selected."""
        base_candidate_dict["education"] = [
            _make_edu("Uni A", "M.S.", "Physics", 2015, 2017, "tier_2"),
            _make_edu("Uni B", "M.S.", "Computer Science", 2018, 2020, "tier_1"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        assert features["highest_degree"] == "M.S."
        assert features["field_of_study"] == "Computer Science"
        assert features["graduation_year"] == 2020
        assert features["education_tier"] == "tier_1"


class TestNoHiddenScoring:
    """Verify the extractor returns only factual data — no numerical scores or ranks."""

    def test_output_keys_are_factual(self, base_candidate_dict) -> None:
        """Returned keys must be factual metadata, not score/weight/rank."""
        base_candidate_dict["education"] = [
            _make_edu("Uni", "B.S.", "CS", 2014, 2018, "tier_1"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        expected_keys = {"highest_degree", "field_of_study", "education_tier",
                          "graduation_year", "education_count"}
        assert set(features.keys()) == expected_keys

        # No scoring-related keys
        for key in features:
            assert not any(term in key.lower() for term in ("score", "weight", "rank", "points")), (
                f"Key '{key}' sounds like scoring logic, not factual data"
            )

    def test_no_numeric_score_in_output(self, base_candidate_dict) -> None:
        """The internal _degree_score must never appear in the returned dict."""
        base_candidate_dict["education"] = [
            _make_edu("Uni", "PhD", "ML", 2016, 2020, "tier_1"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        assert "degree_score" not in features
        assert "score" not in features


class TestSubstringBugPrevention:
    """Regression tests for the substring-matching bug fixed in _degree_score.

    Previously 'degree' matched 'be' (B.E.) because of naive `in` checks.
    Now regex word-boundary patterns prevent false positives.
    """

    def test_degree_string_does_not_match_be(self, base_candidate_dict) -> None:
        """The word 'degree' should not be classified as B.E. (Bachelor)."""
        base_candidate_dict["education"] = [
            _make_edu("Uni", "degree", "General", 2014, 2018, "unknown"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        # "degree" should not match any known degree pattern → score 0
        assert features["highest_degree"] == "degree"
        assert features["education_tier"] == "unknown"

    def test_diploma_does_not_match_ba(self, base_candidate_dict) -> None:
        """'diploma' must classify as diploma (2), not bachelors via 'ba'."""
        extractor = EducationExtractor()
        score = extractor._degree_score("diploma")

        assert score == 2  # diploma, not 3 (bachelors)

    def test_be_matches_bachelor(self, base_candidate_dict) -> None:
        """'B.E.' must classify as bachelors (3)."""
        extractor = EducationExtractor()
        assert extractor._degree_score("B.E.") == 3

    def test_bsc_matches_bachelor(self, base_candidate_dict) -> None:
        """'B.Sc' must classify as bachelors (3)."""
        extractor = EducationExtractor()
        assert extractor._degree_score("B.Sc") == 3

    def test_unknown_degree_string(self, base_candidate_dict) -> None:
        """An unrecognizable degree string should return score 0 and still be reported."""
        base_candidate_dict["education"] = [
            _make_edu("Uni", "Certification", "Cloud", 2020, 2020, "unknown"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = EducationExtractor().extract(candidate)

        assert features["highest_degree"] == "Certification"
        assert features["education_count"] == 1


class TestEducationDeterminism:
    """Verify the extractor is deterministic."""

    def test_same_input_same_output(self, base_candidate_dict) -> None:
        base_candidate_dict["education"] = [
            _make_edu("Uni A", "B.S.", "CS", 2012, 2016, "tier_2"),
            _make_edu("Uni B", "M.S.", "AI", 2016, 2018, "tier_1"),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = EducationExtractor()

        assert extractor.extract(candidate) == extractor.extract(candidate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edu(
    institution: str,
    degree: str,
    field: str,
    start_year: int,
    end_year: int,
    tier: str,
) -> dict:
    """Create a minimal education entry dict."""
    return {
        "institution": institution,
        "degree": degree,
        "field_of_study": field,
        "start_year": start_year,
        "end_year": end_year,
        "grade": None,
        "tier": tier,
    }

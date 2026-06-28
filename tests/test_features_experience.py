import pytest
from datetime import date
from src.models.factory import CandidateFactory
from src.features.experience import ExperienceExtractor


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
            "years_of_experience": 8.5,
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
            "last_active_date": "2025-07-01",
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


class TestExperienceMultiplePositions:
    """Verify that total_positions, total_months_worked, current_tenure,
    average_tenure, and longest_tenure reflect ALL career history entries."""

    def test_total_positions_counts_all_roles(self, base_candidate_dict) -> None:
        """total_positions must count every career_history entry."""
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2021-01-01", 12, False),
            _make_role("B", "2021-02-01", "2022-02-01", 12, False),
            _make_role("C", "2022-03-01", "2023-03-01", 12, False),
            _make_role("D", "2023-04-01", None, 6, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor().extract(candidate)

        assert features["total_positions"] == 4

    def test_total_months_worked_sums_all_durations(self, base_candidate_dict) -> None:
        """total_months_worked must sum duration_months of every role."""
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2020-07-01", 6, False),
            _make_role("B", "2020-08-01", "2021-04-01", 9, False),
            _make_role("C", "2021-05-01", None, 15, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor().extract(candidate)

        assert features["total_months_worked"] == 6 + 9 + 15  # 30

    def test_current_tenure_is_current_role_only(self, base_candidate_dict) -> None:
        """current_tenure must come from the is_current=True role only."""
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2022-01-01", 24, False),
            _make_role("B", "2022-02-01", "2024-02-01", 24, False),
            _make_role("C", "2024-03-01", None, 18, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor().extract(candidate)

        assert features["current_tenure"] == 18

    def test_average_tenure_across_all_roles(self, base_candidate_dict) -> None:
        """average_tenure is total_months_worked / total_positions."""
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2021-01-01", 12, False),
            _make_role("B", "2021-02-01", "2023-02-01", 24, False),
            _make_role("C", "2023-03-01", None, 6, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor().extract(candidate)

        # (12 + 24 + 6) / 3 = 14.0
        assert features["average_tenure"] == 14.0

    def test_longest_tenure_across_all_roles(self, base_candidate_dict) -> None:
        """longest_tenure must be the max duration across all roles."""
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2021-06-01", 18, False),
            _make_role("B", "2021-07-01", "2022-01-01", 6, False),
            _make_role("C", "2022-02-01", None, 10, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor().extract(candidate)

        assert features["longest_tenure"] == 18

    def test_no_current_role_zero_current_tenure(self, base_candidate_dict) -> None:
        """When no role has is_current=True, current_tenure must be 0."""
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2022-01-01", 24, False),
            _make_role("B", "2022-02-01", "2024-02-01", 24, False),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor().extract(candidate)

        assert features["current_tenure"] == 0
        assert features["total_positions"] == 2
        assert features["total_months_worked"] == 48
        assert features["longest_tenure"] == 24


class TestExperienceExtraction:
    """Tests for the original behavior with gaps and empty cases."""

    def test_experience_extraction(self, base_candidate_dict) -> None:
        candidate = CandidateFactory.create(
            _with_history(base_candidate_dict, [
                _make_role("A", "2020-01-01", "2022-01-01", 24, False),
                _make_role("B", "2022-06-01", "2024-06-01", 24, False),
                _make_role("C", "2024-07-01", None, 12, True),
            ])
        )
        extractor = ExperienceExtractor()
        features = extractor.extract(candidate)

        assert features["years_of_experience"] == 8.5
        assert features["total_positions"] == 3
        assert features["current_tenure"] == 12
        assert features["average_tenure"] == 20.0
        assert features["longest_tenure"] == 24
        # Gap between 2022-01-01 and 2022-06-01 is ~5 months
        # Gap between 2024-06-01 and 2024-07-01 is ~1 month
        assert features["employment_gaps_months"] > 4.5
        assert features["total_months_worked"] == 60

    def test_experience_extraction_empty(self, base_candidate_dict) -> None:
        base_candidate_dict["career_history"] = []
        base_candidate_dict["profile"]["years_of_experience"] = 0.0
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = ExperienceExtractor()
        features = extractor.extract(candidate)

        assert features["years_of_experience"] == 0.0
        assert features["total_positions"] == 0
        assert features["current_tenure"] == 0
        assert features["average_tenure"] == 0.0
        assert features["longest_tenure"] == 0
        assert features["employment_gaps_months"] == 0.0
        assert features["total_months_worked"] == 0


class TestExperienceGaps:
    """Verify employment gap calculation with a fixed reference_date."""

    def test_no_gaps_consecutive_roles(self, base_candidate_dict) -> None:
        """Consecutive roles with no gap should produce 0 gap months."""
        ref = date(2025, 1, 1)
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2022-01-01", 24, False),
            _make_role("B", "2022-01-01", "2024-01-01", 24, False),
            _make_role("C", "2024-01-01", None, 12, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor(reference_date=ref).extract(candidate)

        assert features["employment_gaps_months"] == 0.0

    def test_single_role_no_gaps(self, base_candidate_dict) -> None:
        """A single role should always have 0 gap months."""
        ref = date(2025, 6, 1)
        base_candidate_dict["career_history"] = [
            _make_role("A", "2023-01-01", None, 30, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor(reference_date=ref).extract(candidate)

        assert features["employment_gaps_months"] == 0.0
        assert features["total_positions"] == 1

    def test_gap_between_two_past_roles(self, base_candidate_dict) -> None:
        """Exact gap: role A ends 2022-01-01, role B starts 2022-04-01 = ~3 months."""
        ref = date(2025, 1, 1)
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2022-01-01", 24, False),
            _make_role("B", "2022-04-01", "2024-04-01", 24, False),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor(reference_date=ref).extract(candidate)

        # 90 days / 30.44 ≈ 2.96
        assert 2.9 < features["employment_gaps_months"] < 3.1


class TestExperienceReferenceDate:
    """Verify that the reference_date parameter works correctly."""

    def test_fixed_reference_date_used_for_current_role(self, base_candidate_dict) -> None:
        """When reference_date is set, it must be used instead of datetime.now()."""
        ref = date(2025, 3, 15)
        base_candidate_dict["career_history"] = [
            _make_role("A", "2024-01-01", "2024-12-01", 11, False),
            _make_role("B", "2025-01-01", None, 2, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor(reference_date=ref).extract(candidate)

        # Gap between 2024-12-01 and 2025-01-01 = ~1 month
        assert 0.9 < features["employment_gaps_months"] < 1.1

    def test_reference_date_defaults_to_last_active(self, base_candidate_dict) -> None:
        """Without explicit reference_date, last_active_date should be used."""
        base_candidate_dict["redrob_signals"]["last_active_date"] = "2025-06-01"
        base_candidate_dict["career_history"] = [
            _make_role("A", "2024-01-01", "2025-01-01", 12, False),
            _make_role("B", "2025-02-01", None, 4, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        # No reference_date passed — falls back to last_active_date
        features = ExperienceExtractor().extract(candidate)

        # Gap between 2025-01-01 and 2025-02-01 = ~1 month
        assert 0.9 < features["employment_gaps_months"] < 1.1

    def test_reference_date_no_effect_on_tenure_stats(self, base_candidate_dict) -> None:
        """Changing reference_date must NOT affect total_months_worked,
        current_tenure, average_tenure, longest_tenure, total_positions."""
        ref_early = date(2024, 1, 1)
        ref_late = date(2026, 1, 1)

        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2022-06-01", 30, False),
            _make_role("B", "2022-08-01", None, 18, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)

        f_early = ExperienceExtractor(reference_date=ref_early).extract(candidate)
        f_late = ExperienceExtractor(reference_date=ref_late).extract(candidate)

        # These values are computed from duration_months, not dates
        for key in ("total_positions", "total_months_worked", "current_tenure",
                     "average_tenure", "longest_tenure"):
            assert f_early[key] == f_late[key], (
                f"{key} should not change with reference_date: "
                f"{f_early[key]} vs {f_late[key]}"
            )


class TestExperienceYearsOfExperience:
    """Verify years_of_experience comes from profile, not calculated."""

    def test_yoe_from_profile_field(self, base_candidate_dict) -> None:
        """years_of_experience must be the raw value from the profile."""
        base_candidate_dict["profile"]["years_of_experience"] = 12.75
        base_candidate_dict["career_history"] = [
            _make_role("A", "2022-01-01", None, 36, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = ExperienceExtractor().extract(candidate)

        assert features["years_of_experience"] == 12.75


class TestExperienceDeterminism:
    """Verify the extractor is deterministic."""

    def test_same_input_same_output(self, base_candidate_dict) -> None:
        """Running extract twice on the same candidate must produce identical results."""
        ref = date(2025, 5, 1)
        base_candidate_dict["career_history"] = [
            _make_role("A", "2020-01-01", "2022-01-01", 24, False),
            _make_role("B", "2022-03-01", None, 36, True),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = ExperienceExtractor(reference_date=ref)

        result1 = extractor.extract(candidate)
        result2 = extractor.extract(candidate)

        assert result1 == result2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_role(
    company: str,
    start_date: str,
    end_date: str | None,
    duration_months: int,
    is_current: bool,
) -> dict:
    """Create a minimal career_history entry dict."""
    return {
        "company": company,
        "title": "Engineer",
        "start_date": start_date,
        "end_date": end_date,
        "duration_months": duration_months,
        "is_current": is_current,
        "industry": "Tech",
        "company_size": "100-500",
        "description": "",
    }


def _with_history(base: dict, history: list[dict]) -> dict:
    """Return a copy of base_candidate_dict with career_history replaced."""
    d = dict(base)
    d["career_history"] = history
    return d

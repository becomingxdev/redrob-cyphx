import pytest

from src.evidence.confidence import ConfidenceCalculator, DEFAULT_SOURCE_WEIGHTS
from src.evidence.verifier import EvidenceVerifier
from src.models.factory import CandidateFactory


@pytest.fixture
def base_candidate_dict() -> dict:
    return {
        "candidate_id": "CAND_EVID_03",
        "profile": {
            "anonymized_name": "Confidence Tester",
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

def _make_skill(name: str) -> dict:
    return {"name": name, "proficiency": "intermediate", "endorsements": 1, "duration_months": 12}


def _make_role(title: str, description: str = "") -> dict:
    return {
        "company": "Acme",
        "title": title,
        "start_date": "2020-01-01",
        "end_date": "2021-01-01",
        "duration_months": 12,
        "is_current": False,
        "industry": "Tech",
        "company_size": "100-500",
        "description": description,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfidenceShape:
    def test_confidence_in_unit_interval(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConfidenceCalculator().calculate(candidate)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_result_has_required_keys(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConfidenceCalculator().calculate(candidate)
        for key in ("confidence", "feature_count", "corroborated_feature_count",
                    "avg_support_per_feature", "per_feature_confidence"):
            assert key in result

    def test_empty_candidate_zero_confidence(self, base_candidate_dict) -> None:
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConfidenceCalculator().calculate(candidate)
        assert result["confidence"] == 0.0
        assert result["feature_count"] == 0
        assert result["corroborated_feature_count"] == 0


class TestConfidenceMonotonicity:
    """More independent corroborating sources should not decrease confidence."""

    def test_more_sources_increase_confidence(self, base_candidate_dict) -> None:
        # Baseline: skill only in skills list.
        base_candidate_dict["skills"] = [_make_skill("Python")]
        c1 = CandidateFactory.create(base_candidate_dict)
        conf1 = ConfidenceCalculator().calculate(c1)["confidence"]

        # Add summary support.
        d2 = __import__("copy").deepcopy(base_candidate_dict)
        d2["profile"]["summary"] = "Python developer"
        c2 = CandidateFactory.create(d2)
        conf2 = ConfidenceCalculator().calculate(c2)["confidence"]

        # Add experience support.
        d3 = __import__("copy").deepcopy(base_candidate_dict)
        d3["profile"]["summary"] = "Python developer"
        d3["career_history"] = [_make_role("Python Engineer", "Built Python services.")]
        c3 = CandidateFactory.create(d3)
        conf3 = ConfidenceCalculator().calculate(c3)["confidence"]

        assert conf1 > 0.0
        assert conf2 > conf1
        assert conf3 > conf2

    def test_single_source_confidence_is_its_weight_share(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConfidenceCalculator().calculate(candidate)

        total = sum(DEFAULT_SOURCE_WEIGHTS.values())
        expected = DEFAULT_SOURCE_WEIGHTS["skills"] / total
        assert abs(result["per_feature_confidence"]["python"] - round(expected, 4)) < 1e-6


class TestConfidenceUsesProvidedEvidence:
    """Passing pre-computed evidence should be respected and not recomputed."""

    def test_accepts_precomputed_evidence(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)

        result = ConfidenceCalculator().calculate(candidate, evidence=evidence)
        assert result["feature_count"] == 1
        assert result["confidence"] > 0.0

    def test_consistency_modulates_confidence(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        candidate = CandidateFactory.create(base_candidate_dict)
        evidence = EvidenceVerifier().verify(candidate)

        high = ConfidenceCalculator().calculate(
            candidate, evidence=evidence, consistency={"consistency_score": 1.0}
        )
        low = ConfidenceCalculator().calculate(
            candidate, evidence=evidence, consistency={"consistency_score": 0.0}
        )
        # Higher consistency should yield confidence at least as high.
        assert high["confidence"] >= low["confidence"]


class TestConfidenceConfig:
    def test_default_weights_when_no_config(self) -> None:
        calc = ConfidenceCalculator(config_path="config/__does_not_exist__.yaml")
        assert calc.source_weights == DEFAULT_SOURCE_WEIGHTS

    def test_per_feature_confidence_with_all_sources(self, base_candidate_dict) -> None:
        """A feature supported by every source should approach 1.0."""
        base_candidate_dict["skills"] = [_make_skill("Python")]
        base_candidate_dict["profile"]["summary"] = "Python developer"
        base_candidate_dict["profile"]["headline"] = "Python"
        base_candidate_dict["career_history"] = [
            _make_role("Python Engineer", "Python services."),
        ]
        base_candidate_dict["projects"] = ["Python tool"]
        candidate = CandidateFactory.create(base_candidate_dict)
        result = ConfidenceCalculator().calculate(candidate)
        assert result["per_feature_confidence"]["python"] == 1.0


class TestConfidenceDeterminism:
    def test_same_input_same_output(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python")]
        base_candidate_dict["profile"]["summary"] = "Python developer"
        candidate = CandidateFactory.create(base_candidate_dict)
        calc = ConfidenceCalculator()
        assert calc.calculate(candidate) == calc.calculate(candidate)

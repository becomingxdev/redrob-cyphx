"""Tests for the multi-source SkillsExtractor.

Original test classes are preserved **unchanged** so that backward compatibility
is continuously verified. New test classes cover the multi-source pipeline.
"""

import pytest
from src.models.factory import CandidateFactory
from src.features.skills import SkillsExtractor


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(
    name: str,
    proficiency: str,
    endorsements: int,
    duration_months: int | None,
) -> dict:
    """Create a minimal skill entry dict."""
    return {
        "name": name,
        "proficiency": proficiency,
        "endorsements": endorsements,
        "duration_months": duration_months,
    }


def _make_role(
    company: str = "Acme",
    title: str = "Engineer",
    description: str = "",
    industry: str = "Tech",
) -> dict:
    return {
        "company": company,
        "title": title,
        "start_date": "2021-01-01",
        "end_date": "2023-01-01",
        "duration_months": 24,
        "is_current": False,
        "industry": industry,
        "company_size": "100-500",
        "description": description,
    }


def _make_cert(name: str, issuer: str = "Coursera", year: int = 2023) -> dict:
    return {"name": name, "issuer": issuer, "year": year}


def _by_name(multi: list[dict], name: str) -> dict | None:
    """Return the first record whose ``name`` equals ``name``."""
    return next((r for r in multi if r["name"] == name), None)


# ===========================================================================
# ORIGINAL TESTS — preserved unchanged for backward compatibility
# ===========================================================================

class TestSkillsExtraction:
    """Original tests — basic extraction and empty handling."""

    def test_skills_extraction(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("SQL", "expert", 25, 72),
            _make_skill("Git", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = SkillsExtractor()
        features = extractor.extract(candidate)

        assert features["unique_skills"] == 3
        assert features["skills"] == ["git", "python", "sql"]
        assert features["endorsement_total"] == 40
        assert features["expert_count"] == 1
        assert features["advanced_count"] == 1
        assert features["intermediate_count"] == 1
        assert features["duration_statistics"]["min"] == 36
        assert features["duration_statistics"]["max"] == 72
        assert features["duration_statistics"]["avg"] == 56.0

    def test_skills_extraction_empty(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = []
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = SkillsExtractor()
        features = extractor.extract(candidate)

        assert features["unique_skills"] == 0
        assert features["skills"] == []
        assert features["endorsement_total"] == 0
        assert features["expert_count"] == 0
        assert features["duration_statistics"]["min"] == 0
        assert features["duration_statistics"]["avg"] == 0.0


class TestCaseInsensitiveDedup:
    """Verify that duplicate detection is case-insensitive."""

    def test_identical_case_duplicates(self, base_candidate_dict) -> None:
        """Two entries with the exact same name count as one unique skill."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("Python", "expert", 20, 72),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["unique_skills"] == 1
        assert features["skills"] == ["python"]
        # Endorsements should still sum both entries (before dedup)
        assert features["endorsement_total"] == 30

    def test_mixed_case_python_python_upper(self, base_candidate_dict) -> None:
        """Python, python, PYTHON — all three should deduplicate to one."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("python", "expert", 20, 48),
            _make_skill("PYTHON", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["unique_skills"] == 1
        assert features["skills"] == ["python"]

    def test_mixed_case_various_skills(self, base_candidate_dict) -> None:
        """Mixed-case variants of different skills should dedup each group."""
        base_candidate_dict["skills"] = [
            _make_skill("Java", "advanced", 15, 48),
            _make_skill("java", "beginner", 3, 12),
            _make_skill("JAVA", "expert", 30, 72),
            _make_skill("Docker", "intermediate", 8, 24),
            _make_skill("docker", "advanced", 12, 36),
            _make_skill("Kubernetes", "beginner", 2, 6),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["unique_skills"] == 3
        assert sorted(features["skills"]) == ["docker", "java", "kubernetes"]

    def test_whitespace_trimming_in_dedup(self, base_candidate_dict) -> None:
        """Whitespace around skill names should not prevent dedup."""
        base_candidate_dict["skills"] = [
            _make_skill("  Python  ", "advanced", 10, 60),
            _make_skill("python", "expert", 20, 48),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["unique_skills"] == 1
        assert features["skills"] == ["python"]


class TestSkillsEndorsements:
    """Verify endorsement totaling and zero handling."""

    def test_endorsements_sum_all_entries(self, base_candidate_dict) -> None:
        """endorsement_total must sum across all skill entries including duplicates."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("SQL", "expert", 25, 72),
            _make_skill("Git", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["endorsement_total"] == 10 + 25 + 5

    def test_zero_endorsements(self, base_candidate_dict) -> None:
        """Skills with 0 endorsements should not break the total."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 0, 60),
            _make_skill("SQL", "expert", 0, 72),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["endorsement_total"] == 0
        assert features["unique_skills"] == 2


class TestSkillsDurationStatistics:
    """Verify duration min/max/avg/total calculations."""

    def test_duration_with_none_values(self, base_candidate_dict) -> None:
        """Skills with duration_months=None should be excluded from duration stats."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("SQL", "expert", 25, None),
            _make_skill("Git", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        # Only Python (60) and Git (36) have durations
        assert features["duration_statistics"]["min"] == 36
        assert features["duration_statistics"]["max"] == 60
        assert features["duration_statistics"]["avg"] == 48.0  # (60+36)/2
        assert features["duration_statistics"]["total"] == 96
        # SQL should still count as a unique skill
        assert features["unique_skills"] == 3

    def test_duration_all_none(self, base_candidate_dict) -> None:
        """When all skills have None duration, stats should be zeroed."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, None),
            _make_skill("SQL", "expert", 25, None),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["duration_statistics"] == {"min": 0, "max": 0, "avg": 0.0, "total": 0}


class TestSkillsProficiencyBuckets:
    """Verify proficiency bucket mapping and distribution counts."""

    def test_unknown_proficiency_mapped(self, base_candidate_dict) -> None:
        """A proficiency string not in any bucket should be preserved as-is."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "wizard", 10, 60),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        # "wizard" is not in any bucket, so it stays as "wizard"
        assert features["proficiency_distribution"].get("wizard", 0) == 1

    def test_empty_proficiency_mapped_to_unknown(self, base_candidate_dict) -> None:
        """Empty proficiency should map to 'unknown'."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "", 10, 60),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["proficiency_distribution"].get("unknown", 0) == 1

    def test_bucket_count_keys_present(self, base_candidate_dict) -> None:
        """The result must contain explicit {bucket}_count keys for all config buckets."""
        base_candidate_dict["skills"] = [
            _make_skill("Python", "expert", 10, 60),
            _make_skill("SQL", "advanced", 20, 48),
            _make_skill("Git", "beginner", 5, 24),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        assert features["expert_count"] == 1
        assert features["advanced_count"] == 1
        assert features["beginner_count"] == 1
        assert features["intermediate_count"] == 0


class TestSkillsDeterminism:
    """Verify the extractor is deterministic."""

    def test_same_input_same_output(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [
            _make_skill("Python", "advanced", 10, 60),
            _make_skill("python", "expert", 20, 48),
            _make_skill("SQL", "intermediate", 5, 36),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = SkillsExtractor()

        assert extractor.extract(candidate) == extractor.extract(candidate)


# ===========================================================================
# NEW TESTS — multi-source pipeline
# ===========================================================================

class TestMultiSourceKeyPresent:
    """multi_source_skills key must always be present and be a list."""

    def test_key_always_present_empty_candidate(self, base_candidate_dict) -> None:
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        assert "multi_source_skills" in features
        assert isinstance(features["multi_source_skills"], list)

    def test_record_shape(self, base_candidate_dict) -> None:
        """Each record must have the five required keys."""
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        for record in features["multi_source_skills"]:
            for key in ("name", "normalized_name", "sources", "occurrences", "contexts"):
                assert key in record, f"Missing key '{key}' in {record}"


class TestMultiSourceExtraction:
    """Skills should be extracted from headline, summary, career, projects, certs."""

    def test_skill_extracted_from_headline(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["headline"] = "Backend Engineer | Python | Kafka"
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        multi = features["multi_source_skills"]

        # Python from both skills list and headline
        python_rec = _by_name(multi, "python")
        assert python_rec is not None
        assert "headline" in python_rec["sources"]
        assert "skills" in python_rec["sources"]

    def test_skill_extracted_from_summary(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["profile"]["summary"] = "Expert in Python, Airflow, and dbt."
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        assert "python" in names

    def test_skill_extracted_from_career_description(self, base_candidate_dict) -> None:
        """A skill mentioned only in career description must be extracted."""
        base_candidate_dict["career_history"] = [
            _make_role(
                title="Data Engineer",
                description="Implemented Kafka streaming pipelines and built dbt models.",
            )
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        # "kafka streaming" normalizes to "Apache Kafka"
        assert "apache kafka" in names

    def test_skill_extracted_from_career_title(self, base_candidate_dict) -> None:
        """A known skill in the role title must be extracted."""
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["career_history"] = [
            _make_role(title="Python Backend Engineer", description="")
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        python_rec = _by_name(features["multi_source_skills"], "python")
        assert python_rec is not None
        assert "career_history" in python_rec["sources"]

    def test_skill_extracted_from_projects_string_list(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["projects"] = [
            "Built a Python web scraper using Airflow"
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        python_rec = _by_name(features["multi_source_skills"], "python")
        assert python_rec is not None
        assert "projects" in python_rec["sources"]

    def test_skill_extracted_from_projects_dict_list(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Docker", "intermediate", 3, 12)]
        base_candidate_dict["projects"] = [
            {"name": "CI pipeline", "tech": ["Docker", "Kubernetes"]}
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        docker_rec = _by_name(features["multi_source_skills"], "docker")
        assert docker_rec is not None
        assert "projects" in docker_rec["sources"]

    def test_skill_extracted_from_certifications(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["certifications"] = [
            _make_cert("AWS Certified Solutions Architect")
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        # "aws" should appear — from "AWS Certified Solutions Architect" cert name
        names = [r["name"] for r in features["multi_source_skills"]]
        assert "aws" in names

    def test_skill_not_in_explicit_list_but_in_career_appears(self, base_candidate_dict) -> None:
        """Core requirement: career-only skills must appear even if absent from skills list."""
        base_candidate_dict["skills"] = []
        base_candidate_dict["career_history"] = [
            _make_role(
                title="Data Engineer",
                description=(
                    "Used PySpark for batch processing. "
                    "Built Airflow DAGs. "
                    "Wrote dbt models. "
                    "Stored data in Snowflake."
                ),
            )
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        # pyspark → "apache spark"
        assert "apache spark" in names
        assert "airflow" in names
        assert "dbt" in names
        assert "snowflake" in names


class TestNormalization:
    """Alias normalization must produce the canonical display name."""

    def test_pyspark_normalizes_to_apache_spark(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("PySpark", "advanced", 5, 24)]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        rec = _by_name(features["multi_source_skills"], "apache spark")
        assert rec is not None
        assert rec["normalized_name"] == "Apache Spark"

    def test_amazon_web_services_normalizes_to_aws(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["headline"] = "Amazon Web Services | Python"
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        assert "aws" in names

    def test_kafka_streaming_normalizes_to_apache_kafka(self, base_candidate_dict) -> None:
        base_candidate_dict["career_history"] = [
            _make_role(description="Built Kafka streaming pipelines.")
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        assert "apache kafka" in names

    def test_google_cloud_platform_normalizes_to_gcp(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["headline"] = "Google Cloud Platform | DevOps"
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        assert "gcp" in names

    def test_normalized_name_preserved_in_record(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("PySpark", "advanced", 5, 24)]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        rec = _by_name(features["multi_source_skills"], "apache spark")
        assert rec is not None
        assert rec["normalized_name"] == "Apache Spark"

    def test_skills_list_uses_normalized_names_as_keys(self, base_candidate_dict) -> None:
        """The flat 'skills' list must use the lower-cased normalized name, not the raw alias."""
        base_candidate_dict["skills"] = [_make_skill("PySpark", "advanced", 5, 24)]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        # "pyspark" alias → "Apache Spark" → key is "apache spark"
        assert "apache spark" in features["skills"]
        assert "pyspark" not in features["skills"]


class TestSoftSkillFiltering:
    """Soft skills from the blocklist must not appear in extracted results."""

    def test_communication_not_extracted_from_headline(self, base_candidate_dict) -> None:
        base_candidate_dict["profile"]["headline"] = "Communication | Python | Leadership"
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        assert "communication" not in names
        assert "leadership" not in names

    def test_soft_skills_not_extracted_from_career(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["career_history"] = [
            _make_role(
                description=(
                    "Strong communication skills. Used Python for data analysis. "
                    "Leadership and teamwork across departments."
                )
            )
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        assert "communication" not in names
        assert "teamwork" not in names
        assert "python" in names


class TestDeduplication:
    """Same skill from multiple sources must produce one record with all sources listed."""

    def test_skill_from_two_sources_deduped(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["profile"]["summary"] = "Python and SQL expert."
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)

        python_records = [r for r in features["multi_source_skills"] if r["name"] == "python"]
        assert len(python_records) == 1, "Python must appear exactly once even across sources"
        assert "skills" in python_records[0]["sources"]
        assert "summary" in python_records[0]["sources"]

    def test_occurrences_accumulate_across_sources(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["profile"]["headline"] = "Python developer"
        base_candidate_dict["profile"]["summary"] = "Python, SQL"
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        python_rec = _by_name(features["multi_source_skills"], "python")
        assert python_rec is not None
        assert python_rec["occurrences"] >= 3  # skills + headline + summary

    def test_unique_skills_count_reflects_union(self, base_candidate_dict) -> None:
        """unique_skills should equal the number of distinct skills across ALL sources."""
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["career_history"] = [
            _make_role(description="Used Airflow for orchestration. Built dbt models.")
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        # python + airflow + dbt = at least 3
        assert features["unique_skills"] >= 3


class TestContextCapture:
    """Context snippets must be populated and bounded."""

    def test_context_populated_for_career_skill(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Airflow", "intermediate", 2, 12)]
        base_candidate_dict["career_history"] = [
            _make_role(description="Developed Airflow DAGs for data orchestration.")
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        airflow_rec = _by_name(features["multi_source_skills"], "airflow")
        assert airflow_rec is not None
        # At least one context snippet must be non-empty
        assert any(len(c) > 0 for c in airflow_rec["contexts"])

    def test_context_length_bounded(self, base_candidate_dict) -> None:
        long_desc = "A " * 200 + "Python " + "B " * 200
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["career_history"] = [_make_role(description=long_desc)]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        python_rec = _by_name(features["multi_source_skills"], "python")
        assert python_rec is not None
        for ctx in python_rec["contexts"]:
            assert len(ctx) <= 120

    def test_max_contexts_respected(self, base_candidate_dict) -> None:
        """No skill should accumulate more than max_contexts snippets."""
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["career_history"] = [
            _make_role(description=f"Role {i}: Used Python for task {i}.")
            for i in range(10)
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        python_rec = _by_name(features["multi_source_skills"], "python")
        assert python_rec is not None
        # default max_contexts = 3
        assert len(python_rec["contexts"]) <= 3


class TestMultiSourceDeterminism:
    """Repeated calls on the same candidate must return identical results."""

    def test_deterministic_with_all_sources(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [_make_skill("Python", "advanced", 5, 24)]
        base_candidate_dict["profile"]["headline"] = "Python | Spark | Kafka"
        base_candidate_dict["profile"]["summary"] = "Python, dbt, Airflow"
        base_candidate_dict["career_history"] = [
            _make_role(description="PySpark jobs and Kafka streaming pipelines.")
        ]
        base_candidate_dict["projects"] = [{"name": "ETL", "tech": ["Airflow"]}]
        base_candidate_dict["certifications"] = [_make_cert("AWS Certified Developer")]
        candidate = CandidateFactory.create(base_candidate_dict)
        extractor = SkillsExtractor()
        result1 = extractor.extract(candidate)
        result2 = extractor.extract(candidate)
        assert result1 == result2

    def test_multi_source_skills_sorted_by_name(self, base_candidate_dict) -> None:
        base_candidate_dict["skills"] = [
            _make_skill("Spark", "advanced", 5, 24),
            _make_skill("Airflow", "intermediate", 3, 12),
        ]
        candidate = CandidateFactory.create(base_candidate_dict)
        features = SkillsExtractor().extract(candidate)
        names = [r["name"] for r in features["multi_source_skills"]]
        assert names == sorted(names)

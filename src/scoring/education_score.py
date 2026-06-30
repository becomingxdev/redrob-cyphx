"""Education scoring engine for the REDROB candidate ranking system.

Evaluates degree level, field relevance, institution quality (rule-based),
and education count. Education should never dominate ranking.

Fallback #10 fix:
  - Hard cap at 60 (was 81) — education is a tiebreaker, not the main signal.
  - Non-CS/AI/Engineering PhD gets only 50% of degree score.
  - Wildly irrelevant fields (e.g., Chemical Engineering, Biology for an AI role)
    are penalised rather than ignored.

This module is purely responsible for education scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

import re

from src.models.candidate import Candidate
from src.scoring import ScoreResult


# Degree hierarchy scores. These are used only to assign relative points,
# not to determine the "highest" degree (that is the extractor's job).
_RELEVANCE_SCORES: dict[str, float] = {
    # FIX 6b: raised to give field relevance a more meaningful contribution.
    # Previous values: direct=10, partial=7, low=3
    "direct": 15.0,
    "partial": 10.0,
    "low": 5.0,
}

_DEGREE_SCORES: dict[str, float] = {
    # Fallback #10: reduced PhD from 40→30 to limit education dominance.
    # Non-CS/AI PhD will get 50% of this (see _apply_field_relevance_factor).
    "phd": 30.0,
    "masters": 22.0,
    "bachelors": 16.0,
    "diploma": 8.0,
    "school": 4.0,
}

# Institution tier scores (rule-based).
_TIER_SCORES: dict[str, float] = {
    # FIX 6b: raised to give institution tier a more meaningful contribution.
    # Previous values: tier_1=15, tier_2=10, tier_3=5, tier_4=2
    "tier_1": 20.0,
    "tier_2": 14.0,
    "tier_3": 8.0,
    "tier_4": 3.0,
}

# Field relevance keywords for common tech/data roles.
_RELEVANT_FIELDS: list[str] = [
    "computer science",
    "computer engineering",
    "information technology",
    "software engineering",
    "data science",
    "mathematics",
    "statistics",
    "artificial intelligence",
    "machine learning",
    "electronics",
    "electrical engineering",
    "information systems",
]

# Fields that are NOT relevant to an AI engineering role — degree score halved.
_IRRELEVANT_FIELDS: list[str] = [
    "chemical engineering",
    "civil engineering",
    "mechanical engineering",
    "biology",
    "chemistry",
    "physics",  # borderline but acceptable for ML
    "medical",
    "finance",
    "accounting",
    "law",
    "humanities",
    "arts",
    "sociology",
    "psychology",
    "management",
    "marketing",
]


def _classify_degree(degree: str) -> str:
    """Classify a degree string into a standard bucket."""
    if not degree:
        return ""
    d = degree.lower().strip()
    if any(p in d for p in ("phd", "ph.d", "doctor", "doctorate")):
        return "phd"
    if any(p in d for p in ("master", "m.s", "msc", "m.sc", "mba", "m.tech", "mca", "pgdm")):
        return "masters"
    if any(p in d for p in ("bachelor", "b.s", "bsc", "b.sc", "b.tech", "b.e", "bba", "b.com")):
        return "bachelors"
    if any(p in d for p in ("diploma", "associate", "foundation")):
        return "diploma"
    if any(p in d for p in ("school", "high school", "ssc", "hsc", "12th", "10th")):
        return "school"
    return ""


def _field_relevance(field_of_study: str, candidate_skills: set[str]) -> tuple[float, str]:
    """Assess relevance of the field of study.

    Fallback #10: Irrelevant fields now return a small score (not the same as
    relevant fields). The degree score itself is halved separately for
    truly irrelevant PhDs.

    Returns:
        Tuple of (relevance score, reason string).
    """
    if not field_of_study:
        return 0.0, "No field of study recorded"

    field_lower = field_of_study.lower()

    # Check for explicitly irrelevant fields first.
    is_irrelevant = any(irr in field_lower for irr in _IRRELEVANT_FIELDS)
    if is_irrelevant:
        return 1.0, f"Field '{field_of_study}' not relevant to AI engineering role"

    # Direct keyword match against relevant fields.
    is_relevant = any(rf in field_lower for rf in _RELEVANT_FIELDS)
    if is_relevant:
        return 10.0, f"Field '{field_of_study}' is relevant to tech/data roles"

    # Check if field keywords appear in skills.
    field_words = set(re.findall(r"\b\w+\b", field_lower))
    skill_words = set()
    for skill in candidate_skills:
        skill_words.update(re.findall(r"\b\w+\b", skill.lower()))
    overlap = field_words & skill_words
    if len(overlap) >= 2:
        return 6.0, f"Field '{field_of_study}' partially relevant (shares {len(overlap)} terms with skills)"

    return 2.0, f"Field '{field_of_study}' has low direct relevance"


def _apply_field_relevance_factor(
    degree_score: float,
    field_of_study: str,
    degree_bucket: str,
) -> tuple[float, str | None]:
    """Halve the degree score for PhDs/Masters in non-CS/AI fields.

    Fallback #10: A PhD in Chemical Engineering should not outscore
    a B.Tech in CS for an AI Engineering role.

    Returns:
        Tuple of (adjusted degree score, optional reason string).
    """
    if not field_of_study:
        return degree_score, None

    field_lower = field_of_study.lower()
    is_irrelevant = any(irr in field_lower for irr in _IRRELEVANT_FIELDS)
    if not is_irrelevant:
        return degree_score, None

    if degree_bucket in ("phd", "masters"):
        adjusted = degree_score * 0.5
        return adjusted, (
            f"Degree score halved: {degree_bucket.upper()} in '{field_of_study}' "
            f"not relevant to AI engineering (was {degree_score:.1f} → {adjusted:.1f})"
        )
    return degree_score, None


def score_education(
    candidate: Candidate,
    education_features: dict,
    candidate_skills: set[str] | None = None,
) -> ScoreResult:
    """Score a candidate's education profile.

    Args:
        candidate: The candidate to score.
        education_features: Output from ``EducationExtractor.extract()``.
        candidate_skills: Optional set of candidate skill names for relevance
            checking. If ``None``, relevance is scored without skill context.

    Returns:
        A ``ScoreResult`` with score in [0, 100]. Education is naturally
        capped to prevent it from dominating.
    """
    reasons: list[str] = []
    score = 0.0
    skills = candidate_skills or set()

    highest_degree = education_features.get("highest_degree", "")
    field_of_study = education_features.get("field_of_study", "")
    education_tier = education_features.get("education_tier", "")
    education_count = education_features.get("education_count", 0)

    # --- No education at all ---
    if not highest_degree and education_count == 0:
        reasons.append("No education records found")
        return ScoreResult(
            score=0.0,
            reasons=reasons,
            metadata={"degree": "", "tier": "", "relevance": 0.0},
        )

    # --- Degree level scoring ---
    degree_bucket = _classify_degree(highest_degree)
    degree_score = _DEGREE_SCORES.get(degree_bucket, 5.0)

    # Fallback #10: halve degree score for irrelevant fields (PhD in ChemEng etc.)
    adjusted_degree_score, field_adj_reason = _apply_field_relevance_factor(
        degree_score, field_of_study, degree_bucket
    )
    degree_score = adjusted_degree_score
    score += degree_score
    reasons.append(f"Highest degree '{highest_degree}' ({degree_bucket or 'unknown'}): +{degree_score:.1f}")
    if field_adj_reason:
        reasons.append(field_adj_reason)

    # --- Institution tier scoring ---
    tier_lower = (education_tier or "").lower().strip()
    tier_score = _TIER_SCORES.get(tier_lower, 3.0)
    score += tier_score
    if tier_lower in _TIER_SCORES:
        reasons.append(f"Institution tier '{education_tier}': +{tier_score:.1f}")
    else:
        reasons.append(f"Unknown institution tier: +{tier_score:.1f} (default)")

    # --- Field relevance ---
    relevance_score, relevance_reason = _field_relevance(field_of_study, skills)
    score += relevance_score
    reasons.append(relevance_reason)

    # --- Multiple degrees bonus (small) ---
    if education_count > 1:
        multi_bonus = min((education_count - 1) * 3.0, 6.0)
        score += multi_bonus
        reasons.append(f"Multiple degrees ({education_count}): +{multi_bonus:.1f}")

    # Fallback #10: Hard cap at 60. Education is a soft tiebreaker.
    # Natural max for CS PhD from tier_1 with relevant field:
    #   degree(30) + tier(20) + relevance(10) + multi-bonus(6) = 66 → capped at 60.
    # PhD in ChemEng from tier_1: (30*0.5=15) + 20 + 1 + 3 = 39 (well under cap).
    score = max(0.0, min(60.0, round(score, 2)))

    metadata = {
        "degree": highest_degree,
        "degree_bucket": degree_bucket,
        "tier": education_tier,
        "relevance": round(relevance_score, 2),
    }

    return ScoreResult(score=score, reasons=reasons, metadata=metadata)

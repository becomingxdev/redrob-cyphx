"""Skill scoring engine for the REDROB candidate ranking system.

Evaluates a candidate's skill set against required and preferred skills,
with bonuses for complementary skill combinations (synergy).

This module is purely responsible for skill scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import ScoreResult


# Predefined complementary skill groups.
# If a candidate has multiple skills from the same group, they earn a synergy bonus.
_SYNERGY_GROUPS: list[list[str]] = [
    # Backend stack
    ["python", "fastapi", "docker", "flask", "django", "golang", "java", "node"],
    # Data/ML stack
    ["python", "sql", "spark", "airflow", "ml", "machine learning", "deep learning", "pytorch", "tensorflow"],
    # Cloud/DevOps
    ["aws", "gcp", "azure", "kubernetes", "docker", "terraform", "ci/cd", "jenkins"],
    # Frontend stack
    ["react", "vue", "angular", "javascript", "typescript", "html", "css", "next.js"],
    # Data warehousing
    ["sql", "snowflake", "bigquery", "dbt", "spark", "airflow"],
]


def _count_synergy(candidate_skills: set[str]) -> tuple[float, list[str]]:
    """Count synergy bonuses from complementary skill groups.

    For each group, if a candidate has 2+ skills from that group, award a bonus.
    Each group is counted at most once to avoid duplicate rewards.

    Returns:
        Tuple of (total bonus points, list of reason strings).
    """
    total_bonus = 0.0
    synergy_reasons: list[str] = []
    seen_skills: set[str] = set()  # Track skills already rewarded to prevent double-counting.

    for group in _SYNERGY_GROUPS:
        matched = set()
        for skill in candidate_skills:
            if skill in group and skill not in seen_skills:
                matched.add(skill)

        if len(matched) >= 2:
            # Bonus scales with group coverage but is capped.
            bonus = min(len(matched) * 3.0, 12.0)
            total_bonus += bonus
            synergy_reasons.append(
                f"Synergy bonus for {sorted(matched)}: +{bonus:.1f}"
            )
            seen_skills.update(matched)

    return total_bonus, synergy_reasons


def score_skills(
    candidate: Candidate,
    skill_features: dict,
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
) -> ScoreResult:
    """Score a candidate's skills.

    Args:
        candidate: The candidate to score.
        skill_features: Output from ``SkillsExtractor.extract()``.
        required_skills: Skills that are mandatory for the role.
        preferred_skills: Skills that are nice-to-have for the role.

    Returns:
        A ``ScoreResult`` with score in [0, 100].
    """
    reasons: list[str] = []
    score = 0.0
    candidate_skills = set(skill_features.get("skills", []))
    unique_count = skill_features.get("unique_skills", 0)
    endorsement_total = skill_features.get("endorsement_total", 0)

    # --- Base score for having skills ---
    if unique_count == 0:
        reasons.append("No skills found")
        return ScoreResult(
            score=0.0,
            reasons=reasons,
            metadata={"match_count": 0, "required_count": 0, "preferred_count": 0, "synergy": 0.0},
        )

    # Volume score: more skills = better, but with diminishing returns.
    volume_score = min(unique_count * 2.0, 20.0)
    score += volume_score
    reasons.append(f"Skill volume ({unique_count} unique): +{volume_score:.1f}")

    # --- Required skill matching ---
    if required_skills:
        required_lower = {s.lower() for s in required_skills}
        matched_required = candidate_skills & required_lower
        required_ratio = len(matched_required) / len(required_lower) if required_lower else 0.0
        required_score = required_ratio * 40.0  # Up to 40 points for required skills.
        score += required_score
        reasons.append(
            f"Required skills: {len(matched_required)}/{len(required_lower)} matched: +{required_score:.1f}"
        )
    else:
        matched_required = set()
        required_ratio = 1.0  # No requirements = no penalty.

    # --- Preferred skill matching ---
    if preferred_skills:
        preferred_lower = {s.lower() for s in preferred_skills}
        matched_preferred = candidate_skills & preferred_lower
        preferred_ratio = len(matched_preferred) / len(preferred_lower) if preferred_lower else 0.0
        preferred_score = preferred_ratio * 15.0  # Up to 15 points for preferred skills.
        score += preferred_score
        reasons.append(
            f"Preferred skills: {len(matched_preferred)}/{len(preferred_lower)} matched: +{preferred_score:.1f}"
        )
    else:
        matched_preferred = set()

    # --- Endorsement bonus ---
    if endorsement_total > 0:
        endorsement_bonus = min(endorsement_total * 0.15, 10.0)
        score += endorsement_bonus
        reasons.append(f"Endorsements ({endorsement_total} total): +{endorsement_bonus:.1f}")

    # --- Synergy bonuses ---
    synergy_bonus, synergy_reasons = _count_synergy(candidate_skills)
    if synergy_bonus > 0:
        score += synergy_bonus
        reasons.extend(synergy_reasons)

    # Clamp to [0, 100].
    score = max(0.0, min(100.0, round(score, 2)))

    metadata = {
        "match_count": len(matched_required),
        "required_count": len(required_skills or []),
        "preferred_count": len(matched_preferred),
        "synergy": round(synergy_bonus, 2),
        "unique_skills": unique_count,
    }

    return ScoreResult(score=score, reasons=reasons, metadata=metadata)

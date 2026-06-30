"""Penalty engine for the REDROB candidate ranking system.

Applies soft, non-rejecting penalties for profile quality issues.
Penalties reduce the final score but never eliminate a candidate.

This module is purely responsible for penalty assessment. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import PenaltyResult


def _check_job_hopping(career_features: dict) -> tuple[float, str]:
    """Detect excessive job hopping.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    stability = career_features.get("career_stability_metrics", {})
    hopping_index = stability.get("job_hopping_index", 0.0)
    avg_tenure = stability.get("average_tenure_months", 0.0)

    # Penalize if hopping index is high AND average tenure is low.
    if hopping_index > 1.0 and avg_tenure < 12:
        penalty = min((hopping_index - 1.0) * 5.0 + (12 - avg_tenure) * 0.3, 15.0)
        return penalty, (
            f"Job hopping detected (index={hopping_index:.2f}, "
            f"avg tenure={avg_tenure:.1f} months)"
        )

    return 0.0, ""


def _check_inflated_title(
    title_features: dict,
    career_features: dict,
    experience_features: dict,
) -> tuple[float, str]:
    """Detect potentially inflated titles relative to career history.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    seniority = title_features.get("seniority", "mid")
    yoe = experience_features.get("years_of_experience", 0.0)
    total_positions = experience_features.get("total_positions", 0)
    has_promotion = career_features.get("promotion_indicators", False)

    # A "senior" or "lead" title with very little experience is suspicious.
    if seniority in ("senior", "lead", "principal", "staff") and yoe < 2.0:
        penalty = min((2.0 - yoe) * 5.0, 10.0)
        return penalty, (
            f"Title '{seniority}' with only {yoe:.1f} years of experience"
        )

    # "Principal" or "staff" without promotion history and few positions.
    if seniority in ("principal", "staff") and not has_promotion and total_positions <= 2:
        penalty = 8.0
        return penalty, (
            f"Title '{seniority}' with no promotion history and only "
            f"{total_positions} positions"
        )

    return 0.0, ""


def _check_weak_evidence(evidence_result: dict) -> tuple[float, str]:
    """Detect candidates with weak overall evidence support.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    if not evidence_result:
        return 5.0, "No evidence available"

    # Count skills with only one source or zero corroborating sources.
    weak_skills = 0
    total_skills = len(evidence_result)
    for skill_evidence in evidence_result.values():
        support = skill_evidence.get("source_support_count", 0)
        if support <= 1:
            weak_skills += 1

    if total_skills > 0:
        weak_ratio = weak_skills / total_skills
        if weak_ratio > 0.8:
            penalty = min(weak_ratio * 8.0, 10.0)
            return penalty, (
                f"Weak evidence for {weak_skills}/{total_skills} skills "
                f"({weak_ratio:.0%} with single-source support)"
            )

    return 0.0, ""


def _check_inconsistent_profile(consistency_result: dict) -> tuple[float, str]:
    """Detect profile inconsistency based on consistency analysis.

    FIX 5: Max penalty reduced from 10 → 5 points. Consistency is now the
    primary signal via the composite engine's _apply_consistency_bonus
    multiplier. This function handles only severely inconsistent profiles
    (< 0.5 score) to avoid triple-counting.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    consistency_score = consistency_result.get("consistency_score", 1.0)
    conflicts = consistency_result.get("conflicts", [])

    if consistency_score < 0.5:
        # FIX 5: cap reduced from 10 to 5
        penalty = min((0.5 - consistency_score) * 10.0, 5.0)
        return penalty, (
            f"Low consistency score ({consistency_score:.2f}): "
            f"{len(conflicts)} conflict(s)"
        )

    return 0.0, ""


def _check_missing_information(
    candidate: Candidate,
    experience_features: dict,
    education_features: dict,
) -> tuple[float, str]:
    """Detect missing or incomplete profile information.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    missing: list[str] = []
    penalty = 0.0

    if not candidate.profile or not candidate.profile.summary:
        missing.append("summary")
        penalty += 2.0

    if not candidate.career_history or len(candidate.career_history) == 0:
        missing.append("career_history")
        penalty += 3.0

    if not candidate.skills or len(candidate.skills) == 0:
        missing.append("skills")
        penalty += 3.0

    if education_features.get("highest_degree", "") == "" and education_features.get("education_count", 0) == 0:
        missing.append("education")
        penalty += 1.0

    if not candidate.redrob_signals or not getattr(candidate.redrob_signals, "verified_email", False):
        missing.append("unverified_email")
        penalty += 1.0

    if missing:
        penalty = min(penalty, 10.0)
        return penalty, f"Missing profile sections: {', '.join(missing)}"

    return 0.0, ""


def apply_penalties(
    candidate: Candidate,
    title_features: dict,
    experience_features: dict,
    education_features: dict,
    career_features: dict,
    evidence_result: dict,
    consistency_result: dict,
) -> PenaltyResult:
    """Apply soft penalties to a candidate's profile.

    Penalties are additive and non-rejecting. They reduce the final score
    via a multiplier in the composite engine, but never eliminate candidates.

    Args:
        candidate: The candidate to assess.
        title_features: Output from ``TitleExtractor.extract()``.
        experience_features: Output from ``ExperienceExtractor.extract()``.
        education_features: Output from ``EducationExtractor.extract()``.
        career_features: Output from ``CareerExtractor.extract()``.
        evidence_result: Output from ``EvidenceVerifier.verify()``.
        consistency_result: Output from ``ConsistencyAnalyzer.analyze()``.

    Returns:
        A ``PenaltyResult`` with the total penalty score and breakdown.
    """
    total_penalty = 0.0
    reasons: list[str] = []
    breakdown: dict[str, float] = {}

    # Run each penalty check.
    # FIX 4: job_hopping removed — it is already penalized inside
    # score_experience (up to -10 pts via hopping_index deduction).
    # FIX 5: inconsistent_profile penalty reduced; see _check_inconsistent_profile.
    checks = [
        ("inflated_title", _check_inflated_title(title_features, career_features, experience_features)),
        ("weak_evidence", _check_weak_evidence(evidence_result)),
        ("inconsistent_profile", _check_inconsistent_profile(consistency_result)),
        ("missing_information", _check_missing_information(candidate, experience_features, education_features)),
    ]

    for name, (penalty, reason) in checks:
        if penalty > 0 and reason:
            total_penalty += penalty
            reasons.append(f"[{name}] {reason}: -{penalty:.1f}")
            breakdown[name] = penalty
        else:
            breakdown[name] = 0.0

    # Cap total penalty to avoid harsh multipliers.
    total_penalty = min(total_penalty, 30.0)
    total_penalty = round(total_penalty, 2)

    metadata = {"breakdown": breakdown}

    return PenaltyResult(
        penalty_score=total_penalty,
        reasons=reasons,
        metadata=metadata,
    )

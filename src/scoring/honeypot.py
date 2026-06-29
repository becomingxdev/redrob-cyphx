"""Honeypot detection engine for the REDROB candidate ranking system.

Detects suspicious or fabricated profile signals: impossible timelines,
contradictory experience, fake expertise claims, and suspicious patterns.

This module never removes candidates. It produces a suspicion score
that the composite engine uses as a soft down-weighting factor.

This module is purely responsible for honeypot detection. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import HoneypotResult


def _check_impossible_timelines(
    candidate: Candidate,
    experience_features: dict,
    education_features: dict,
) -> tuple[float, str]:
    """Detect temporally impossible claims.

    Checks:
    - Career starting before education ended.
    - More total career months than plausible given YoE.

    Returns:
        Tuple of (suspicion points, reason string).
    """
    career_start_year = None
    education_end_year = education_features.get("graduation_year", 0)

    # Find earliest career start.
    if candidate.career_history:
        for role in candidate.career_history:
            if role.start_date:
                try:
                    year = int(role.start_date[:4])
                    if career_start_year is None or year < career_start_year:
                        career_start_year = year
                except (ValueError, IndexError):
                    pass

    # Career starts 2+ years before education ends.
    if (
        career_start_year is not None
        and education_end_year > 0
        and career_start_year < education_end_year - 2
    ):
        suspicion = 20.0
        return suspicion, (
            f"Career started in {career_start_year} but education ended in "
            f"{education_end_year} (gap of {education_end_year - career_start_year} years)"
        )

    # Total months worked wildly exceeds declared YoE.
    yoe = experience_features.get("years_of_experience", 0.0)
    total_months = experience_features.get("total_months_worked", 0)
    if yoe > 0 and total_months > 0:
        max_plausible_months = (yoe + 2.0) * 12  # Generous 2-year buffer.
        if total_months > max_plausible_months:
            suspicion = 15.0
            return suspicion, (
                f"Total months worked ({total_months}) exceeds plausible "
                f"range for {yoe:.1f} YoE (max ~{max_plausible_months:.0f} months)"
            )

    return 0.0, ""


def _check_contradictory_experience(
    candidate: Candidate,
    experience_features: dict,
    title_features: dict,
) -> tuple[float, str]:
    """Detect contradictions between title, YoE, and career history.

    Returns:
        Tuple of (suspicion points, reason string).
    """
    yoe = experience_features.get("years_of_experience", 0.0)
    total_positions = experience_features.get("total_positions", 0)
    seniority = title_features.get("seniority", "mid")

    # Very high YoE with very few positions.
    if yoe > 15.0 and total_positions <= 1:
        suspicion = 15.0
        return suspicion, (
            f"{yoe:.1f} YoE with only {total_positions} position(s) recorded"
        )

    # Senior/principal title with zero career history.
    if seniority in ("senior", "lead", "principal", "staff") and total_positions == 0:
        suspicion = 20.0
        return suspicion, (
            f"Title seniority '{seniority}' with no career history entries"
        )

    # High YoE but very short total tenure (suggests inflated claim).
    total_months = experience_features.get("total_months_worked", 0)
    if yoe > 5.0 and total_months > 0:
        implied_months = yoe * 12
        if total_months < implied_months * 0.3:
            suspicion = 10.0
            return suspicion, (
                f"Declared {yoe:.1f} YoE but only {total_months} months of "
                f"recorded tenure ({total_months / (implied_months + 0.01) * 100:.0f}% of expected)"
            )

    return 0.0, ""


def _check_fake_expertise(
    candidate: Candidate,
    evidence_result: dict,
) -> tuple[float, str]:
    """Detect potentially fabricated expertise claims.

    Checks:
    - Skills declared but with zero evidence sources.
    - All skills have only the explicit "skills" source (self-reported only).

    Returns:
        Tuple of (suspicion points, reason string).
    """
    if not evidence_result:
        return 0.0, "No evidence available"

    # Count skills that have ONLY the explicit skills source.
    self_reported_only = 0
    total = len(evidence_result)
    for skill, evidence in evidence_result.items():
        sources = evidence.get("sources", [])
        if sources == ["skills"]:
            self_reported_only += 1

    if total >= 3 and self_reported_only == total:
        suspicion = 15.0
        return suspicion, (
            f"All {total} skills are self-reported only (no corroboration)"
        )

    if total >= 5 and self_reported_only / total > 0.8:
        suspicion = 10.0
        return suspicion, (
            f"{self_reported_only}/{total} skills are self-reported only "
            f"({self_reported_only / total:.0%})"
        )

    return 0.0, ""


def _check_suspicious_claims(
    candidate: Candidate,
    consistency_result: dict,
    evidence_result: dict,
) -> tuple[float, str]:
    """Detect suspicious patterns from consistency and evidence data.

    Returns:
        Tuple of (suspicion points, reason string).
    """
    consistency_score = consistency_result.get("consistency_score", 1.0)
    conflicts = consistency_result.get("conflicts", [])

    # Very low consistency is suspicious.
    if consistency_score < 0.3 and len(conflicts) >= 3:
        suspicion = 15.0
        return suspicion, (
            f"Very low consistency ({consistency_score:.2f}) with "
            f"{len(conflicts)} conflicts"
        )

    # Many zero-source skills is suspicious.
    if evidence_result:
        zero_source = sum(
            1 for e in evidence_result.values()
            if e.get("source_support_count", 0) == 0
        )
        if zero_source > 3:
            suspicion = min(zero_source * 3.0, 15.0)
            return suspicion, (
                f"{zero_source} skills with zero evidence sources"
            )

    return 0.0, ""


def detect_honeypot(
    candidate: Candidate,
    experience_features: dict,
    education_features: dict,
    title_features: dict,
    evidence_result: dict,
    consistency_result: dict,
) -> HoneypotResult:
    """Run honeypot detection checks on a candidate.

    Produces a suspicion score that the composite engine uses to soft-
    down-weight the final score. Candidates are never removed.

    Args:
        candidate: The candidate to assess.
        experience_features: Output from ``ExperienceExtractor.extract()``.
        education_features: Output from ``EducationExtractor.extract()``.
        title_features: Output from ``TitleExtractor.extract()``.
        evidence_result: Output from ``EvidenceVerifier.verify()``.
        consistency_result: Output from ``ConsistencyAnalyzer.analyze()``.

    Returns:
        A ``HoneypotResult`` with the total suspicion score and breakdown.
    """
    total_suspicion = 0.0
    reasons: list[str] = []
    breakdown: dict[str, float] = {}

    checks = [
        ("impossible_timelines", _check_impossible_timelines(
            candidate, experience_features, education_features)),
        ("contradictory_experience", _check_contradictory_experience(
            candidate, experience_features, title_features)),
        ("fake_expertise", _check_fake_expertise(candidate, evidence_result)),
        ("suspicious_claims", _check_suspicious_claims(
            candidate, consistency_result, evidence_result)),
    ]

    for name, (suspicion, reason) in checks:
        if suspicion > 0 and reason:
            total_suspicion += suspicion
            reasons.append(f"[{name}] {reason}: +{suspicion:.1f} suspicion")
            breakdown[name] = suspicion
        else:
            breakdown[name] = 0.0

    # Cap suspicion score.
    total_suspicion = min(total_suspicion, 100.0)
    total_suspicion = round(total_suspicion, 2)

    metadata = {"breakdown": breakdown}

    return HoneypotResult(
        suspicion_score=total_suspicion,
        reasons=reasons,
        metadata=metadata,
    )

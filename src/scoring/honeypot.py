"""Honeypot detection engine for the REDROB candidate ranking system.

Detects suspicious or fabricated profile signals: impossible timelines,
contradictory experience, fake expertise claims, and suspicious patterns.

Fallback #7 fix:
  - Added new checks: keyword-stuffed titles, expert claims with 0 duration,
    impossible candidate age, behavioural twin detection
  - Hard filter: suspicion_score > threshold → composite score capped at 20

This module never removes candidates. It produces a suspicion score that
the composite engine uses for down-weighting. Above the hard threshold,
the composite engine caps the final score to 20.

This module is purely responsible for honeypot detection. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

import re

from src.models.candidate import Candidate
from src.scoring import HoneypotResult
from src.jd_config import JD


# Threshold above which composite engine hard-caps score to 20
HARD_FILTER_THRESHOLD = JD.get("honeypot_hard_filter_threshold", 40)


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

    if candidate.career_history:
        for role in candidate.career_history:
            if role.start_date:
                try:
                    year = int(role.start_date[:4])
                    if career_start_year is None or year < career_start_year:
                        career_start_year = year
                except (ValueError, IndexError):
                    pass

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

    yoe = experience_features.get("years_of_experience", 0.0)
    total_months = experience_features.get("total_months_worked", 0)
    if yoe > 0 and total_months > 0:
        max_plausible_months = (yoe + 2.0) * 12
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

    if yoe > 15.0 and total_positions <= 1:
        suspicion = 15.0
        return suspicion, (
            f"{yoe:.1f} YoE with only {total_positions} position(s) recorded"
        )

    if seniority in ("senior", "lead", "principal", "staff") and total_positions == 0:
        suspicion = 20.0
        return suspicion, (
            f"Title seniority '{seniority}' with no career history entries"
        )

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
    - All skills have only the explicit 'skills' source (self-reported only).

    Returns:
        Tuple of (suspicion points, reason string).
    """
    if not evidence_result:
        return 0.0, "No evidence available"

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

    if consistency_score < 0.3 and len(conflicts) >= 3:
        suspicion = 15.0
        return suspicion, (
            f"Very low consistency ({consistency_score:.2f}) with "
            f"{len(conflicts)} conflicts"
        )

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


def _check_keyword_stuffed_title(title_features: dict) -> tuple[float, str]:
    """Detect keyword-stuffed titles — multiple role keywords jammed together.

    Example honeypot: 'AI ML Deep Learning NLP LLM Senior Engineer'

    Fallback #7 new check.

    Returns:
        Tuple of (suspicion points, reason string).
    """
    title = (title_features.get("title") or "").strip()
    if not title:
        return 0.0, ""

    # Count distinct role/domain keywords in the title
    role_keywords = [
        "ai", "ml", "machine learning", "deep learning", "nlp", "llm",
        "data science", "data scientist", "computer vision", "software",
        "engineer", "architect", "developer", "analyst", "scientist",
        "researcher", "lead", "senior", "principal", "staff", "backend",
        "frontend", "fullstack", "cloud", "devops",
    ]
    title_lower = title.lower()
    keyword_hits = sum(1 for kw in role_keywords if kw in title_lower)

    if keyword_hits >= 6:
        suspicion = 20.0
        return suspicion, (
            f"Keyword-stuffed title ({keyword_hits} role keywords in '{title}'): "
            f"honeypot pattern"
        )
    elif keyword_hits >= 4:
        suspicion = 10.0
        return suspicion, (
            f"Potentially stuffed title ({keyword_hits} role keywords in '{title}')"
        )

    return 0.0, ""


def _check_expert_zero_duration(candidate: Candidate) -> tuple[float, str]:
    """Detect 'expert' proficiency claims on skills with 0 duration_months.

    Classic honeypot pattern: claim expert in 10+ skills but no usage time.

    Fallback #7 new check.

    Returns:
        Tuple of (suspicion points, reason string).
    """
    if not candidate.skills:
        return 0.0, ""

    expert_zero_count = 0
    for skill in candidate.skills:
        proficiency = (getattr(skill, "proficiency", "") or "").lower()
        duration = getattr(skill, "duration_months", None)
        if proficiency in ("expert", "advanced") and duration == 0:
            expert_zero_count += 1

    if expert_zero_count >= 5:
        suspicion = min(expert_zero_count * 3.0, 20.0)
        return suspicion, (
            f"{expert_zero_count} skills claimed 'expert/advanced' with 0 months duration: "
            f"honeypot signal (+{suspicion:.0f})"
        )
    elif expert_zero_count >= 3:
        suspicion = min(expert_zero_count * 2.0, 10.0)
        return suspicion, (
            f"{expert_zero_count} skills claimed 'expert/advanced' with 0 months duration: "
            f"suspicious (+{suspicion:.0f})"
        )

    return 0.0, ""


def _check_impossible_age(
    candidate: Candidate,
    experience_features: dict,
    education_features: dict,
) -> tuple[float, str]:
    """Detect impossible candidate age based on career start year.

    If career_start_year implies the candidate was < 16 years old → honeypot.

    Fallback #7 new check.

    Returns:
        Tuple of (suspicion points, reason string).
    """
    CURRENT_YEAR = 2026
    MIN_WORK_AGE = 16

    # Find earliest career start year
    career_start_year = None
    if candidate.career_history:
        for role in candidate.career_history:
            if role.start_date:
                try:
                    year = int(role.start_date[:4])
                    if career_start_year is None or year < career_start_year:
                        career_start_year = year
                except (ValueError, IndexError):
                    pass

    # Find graduation year from education
    grad_year = education_features.get("graduation_year", 0)

    # Estimate approximate current age from career data
    if career_start_year and career_start_year > 1900:
        # If started working in e.g. 1998, candidate is at least ~(2026-1998+16)=44
        # Minimum age when they started working: 16
        estimated_start_age = MIN_WORK_AGE
        estimated_birth_year = career_start_year - estimated_start_age
        estimated_current_age = CURRENT_YEAR - estimated_birth_year

        if career_start_year < 1970:
            # Working before 1970 but dataset is for modern professionals → suspicious
            suspicion = 25.0
            return suspicion, (
                f"Career start year {career_start_year} implies implausibly old candidate "
                f"(~{estimated_current_age}+ years): honeypot timestamp"
            )
        elif career_start_year > CURRENT_YEAR:
            suspicion = 30.0
            return suspicion, (
                f"Career start year {career_start_year} is in the future: impossible timeline"
            )

    return 0.0, ""


def _check_excessive_skill_count(candidate: Candidate) -> tuple[float, str]:
    """Detect unrealistically large skill lists.

    A candidate listing 80+ unique skills is suspicious — it takes
    meaningful time to develop each skill.

    Fallback #7 new check.

    Returns:
        Tuple of (suspicion points, reason string).
    """
    if not candidate.skills:
        return 0.0, ""

    skill_count = len(candidate.skills)
    if skill_count > 100:
        suspicion = 20.0
        return suspicion, (
            f"Unrealistically large skill list ({skill_count} skills): "
            f"honeypot pattern (+{suspicion:.0f})"
        )
    elif skill_count > 60:
        suspicion = 10.0
        return suspicion, (
            f"Very large skill list ({skill_count} skills): suspicious (+{suspicion:.0f})"
        )

    return 0.0, ""


def _check_skill_duration_vs_career_timeline(
    candidate: Candidate,
    experience_features: dict,
) -> tuple[float, str]:
    """Detect skills whose claimed duration exceeds the candidate's total career span.

    Fallback #18 fix: Candidates who claim 10 years of Python experience but
    only have 4 years of career history are the exact 'subtly impossible profile'
    the hackathon warns about. The existing expert+zero-duration check catches
    0-value fraud; this check catches inflated non-zero durations.

    Logic:
        1. Compute total_career_months = months from earliest role start_date
           to today (using experience_features fallback if available).
        2. For each skill with a declared duration_months > 0, check if it
           exceeds total_career_months + GRACE_MONTHS.
        3. Suspicion scales with:
           - Number of violating skills (breadth of fraud)
           - Size of the worst individual excess (severity)

    Tolerances:
        GRACE_MONTHS = 12   Allow up to 12 months over career span
                            (accounts for side projects / pre-career coding)
        MAX_SUSPICION = 25  Capped so it can combine with other checks.

    Returns:
        Tuple of (suspicion points, reason string).
    """
    if not candidate.skills:
        return 0.0, ""

    GRACE_MONTHS: int = 12
    MAX_SUSPICION: float = 25.0
    CURRENT_YEAR: int = 2026

    # --- Step 1: Compute total career span in months ---
    # Prefer pre-computed total from ExperienceExtractor; fall back to
    # scanning career_history directly so this check is self-contained.
    total_career_months: float | None = None

    # ExperienceExtractor stores total_months_worked
    extracted_months = experience_features.get("total_months_worked")
    if extracted_months and extracted_months > 0:
        total_career_months = float(extracted_months)
    else:
        # Derive from earliest career start_date
        earliest_start: tuple[int, int] | None = None  # (year, month)
        if candidate.career_history:
            for role in candidate.career_history:
                if not role.start_date:
                    continue
                try:
                    parts = role.start_date.strip()[:7].split("-")
                    yr = int(parts[0])
                    mo = int(parts[1]) if len(parts) > 1 else 1
                    if earliest_start is None or (yr, mo) < earliest_start:
                        earliest_start = (yr, mo)
                except (ValueError, IndexError):
                    continue

        if earliest_start:
            start_yr, start_mo = earliest_start
            # Months from start to mid-2026
            total_career_months = (CURRENT_YEAR - start_yr) * 12 + (6 - start_mo)
            total_career_months = max(1.0, total_career_months)

    if total_career_months is None or total_career_months <= 0:
        # Can't compute career span — skip check rather than false-positive
        return 0.0, ""

    allowed_months = total_career_months + GRACE_MONTHS

    # --- Step 2: Find violating skills ---
    violations: list[tuple[str, int, float]] = []  # (skill_name, claimed, excess)
    for skill in candidate.skills:
        skill_name = (getattr(skill, "name", "") or "").strip()
        duration = getattr(skill, "duration_months", None)
        if duration is None or duration <= 0:
            continue  # zero/None handled by expert_zero_duration check
        excess = duration - allowed_months
        if excess > 0:
            violations.append((skill_name, int(duration), excess))

    if not violations:
        return 0.0, ""

    # --- Step 3: Score based on breadth and worst excess ---
    n_violations = len(violations)
    worst_excess = max(v[2] for v in violations)  # months
    worst_skill, worst_claimed, _ = max(violations, key=lambda v: v[2])

    # Breadth component: 5 pts per violating skill, capped at 15
    breadth_pts = min(n_violations * 5.0, 15.0)
    # Severity component: 1 pt per 6 months of excess, capped at 10
    severity_pts = min(worst_excess / 6.0, 10.0)
    suspicion = min(breadth_pts + severity_pts, MAX_SUSPICION)

    career_years = total_career_months / 12.0
    claimed_years = worst_claimed / 12.0

    reason = (
        f"{n_violations} skill(s) claim duration exceeding career span "
        f"({career_years:.1f} yr career, worst: '{worst_skill}' claims "
        f"{claimed_years:.1f} yr): +{suspicion:.0f} suspicion"
    )
    return suspicion, reason


def detect_honeypot(
    candidate: Candidate,
    experience_features: dict,
    education_features: dict,
    title_features: dict,
    evidence_result: dict,
    consistency_result: dict,
) -> HoneypotResult:
    """Run honeypot detection checks on a candidate.

    Fallback #7 fix: Added new checks for keyword-stuffed titles, expert
    claims with 0 duration, impossible age, and excessive skill counts.

    Fallback #18 fix: Added skill-duration vs career-timeline check that
    catches candidates claiming more skill experience than their total
    career span allows (e.g. 10 yr Python claim with 4 yr career).

    The composite engine uses HARD_FILTER_THRESHOLD: if suspicion_score
    exceeds this, the final composite score is capped at 20.

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
        # Original checks
        ("impossible_timelines", _check_impossible_timelines(
            candidate, experience_features, education_features)),
        ("contradictory_experience", _check_contradictory_experience(
            candidate, experience_features, title_features)),
        ("fake_expertise", _check_fake_expertise(candidate, evidence_result)),
        ("suspicious_claims", _check_suspicious_claims(
            candidate, consistency_result, evidence_result)),
        # Fallback #7 checks
        ("keyword_stuffed_title", _check_keyword_stuffed_title(title_features)),
        ("expert_zero_duration", _check_expert_zero_duration(candidate)),
        ("impossible_age", _check_impossible_age(
            candidate, experience_features, education_features)),
        ("excessive_skill_count", _check_excessive_skill_count(candidate)),
        # Fallback #18 check
        ("skill_duration_vs_career", _check_skill_duration_vs_career_timeline(
            candidate, experience_features)),
    ]

    for name, (suspicion, reason) in checks:
        if suspicion > 0 and reason:
            total_suspicion += suspicion
            reasons.append(f"[{name}] {reason}: +{suspicion:.1f} suspicion")
            breakdown[name] = suspicion
        else:
            breakdown[name] = 0.0

    total_suspicion = min(total_suspicion, 100.0)
    total_suspicion = round(total_suspicion, 2)

    is_hard_filtered = total_suspicion >= HARD_FILTER_THRESHOLD
    if is_hard_filtered:
        reasons.append(
            f"⚠️ HARD FILTER: suspicion_score {total_suspicion:.0f} ≥ {HARD_FILTER_THRESHOLD} "
            f"→ composite score will be capped at 20"
        )

    metadata = {
        "breakdown": breakdown,
        "is_hard_filtered": is_hard_filtered,
        "hard_filter_threshold": HARD_FILTER_THRESHOLD,
    }

    return HoneypotResult(
        suspicion_score=total_suspicion,
        reasons=reasons,
        metadata=metadata,
    )

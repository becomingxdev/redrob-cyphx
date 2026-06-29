"""Experience scoring engine for the REDROB candidate ranking system.

Evaluates years of experience, domain relevance, and career progression.
Rewards upward career trajectories and penalizes erratic paths.

This module is purely responsible for experience scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import ScoreResult


# Seniority keywords mapped to ordinal scores for progression detection.
_SENIORITY_KEYWORDS: dict[str, float] = {
    "principal": 5.0,
    "distinguished": 5.0,
    "fellow": 5.0,
    "director": 4.5,
    "vp": 4.5,
    "chief": 4.5,
    "cto": 4.5,
    "ceo": 4.5,
    "lead": 3.5,
    "head": 3.5,
    "manager": 3.5,
    "architect": 3.5,
    "senior": 2.5,
    "sr": 2.5,
    "sr.": 2.5,
    "mid": 1.5,
    "junior": 0.5,
    "jr": 0.5,
    "jr.": 0.5,
    "entry": 0.5,
    "associate": 0.5,
    "trainee": 0.5,
    "intern": 0.5,
}


def _title_seniority_value(title: str) -> float:
    """Extract a numeric seniority value from a job title.

    Returns a value in [0.5, 5.0]. Defaults to 1.5 (mid) if no keyword found.
    """
    if not title:
        return 1.5
    lower = title.lower()
    for keyword, value in _SENIORITY_KEYWORDS.items():
        if keyword in lower:
            return value
    return 1.5


def _measure_progression(title_progression: list[str]) -> tuple[float, str]:
    """Measure career progression quality from chronological title list.

    Returns:
        Tuple of (progression score in [0, 1], reason string).
    """
    if len(title_progression) < 2:
        return 0.0, "Insufficient roles to measure progression"

    values = [_title_seniority_value(t) for t in title_progression]
    upward_steps = 0
    flat_steps = 0
    downward_steps = 0

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff > 0.3:
            upward_steps += 1
        elif diff < -0.3:
            downward_steps += 1
        else:
            flat_steps += 1

    total_steps = len(values) - 1
    # Progression ratio: upward / total, with a penalty for downward moves.
    progression_ratio = (upward_steps - 0.5 * downward_steps) / total_steps
    progression_ratio = max(0.0, min(1.0, progression_ratio))

    if progression_ratio >= 0.5:
        reason = f"Strong upward progression ({upward_steps}/{total_steps} steps up)"
    elif progression_ratio >= 0.2:
        reason = f"Moderate progression ({upward_steps}/{total_steps} steps up)"
    else:
        reason = f"Flat or downward trajectory ({downward_steps} downward steps)"

    return progression_ratio, reason


def score_experience(
    candidate: Candidate,
    experience_features: dict,
    career_features: dict,
    target_years: float | None = None,
) -> ScoreResult:
    """Score a candidate's experience profile.

    Args:
        candidate: The candidate to score.
        experience_features: Output from ``ExperienceExtractor.extract()``.
        career_features: Output from ``CareerExtractor.extract()``.
        target_years: Optional ideal years of experience for the role.

    Returns:
        A ``ScoreResult`` with score in [0, 100].
    """
    reasons: list[str] = []
    score = 0.0

    yoe = experience_features.get("years_of_experience", 0.0)
    total_positions = experience_features.get("total_positions", 0)
    total_months = experience_features.get("total_months_worked", 0)
    gaps_months = experience_features.get("employment_gaps_months", 0.0)
    title_progression = career_features.get("title_progression", [])
    has_promotion = career_features.get("promotion_indicators", False)
    stability = career_features.get("career_stability_metrics", {})

    # --- Years of experience scoring ---
    if yoe <= 0:
        reasons.append("No experience recorded")
        return ScoreResult(
            score=0.0,
            reasons=reasons,
            metadata={"yoe": 0.0, "progression": 0.0, "stability": 0.0},
        )

    if target_years is not None and target_years > 0:
        # Score based on proximity to target years.
        ratio = min(yoe / target_years, 2.0)
        if ratio <= 1.0:
            yoe_score = ratio * 30.0
        else:
            # Over-qualified but not penalized beyond a point.
            yoe_score = 30.0 - max(0.0, (ratio - 1.0) * 10.0)
        yoe_score = max(5.0, yoe_score)
        reasons.append(f"YoE {yoe:.1f} vs target {target_years:.0f}: +{yoe_score:.1f}")
    else:
        # General YoE scoring: diminishing returns after 10 years.
        yoe_score = min(yoe * 3.0, 30.0)
        reasons.append(f"Years of experience ({yoe:.1f}): +{yoe_score:.1f}")

    score += yoe_score

    # --- Career progression ---
    progression_ratio, progression_reason = _measure_progression(title_progression)
    progression_score = progression_ratio * 25.0
    score += progression_score
    reasons.append(f"{progression_reason}: +{progression_score:.1f}")

    # --- Promotion bonus ---
    if has_promotion:
        score += 5.0
        reasons.append("Internal promotion detected: +5.0")

    # --- Stability bonus ---
    avg_tenure = stability.get("average_tenure_months", 0.0)
    hopping_index = stability.get("job_hopping_index", 0.0)

    if avg_tenure > 0:
        # Reward longer average tenures (cap at 24 months = +15).
        stability_score = min(avg_tenure * 0.625, 15.0)
        score += stability_score
        reasons.append(f"Average tenure {avg_tenure:.1f} months: +{stability_score:.1f}")

    # Penalize excessive job hopping.
    if hopping_index > 1.5:
        hop_penalty = min((hopping_index - 1.5) * 5.0, 10.0)
        score -= hop_penalty
        reasons.append(f"Job hopping index {hopping_index:.2f}: -{hop_penalty:.1f}")

    # --- Employment gap penalty (mild) ---
    if gaps_months > 12:
        gap_penalty = min((gaps_months - 12) * 0.5, 5.0)
        score -= gap_penalty
        reasons.append(f"Employment gaps ({gaps_months:.1f} months): -{gap_penalty:.1f}")

    # --- Multiple positions bonus (shows breadth) ---
    if total_positions >= 3:
        breadth_bonus = min((total_positions - 2) * 2.0, 6.0)
        score += breadth_bonus
        reasons.append(f"Multiple positions ({total_positions}): +{breadth_bonus:.1f}")

    # Clamp to [0, 100].
    score = max(0.0, min(100.0, round(score, 2)))

    metadata = {
        "yoe": yoe,
        "progression": round(progression_ratio, 4),
        "stability": round(avg_tenure, 2),
        "hopping_index": round(hopping_index, 2),
        "total_positions": total_positions,
    }

    return ScoreResult(score=score, reasons=reasons, metadata=metadata)

"""Experience scoring engine for the REDROB candidate ranking system.

Evaluates years of experience, domain relevance, career progression, and
company quality (product vs. IT services). Also incorporates career
description signals for production ML experience.

Fallback #6 fix: Distinguishes product companies from IT services companies.
  - Top-tier companies (Google, Meta, etc.) → bonus
  - Product / startup companies → bonus
  - IT services (TCS, Infosys, etc.) → penalty

Fallback #9 fix: Uses career description analysis from CareerExtractor.
  - Production ML deployment signals → bonus
  - Research-only profile → penalty
  - Domain-relevant work (ranking/retrieval) → bonus

This module is purely responsible for experience scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import ScoreResult
from src.jd_config import JD


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
    progression_ratio = (upward_steps - 0.5 * downward_steps) / total_steps
    progression_ratio = max(0.0, min(1.0, progression_ratio))

    if progression_ratio >= 0.5:
        reason = f"Strong upward progression ({upward_steps}/{total_steps} steps up)"
    elif progression_ratio >= 0.2:
        reason = f"Moderate progression ({upward_steps}/{total_steps} steps up)"
    else:
        reason = f"Flat or downward trajectory ({downward_steps} downward steps)"

    return progression_ratio, reason


def _score_company_quality(career_features: dict) -> tuple[float, list[str]]:
    """Score company quality — product/top-tier bonuses, IT services penalties.

    Fallback #6 fix.

    Returns:
        Tuple of (net quality score, list of reason strings).
    """
    reasons: list[str] = []
    quality_score = 0.0

    it_services_count = career_features.get("it_services_count", 0)
    top_tier_count = career_features.get("top_tier_count", 0)
    product_count = career_features.get("product_count", 0)
    it_services_fraction = career_features.get("it_services_fraction", 0.0)
    total = it_services_count + top_tier_count + product_count

    if total == 0:
        return 0.0, []

    # Top-tier companies (FAANG, well-known AI labs)
    if top_tier_count > 0:
        top_tier_bonus = min(top_tier_count * 10.0, 15.0)
        quality_score += top_tier_bonus
        reasons.append(f"Top-tier company experience ({top_tier_count} companies): +{top_tier_bonus:.1f}")

    # Pure product / startup companies
    if product_count > 0 and it_services_fraction < 0.5:
        product_bonus = min(product_count * 5.0, 10.0)
        quality_score += product_bonus
        reasons.append(f"Product/startup company experience ({product_count} companies): +{product_bonus:.1f}")

    # IT services penalty
    if it_services_count > 0:
        it_penalty = min(it_services_count * 5.0, 15.0)
        quality_score -= it_penalty
        reasons.append(f"IT services company experience ({it_services_count} companies): -{it_penalty:.1f}")

    # Extra penalty for consulting-only career (≥80% IT services)
    if it_services_fraction >= 0.8 and total >= 2:
        consulting_penalty = 10.0
        quality_score -= consulting_penalty
        reasons.append(
            f"Consulting-heavy career ({it_services_fraction:.0%} IT services): -{consulting_penalty:.1f}"
        )

    return quality_score, reasons


def _score_career_description(career_features: dict) -> tuple[float, list[str]]:
    """Score based on career description analysis for JD relevance.

    Fallback #9 fix.

    Returns:
        Tuple of (description score, list of reason strings).
    """
    reasons: list[str] = []
    desc_score = 0.0

    production_ml = career_features.get("production_ml_signal", False)
    research_only = career_features.get("research_only_signal", False)
    domain_relevance = career_features.get("domain_relevance_signal", False)
    prod_count = career_features.get("total_production_signals", 0)
    domain_count = career_features.get("total_domain_signals", 0)

    if production_ml:
        prod_bonus = min(prod_count * 2.0, 12.0)
        desc_score += prod_bonus
        reasons.append(f"Production ML deployment evidence ({prod_count} signals): +{prod_bonus:.1f}")
    elif research_only:
        desc_score -= 10.0
        reasons.append("Research-only profile detected (no production deployment signals): -10.0")

    if domain_relevance:
        domain_bonus = min(domain_count * 2.0, 12.0)
        desc_score += domain_bonus
        reasons.append(
            f"Domain-relevant work (ranking/retrieval/search, {domain_count} signals): +{domain_bonus:.1f}"
        )

    return desc_score, reasons


def score_experience(
    candidate: Candidate,
    experience_features: dict,
    career_features: dict,
    target_years: float | None = None,
) -> ScoreResult:
    """Score a candidate's experience profile.

    Fallback #6 + #9 fix: adds product-company quality scoring and career
    description analysis for production ML / domain relevance signals.

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
        # Score based on proximity to target years using ideal range
        # JD ideal: 5-9 years, midpoint 7
        ideal_min = JD.get("ideal_yoe_min") or (target_years - 2)
        ideal_max = JD.get("ideal_yoe_max") or (target_years + 2)

        if ideal_min <= yoe <= ideal_max:
            # In the sweet spot
            yoe_score = 30.0
            reasons.append(f"YoE {yoe:.1f} in ideal range ({ideal_min:.0f}-{ideal_max:.0f} years): +{yoe_score:.1f}")
        elif yoe < ideal_min:
            # Under-experienced
            ratio = yoe / ideal_min
            yoe_score = max(5.0, ratio * 25.0)
            reasons.append(f"YoE {yoe:.1f} below ideal min ({ideal_min:.0f} years): +{yoe_score:.1f}")
        else:
            # Over-experienced (diminishing returns)
            excess = yoe - ideal_max
            yoe_score = max(15.0, 30.0 - excess * 2.0)
            reasons.append(f"YoE {yoe:.1f} above ideal max ({ideal_max:.0f} years, over-experienced): +{yoe_score:.1f}")
    else:
        # Generic YoE scoring: diminishing returns after 10 years
        yoe_score = min(yoe * 3.0, 30.0)
        reasons.append(f"Years of experience ({yoe:.1f}): +{yoe_score:.1f}")

    score += yoe_score

    # --- Career progression ---
    progression_ratio, progression_reason = _measure_progression(title_progression)
    progression_score = progression_ratio * 20.0
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
        stability_score = min(avg_tenure * 0.5, 10.0)
        score += stability_score
        reasons.append(f"Average tenure {avg_tenure:.1f} months: +{stability_score:.1f}")

    if hopping_index > 1.5:
        hop_penalty = min((hopping_index - 1.5) * 5.0, 10.0)
        score -= hop_penalty
        reasons.append(f"Job hopping index {hopping_index:.2f}: -{hop_penalty:.1f}")

    # --- Employment gap penalty (mild) ---
    if gaps_months > 12:
        gap_penalty = min((gaps_months - 12) * 0.5, 5.0)
        score -= gap_penalty
        reasons.append(f"Employment gaps ({gaps_months:.1f} months): -{gap_penalty:.1f}")

    # --- Multiple positions bonus (breadth) ---
    if total_positions >= 3:
        breadth_bonus = min((total_positions - 2) * 1.5, 5.0)
        score += breadth_bonus
        reasons.append(f"Multiple positions ({total_positions}): +{breadth_bonus:.1f}")

    # --- Company quality scoring (Fallback #6) ---
    quality_score, quality_reasons = _score_company_quality(career_features)
    if quality_score != 0:
        score += quality_score
        reasons.extend(quality_reasons)

    # --- Career description analysis (Fallback #9) ---
    desc_score, desc_reasons = _score_career_description(career_features)
    if desc_score != 0:
        score += desc_score
        reasons.extend(desc_reasons)

    # Clamp to [0, 100].
    score = max(0.0, min(100.0, round(score, 2)))

    metadata = {
        "yoe": yoe,
        "progression": round(progression_ratio, 4),
        "stability": round(avg_tenure, 2),
        "hopping_index": round(hopping_index, 2),
        "total_positions": total_positions,
        "it_services_fraction": career_features.get("it_services_fraction", 0.0),
        "top_tier_count": career_features.get("top_tier_count", 0),
        "production_ml_signal": career_features.get("production_ml_signal", False),
        "domain_relevance_signal": career_features.get("domain_relevance_signal", False),
        "companies": career_features.get("companies_worked_for", []),
    }

    return ScoreResult(score=score, reasons=reasons, metadata=metadata)

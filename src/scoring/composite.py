"""Composite scoring engine for the REDROB candidate ranking system.

This is the ONLY module that combines individual scoring components into
a final score. It receives pre-computed outputs from every other scoring
engine and assembles them using configurable weights.

Pseudo-flow:
    weighted score
        → confidence adjustment
        → consistency bonus
        → penalty multiplier
        → honeypot multiplier (+ hard cap at 20 for high-suspicion)
        → final score

This module does NOT sort or rank candidates.
"""

from __future__ import annotations

from src.scoring import (
    FinalScore,
    HoneypotResult,
    PenaltyResult,
    ScoreResult,
    load_weights,
)


def _compute_weighted_score(
    component_scores: dict[str, ScoreResult],
    weights: dict[str, float],
) -> float:
    """Compute a weighted sum of component scores.

    Args:
        component_scores: Mapping of component name to its ``ScoreResult``.
        weights: Weight configuration from ``config/weights.yaml``.

    Returns:
        Weighted score in [0, 100].
    """
    weighted = 0.0
    total_weight = 0.0

    for name, weight in weights.items():
        if name in ("penalty_multiplier", "honeypot_multiplier"):
            continue  # These are multipliers, not component weights.
        if name in component_scores:
            result = component_scores[name]
            weighted += result.score * weight
            total_weight += weight

    if total_weight <= 0:
        return 0.0

    return weighted / total_weight


def _apply_confidence_adjustment(
    raw_score: float,
    confidence_result: dict,
    confidence_weight: float,
) -> float:
    """Adjust score based on evidence confidence.

    Low confidence scales the score down. High confidence preserves it.

    Args:
        raw_score: The score before confidence adjustment.
        confidence_result: Output from ``ConfidenceCalculator.calculate()``.
        confidence_weight: Weight of confidence in [0, 1].

    Returns:
        Adjusted score.
    """
    confidence = confidence_result.get("confidence", 0.5) if confidence_result else 0.5
    # Adjustment: blend raw score toward (raw_score * confidence).
    # With weight=0.05, a confidence of 0.5 reduces score by 2.5%.
    factor = 1.0 - confidence_weight + confidence_weight * confidence
    return raw_score * factor


def _apply_consistency_bonus(
    score: float,
    consistency_result: dict,
    consistency_weight: float,
) -> float:
    """Apply a small bonus or penalty based on profile consistency.

    Args:
        score: The score after confidence adjustment.
        consistency_result: Output from ``ConsistencyAnalyzer.analyze()``.
        consistency_weight: Weight of consistency in [0, 1].

    Returns:
        Adjusted score.
    """
    consistency = consistency_result.get("consistency_score", 1.0) if consistency_result else 1.0
    # Scale the score by a consistency factor centered at 1.0.
    # consistency_score in [0, 1] maps to factor in [1 - weight, 1 + weight].
    factor = 1.0 - consistency_weight + 2.0 * consistency_weight * consistency
    return score * factor


def _apply_penalty_multiplier(
    score: float,
    penalty: PenaltyResult,
    penalty_multiplier: float,
) -> float:
    """Apply a soft penalty multiplier.

    Higher penalties reduce the score more, but never below 0.

    Args:
        score: The score after consistency adjustment.
        penalty: Output from the penalty engine.
        penalty_multiplier: Global penalty multiplier from config.

    Returns:
        Adjusted score.
    """
    if penalty.penalty_score <= 0:
        return score

    # Map penalty (0-30) to a multiplier (1.0 down to 0.7).
    penalty_factor = 1.0 - (penalty.penalty_score / 100.0) * penalty_multiplier
    penalty_factor = max(0.5, penalty_factor)  # Never reduce by more than 50%.

    return score * penalty_factor


def _apply_honeypot_multiplier(
    score: float,
    honeypot: HoneypotResult,
    honeypot_multiplier: float,
) -> float:
    """Apply a honeypot down-weighting multiplier with hard filter.

    Fallback #7 fix: If suspicion_score exceeds the hard filter threshold
    (from jd_config.yaml), the score is hard-capped at 20.0, effectively
    excluding the candidate from the top-100 rankings.

    For scores below the threshold, a soft multiplier is applied.

    Args:
        score: The score after penalty adjustment.
        honeypot: Output from the honeypot engine.
        honeypot_multiplier: Global honeypot multiplier from config.

    Returns:
        Adjusted score.
    """
    if honeypot.suspicion_score <= 0:
        return score

    # Hard filter (Fallback #7)
    is_hard_filtered = (honeypot.metadata or {}).get("is_hard_filtered", False)
    if is_hard_filtered:
        return min(score, 20.0)  # Hard cap — candidate excluded from top-100

    # Soft multiplier for lower suspicion scores
    suspicion_factor = 1.0 - (honeypot.suspicion_score / 200.0) * honeypot_multiplier
    suspicion_factor = max(0.4, suspicion_factor)  # Never reduce below 40%.

    return score * suspicion_factor


def compose_score(
    candidate_id: str,
    title_score: ScoreResult,
    skill_score: ScoreResult,
    experience_score: ScoreResult,
    education_score: ScoreResult,
    behavior_score: ScoreResult,
    penalty: PenaltyResult,
    confidence_result: dict,
    consistency_result: dict,
    honeypot: HoneypotResult,
    location_score: ScoreResult | None = None,
    weights: dict[str, float] | None = None,
) -> FinalScore:
    """Compose all scoring components into a single final score.

    Fallback #7 fix: honeypot hard filter now caps score at 20 for
    candidates exceeding the suspicion threshold.

    Pseudo-flow:
        weighted_score
            → confidence_adjustment
            → consistency_bonus
            → penalty_multiplier
            → honeypot_multiplier (+ hard cap)
            → final_score

    Args:
        candidate_id: The candidate's identifier.
        title_score: Output from the title scoring engine.
        skill_score: Output from the skill scoring engine.
        experience_score: Output from the experience scoring engine.
        education_score: Output from the education scoring engine.
        behavior_score: Output from the behavior scoring engine.
        penalty: Output from the penalty engine.
        confidence_result: Output from ``ConfidenceCalculator.calculate()``.
        consistency_result: Output from ``ConsistencyAnalyzer.analyze()``.
        honeypot: Output from the honeypot engine.
        location_score: Optional output from the location scoring engine.
        weights: Optional weight overrides. If ``None``, loads from config.

    Returns:
        A ``FinalScore`` with the composite score and all component details.
    """
    if weights is None:
        weights = load_weights()

    # Map component scores to the weighted sum.
    component_scores = {
        "title": title_score,
        "skills": skill_score,
        "experience": experience_score,
        "education": education_score,
        "behavior": behavior_score,
    }
    if location_score is not None and "location" in weights:
        component_scores["location"] = location_score

    # --- Step 1: Weighted score ---
    raw_score = _compute_weighted_score(component_scores, weights)

    # --- Step 2: Confidence adjustment ---
    confidence_weight = weights.get("confidence", 0.05)
    after_confidence = _apply_confidence_adjustment(
        raw_score, confidence_result, confidence_weight
    )

    # --- Step 3: Consistency bonus ---
    consistency_weight = weights.get("consistency", 0.05)
    after_consistency = _apply_consistency_bonus(
        after_confidence, consistency_result, consistency_weight
    )

    # --- Step 4: Penalty multiplier ---
    penalty_multiplier = weights.get("penalty_multiplier", 1.0)
    after_penalty = _apply_penalty_multiplier(
        after_consistency, penalty, penalty_multiplier
    )

    # --- Step 5: Honeypot multiplier ---
    honeypot_multiplier = weights.get("honeypot_multiplier", 1.0)
    after_honeypot = _apply_honeypot_multiplier(
        after_penalty, honeypot, honeypot_multiplier
    )

    # --- Clamp to [0, 100] ---
    final = max(0.0, min(100.0, round(after_honeypot, 2)))

    # --- Aggregate reasons ---
    all_reasons: list[str] = []
    for name, result in component_scores.items():
        all_reasons.append(f"[{name}] score={result.score:.1f}")
    all_reasons.append(f"[confidence] {confidence_result.get('confidence', 0):.2f}")
    all_reasons.append(f"[consistency] {consistency_result.get('consistency_score', 0):.2f}")
    if penalty.penalty_score > 0:
        all_reasons.append(f"[penalty] -{penalty.penalty_score:.1f}")
    if honeypot.suspicion_score > 0:
        all_reasons.append(f"[honeypot] suspicion={honeypot.suspicion_score:.1f}")

    # Store individual component scores for transparency.
    component_score_values = {
        name: result.score for name, result in component_scores.items()
    }
    component_score_values["confidence"] = confidence_result.get("confidence", 0.0)
    component_score_values["consistency"] = consistency_result.get("consistency_score", 0.0)

    return FinalScore(
        candidate_id=candidate_id,
        score=final,
        component_scores=component_score_values,
        penalty_result=penalty,
        honeypot_result=honeypot,
        reasons=all_reasons,
    )

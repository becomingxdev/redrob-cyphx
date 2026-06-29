"""Title scoring engine for the REDROB candidate ranking system.

Evaluates how well a candidate's current title matches the expected target
title, factoring in seniority alignment and AI/ML relevance.

This module is purely responsible for title scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

import re

from src.models.candidate import Candidate
from src.scoring import ScoreResult


# Seniority hierarchy ordered from most senior to least.
_SENIORITY_ORDER: list[str] = ["principal", "staff", "lead", "senior", "mid", "junior"]


def _seniority_index(seniority: str) -> int:
    """Return ordinal position in the seniority hierarchy. Higher = more senior."""
    try:
        return _SENIORITY_ORDER.index(seniority)
    except ValueError:
        return _SENIORITY_ORDER.index("mid")


def _tokenize(text: str) -> list[str]:
    """Lower-case, tokenize, and deduplicate words from text."""
    if not text:
        return []
    return sorted(set(re.findall(r"\b\w+\b", text.lower())))


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard index between two sets. Returns 0.0 if both are empty."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def score_title(
    candidate: Candidate,
    title_features: dict,
    target_title: str | None = None,
) -> ScoreResult:
    """Score a candidate's title relevance.

    Args:
        candidate: The candidate to score.
        title_features: Output from ``TitleExtractor.extract()``.
        target_title: Optional target job title for exact/close matching.
            If ``None``, scoring is based purely on signal strength
            (seniority, AI relevance, completeness).

    Returns:
        A ``ScoreResult`` with score in [0, 100].
    """
    reasons: list[str] = []
    score = 0.0

    # --- Signal strength: completeness ---
    title = title_features.get("title", "")
    normalized = title_features.get("normalized_title", "")
    tokens = title_features.get("tokens", [])
    seniority = title_features.get("seniority", "mid")
    is_manager = title_features.get("is_manager", False)
    is_ai_related = title_features.get("is_ai_related", False)

    # Base score for having a title at all.
    if title:
        score += 15.0
        reasons.append(f"Title present: '{title}'")
    else:
        reasons.append("No title provided")
        return ScoreResult(score=0.0, reasons=reasons, metadata={"title": "", "seniority": "", "is_ai_related": False, "is_manager": False})

    # --- Seniority scoring ---
    # _SENIORITY_ORDER is most-senior-first, so a LOWER index = MORE senior.
    # Reward seniority: map index to a bonus where the most senior earns 25.
    last_index = len(_SENIORITY_ORDER) - 1
    seniority_idx = _seniority_index(seniority)
    seniority_rank = last_index - seniority_idx  # higher = more senior
    seniority_bonus = min(seniority_rank * 5.0, 25.0)
    score += seniority_bonus
    reasons.append(f"Seniority '{seniority}': +{seniority_bonus:.1f}")

    # --- AI/ML relevance bonus ---
    if is_ai_related:
        score += 15.0
        reasons.append("AI/ML related title: +15.0")
    else:
        reasons.append("Title not AI/ML related")

    # --- Management bonus (small) ---
    if is_manager:
        score += 5.0
        reasons.append("Management title: +5.0")

    # --- Target title matching (if provided) ---
    if target_title:
        target_lower = target_title.lower()
        target_tokens = set(_tokenize(target_title))
        title_tokens = set(tokens)

        # Exact match
        if normalized == target_lower:
            score += 30.0
            reasons.append(f"Exact title match with '{target_title}': +30.0")
        else:
            # Token overlap (Jaccard similarity)
            similarity = _jaccard_similarity(target_tokens, title_tokens)
            if similarity > 0.0:
                overlap_bonus = min(similarity * 25.0, 25.0)
                score += overlap_bonus
                reasons.append(
                    f"Token overlap with '{target_title}' "
                    f"(similarity={similarity:.2f}): +{overlap_bonus:.1f}"
                )
            else:
                score -= 5.0
                reasons.append(f"No token overlap with target '{target_title}': -5.0")

    # Clamp to [0, 100].
    score = max(0.0, min(100.0, round(score, 2)))

    metadata = {
        "title": title,
        "seniority": seniority,
        "is_ai_related": is_ai_related,
        "is_manager": is_manager,
        "target_title": target_title or None,
    }

    return ScoreResult(score=score, reasons=reasons, metadata=metadata)

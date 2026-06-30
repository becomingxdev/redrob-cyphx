"""Title scoring engine for the REDROB candidate ranking system.

Evaluates how well a candidate's current title matches the expected target
title, factoring in seniority alignment and AI/ML relevance.

Fallback #11 fix: When a target_title is provided (JD-aware mode), the
seniority preference is INVERTED for this IC engineering role:
  - "senior" → highest bonus (JD wants senior ICs)
  - "lead/manager/principal/director" → penalty (JD wants code-writers)
  - "staff" → mild penalty
  - "mid" → neutral
  - "junior" → small bonus (potential, still relevant)

The JD says: "People who haven't written production code in the last 18 months
because they've moved into architecture or tech lead roles — we will probably
not move forward."

This module is purely responsible for title scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

import re

from src.models.candidate import Candidate
from src.scoring import ScoreResult


# Seniority hierarchy ordered from most senior to least.
_SENIORITY_ORDER: list[str] = ["principal", "staff", "lead", "senior", "mid", "junior"]

# Domain keywords used to detect if career history confirms title domain.
_CAREER_DOMAIN_KEYWORDS: frozenset[str] = frozenset([
    # AI / ML
    "machine learning", "ml", "deep learning", "nlp", "llm", "ai", "neural",
    "data science", "data scientist", "computer vision", "reinforcement",
    # Engineering
    "software engineer", "backend", "frontend", "full stack", "fullstack",
    "data engineer", "platform engineer", "site reliability", "sre",
    "devops", "cloud", "distributed",
    # Analytics
    "analyst", "analytics", "business intelligence", "bi",
])

# Management/architecture indicators — penalised for IC roles
_MANAGEMENT_KEYWORDS: frozenset[str] = frozenset([
    "manager", "director", "vp", "vice president", "cto", "chief",
    "head of", "engineering manager", "em ",
])

_ARCHITECT_KEYWORDS: frozenset[str] = frozenset([
    "architect", "solution architect", "enterprise architect", "technical architect",
])


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


def _is_management_role(title: str) -> bool:
    """Return True if the title implies a management or pure-management role."""
    t = title.lower()
    return any(kw in t for kw in _MANAGEMENT_KEYWORDS)


def _is_architect_only_role(title: str, is_manager: bool) -> bool:
    """Return True if the title is a pure architect role (not IC engineer)."""
    t = title.lower()
    if not any(kw in t for kw in _ARCHITECT_KEYWORDS):
        return False
    # Hybrid like "ML Architect" who also codes is acceptable
    # Pure architect without engineering context → flag
    engineering_markers = ["engineer", "ml", "ai", "data", "software", "platform"]
    has_engineering_context = any(m in t for m in engineering_markers)
    return not has_engineering_context


def score_title(
    candidate: Candidate,
    title_features: dict,
    target_title: str | None = None,
    career_features: dict | None = None,
) -> ScoreResult:
    """Score a candidate's title relevance.

    Fallback #11 fix: When target_title is provided, seniority preference is
    inverted — senior IC roles score highest; management/principal/lead penalised.

    Args:
        candidate: The candidate to score.
        title_features: Output from ``TitleExtractor.extract()``.
        target_title: Optional target job title for exact/close matching.
            When provided, activates JD-aware IC-engineer seniority scoring.
        career_features: Optional output from ``CareerExtractor.extract()``.
            Used for career-context validation bonus.

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
        return ScoreResult(score=0.0, reasons=reasons, metadata={
            "title": "", "seniority": "", "is_ai_related": False, "is_manager": False
        })

    # --- AI/ML relevance bonus ---
    if is_ai_related:
        score += 15.0
        reasons.append("AI/ML related title: +15.0")
    else:
        reasons.append("Title not AI/ML related")

    # --- Seniority scoring ---
    # Fallback #11: When target_title provided (JD-aware mode), prefer IC "senior"
    # and penalise management/architecture titles.
    jd_aware = target_title is not None

    if jd_aware:
        # JD-specific IC seniority preference
        title_lower = title.lower()

        # Management roles: JD explicitly rejects these
        if _is_management_role(title_lower) or is_manager:
            score -= 15.0
            reasons.append(f"Management/director title detected for IC role: -15.0")
        # Pure architect roles
        elif _is_architect_only_role(title_lower, is_manager):
            score -= 10.0
            reasons.append(f"Architecture-only title for IC coding role: -10.0")
        # Principal/staff: over-senior for hands-on IC role
        elif seniority in ("principal", "staff"):
            score -= 5.0
            reasons.append(f"Seniority '{seniority}' (over-senior for IC role): -5.0")
        # Lead: ambiguous but leans management
        elif seniority == "lead":
            # Small penalty — could be tech lead who still codes
            score += 5.0
            reasons.append(f"Seniority 'lead' (possible tech lead): +5.0")
        # Senior: ideal match for this JD
        elif seniority == "senior":
            score += 25.0
            reasons.append(f"Seniority 'senior' (ideal for IC role): +25.0")
        # Mid-level: acceptable
        elif seniority == "mid":
            score += 10.0
            reasons.append(f"Seniority 'mid': +10.0")
        # Junior: lower but not penalised
        elif seniority == "junior":
            score += 5.0
            reasons.append(f"Seniority 'junior': +5.0")
        else:
            score += 10.0
            reasons.append(f"Seniority '{seniority}': +10.0")
    else:
        # Generic mode (no JD): original seniority scoring
        last_index = len(_SENIORITY_ORDER) - 1
        seniority_idx = _seniority_index(seniority)
        seniority_rank = last_index - seniority_idx
        seniority_bonus = min(seniority_rank * 5.0, 25.0)
        score += seniority_bonus
        reasons.append(f"Seniority '{seniority}': +{seniority_bonus:.1f}")

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

    # --- Career context validation bonus ---
    if career_features and is_ai_related:
        past_titles = career_features.get("title_progression") or []
        industries = career_features.get("industries_worked_in") or []
        career_text = " ".join(
            str(t) for t in (past_titles + industries)
        ).lower()
        domain_hits = sum(1 for kw in _CAREER_DOMAIN_KEYWORDS if kw in career_text)
        if domain_hits >= 3:
            score += 10.0
            reasons.append(f"Career history confirms AI/ML/Eng domain ({domain_hits} signals): +10.0")
        elif domain_hits >= 1:
            score += 5.0
            reasons.append(f"Career history partially confirms domain ({domain_hits} signal): +5.0")
        else:
            reasons.append("Career history does not confirm AI/ML title domain")

    # Clamp to [0, 100].
    score = max(0.0, min(100.0, round(score, 2)))

    metadata = {
        "title": title,
        "seniority": seniority,
        "is_ai_related": is_ai_related,
        "is_manager": is_manager,
        "target_title": target_title or None,
        "jd_aware": jd_aware,
    }

    return ScoreResult(score=score, reasons=reasons, metadata=metadata)

"""Behavior scoring engine for the REDROB candidate ranking system.

Evaluates extracurricular professional behavior signals: open source
contributions, mentoring, speaking, leadership, and platform engagement.

This module is purely responsible for behavior scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import ScoreResult


# Behavior indicators extracted from profile signals.
# These keywords signal community involvement, leadership, or thought leadership.
_LEADERSHIP_KEYWORDS: list[str] = [
    "mentor", "cofounder", "co-founder", "founder", "board member",
    "organizer", "speaker", "instructor", "advisor", "consultant",
    "volunteer", "open source", "oss", "community", "blog",
    "published", "author", "columnist", "podcast",
]


def _extract_behavior_signals(candidate: Candidate) -> dict:
    """Extract behavior-related signals from candidate profile and raw data.

    Returns a dict of behavior indicators:
        - github_activity: float (from redrob_signals)
        - connection_count: int
        - endorsements_received: int
        - profile_views_30d: int
        - has_leadership_keywords: bool
        - verified_count: int (email + phone + linkedin)
        - profile_completeness: float
    """
    signals = {
        "github_activity": 0.0,
        "connection_count": 0,
        "endorsements_received": 0,
        "profile_views_30d": 0,
        "has_leadership_keywords": False,
        "verified_count": 0,
        "profile_completeness": 0.0,
    }

    redrob = candidate.redrob_signals
    if redrob:
        signals["github_activity"] = getattr(redrob, "github_activity_score", 0.0) or 0.0
        signals["connection_count"] = getattr(redrob, "connection_count", 0) or 0
        signals["endorsements_received"] = getattr(redrob, "endorsements_received", 0) or 0
        signals["profile_views_30d"] = getattr(redrob, "profile_views_received_30d", 0) or 0
        signals["profile_completeness"] = getattr(redrob, "profile_completeness_score", 0.0) or 0.0

        verified_count = 0
        if getattr(redrob, "verified_email", False):
            verified_count += 1
        if getattr(redrob, "verified_phone", False):
            verified_count += 1
        if getattr(redrob, "linkedin_connected", False):
            verified_count += 1
        signals["verified_count"] = verified_count

    # Scan headline and summary for leadership/behavior keywords.
    profile = candidate.profile
    text_blob = ""
    if profile:
        headline = profile.headline or ""
        summary = profile.summary or ""
        text_blob = f"{headline} {summary}".lower()

    if text_blob:
        signals["has_leadership_keywords"] = any(
            kw in text_blob for kw in _LEADERSHIP_KEYWORDS
        )

    return signals


def score_behavior(candidate: Candidate) -> ScoreResult:
    """Score a candidate's professional behavior signals.

    Behavior is a small contribution to the overall score. It rewards
    engagement, community involvement, and platform activity.

    Args:
        candidate: The candidate to score.

    Returns:
        A ``ScoreResult`` with score in [0, 100]. Naturally capped at a
        modest range since behavior is a minor factor.
    """
    reasons: list[str] = []
    score = 0.0

    signals = _extract_behavior_signals(candidate)

    # --- GitHub activity (open source signal) ---
    github = signals["github_activity"]
    if github > 0:
        github_score = min(github * 1.5, 15.0)
        score += github_score
        reasons.append(f"GitHub activity score {github:.1f}: +{github_score:.1f}")
    else:
        reasons.append("No GitHub activity signal")

    # --- Leadership / community keywords ---
    if signals["has_leadership_keywords"]:
        score += 10.0
        reasons.append("Leadership/community keywords found: +10.0")
    else:
        reasons.append("No leadership/community keywords found")

    # --- Network engagement (connections) ---
    connections = signals["connection_count"]
    if connections > 100:
        network_score = min((connections - 100) * 0.02, 10.0)
        score += network_score
        reasons.append(f"Strong network ({connections} connections): +{network_score:.1f}")
    elif connections > 0:
        reasons.append(f"Moderate network ({connections} connections): no bonus")
    else:
        reasons.append("No connections recorded")

    # --- Endorsements ---
    endorsements = signals["endorsements_received"]
    if endorsements > 10:
        endorsement_score = min((endorsements - 10) * 0.1, 10.0)
        score += endorsement_score
        reasons.append(f"Endorsements ({endorsements}): +{endorsement_score:.1f}")

    # --- Profile completeness ---
    completeness = signals["profile_completeness"]
    if completeness > 80:
        completeness_score = (completeness - 80) * 0.1
        score += completeness_score
        reasons.append(f"Profile completeness {completeness:.1f}%: +{completeness_score:.1f}")

    # --- Verified identity ---
    verified = signals["verified_count"]
    if verified >= 3:
        score += 5.0
        reasons.append(f"Fully verified identity ({verified}/3): +5.0")
    elif verified >= 1:
        score += 2.0
        reasons.append(f"Partially verified identity ({verified}/3): +2.0")
    else:
        reasons.append("No verified identity signals")

    # --- Natural cap: behavior is a small factor ---
    score = min(score, 30.0)
    score = max(0.0, round(score, 2))

    metadata = {
        "github_activity": github,
        "has_leadership_keywords": signals["has_leadership_keywords"],
        "connections": connections,
        "endorsements": endorsements,
        "verified_count": verified,
        "profile_completeness": completeness,
    }

    return ScoreResult(score=score, reasons=reasons, metadata=metadata)

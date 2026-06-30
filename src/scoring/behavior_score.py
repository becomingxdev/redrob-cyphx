"""Behavior scoring engine for the REDROB candidate ranking system.

Evaluates candidate availability and engagement signals from the Redrob
platform. The JD explicitly prioritises available candidates:
  "A perfect-on-paper candidate who hasn't logged in for 6 months and has
   a 5% recruiter response rate is not actually available."

This rewrite (Fallback #4) incorporates 13+ Redrob signals that were
previously ignored. Availability signals are treated as availability
multipliers: a highly unavailable candidate loses score regardless of
how strong their technical profile is.

Signal hierarchy:
  1. Availability (last_active, open_to_work, response_rate) — highest weight
  2. Notice period + work mode alignment
  3. Platform engagement (GitHub, connections, endorsements)
  4. Verified identity

This module is purely responsible for behavior scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from datetime import datetime, timezone
from src.models.candidate import Candidate
from src.scoring import ScoreResult
from src.jd_config import JD_REQUIRED_SKILLS


# Leadership/community keywords (kept for community engagement bonus)
_LEADERSHIP_KEYWORDS: list[str] = [
    "mentor", "cofounder", "co-founder", "founder", "board member",
    "organizer", "speaker", "instructor", "advisor", "consultant",
    "volunteer", "open source", "oss", "community", "blog",
    "published", "author", "columnist", "podcast",
]

# Months of inactivity that constitute "unavailable"
_INACTIVE_THRESHOLD_MONTHS = 6


def _months_since(date_str: str | None) -> float | None:
    """Return months elapsed since a date string (YYYY-MM-DD or YYYY-MM).

    Returns None if date_str is missing or unparseable.
    """
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            # Make timezone-aware for comparison
            dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            delta_days = (now - dt).days
            return max(0.0, delta_days / 30.44)
        except ValueError:
            continue
    return None


def _extract_behavior_signals(candidate: Candidate) -> dict:
    """Extract all behavior-related signals from candidate's Redrob data.

    Returns a comprehensive dict of availability and engagement signals.
    """
    signals = {
        # Availability signals
        "months_since_active": None,
        "open_to_work": True,
        "recruiter_response_rate": None,
        "avg_response_time_hours": None,
        "notice_period_days": None,
        "preferred_work_mode": "",
        # Active search signals
        "applications_submitted_30d": 0,
        "saved_by_recruiters_30d": 0,
        "search_appearance_30d": 0,
        # Platform engagement
        "github_activity": 0.0,
        "connection_count": 0,
        "endorsements_received": 0,
        "profile_views_30d": 0,
        "has_leadership_keywords": False,
        "profile_completeness": 0.0,
        # Reliability
        "interview_completion_rate": None,
        "offer_acceptance_rate": None,
        # Verified identity
        "verified_count": 0,
        # Skill assessment scores (verified skills)
        "skill_assessment_scores": {},
    }

    redrob = candidate.redrob_signals
    if redrob:
        # Availability
        signals["months_since_active"] = _months_since(
            getattr(redrob, "last_active_date", None)
        )
        open_to_work = getattr(redrob, "open_to_work_flag", None)
        if open_to_work is not None:
            signals["open_to_work"] = bool(open_to_work)
        signals["recruiter_response_rate"] = getattr(redrob, "recruiter_response_rate", None) or None
        signals["avg_response_time_hours"] = getattr(redrob, "avg_response_time_hours", None) or None
        signals["notice_period_days"] = getattr(redrob, "notice_period_days", None) or None
        signals["preferred_work_mode"] = (getattr(redrob, "preferred_work_mode", "") or "").lower()

        # Active search
        signals["applications_submitted_30d"] = getattr(redrob, "applications_submitted_30d", 0) or 0
        signals["saved_by_recruiters_30d"] = getattr(redrob, "saved_by_recruiters_30d", 0) or 0
        signals["search_appearance_30d"] = getattr(redrob, "search_appearance_30d", 0) or 0

        # Platform engagement
        signals["github_activity"] = getattr(redrob, "github_activity_score", 0.0) or 0.0
        signals["connection_count"] = getattr(redrob, "connection_count", 0) or 0
        signals["endorsements_received"] = getattr(redrob, "endorsements_received", 0) or 0
        signals["profile_views_30d"] = getattr(redrob, "profile_views_received_30d", 0) or 0
        signals["profile_completeness"] = getattr(redrob, "profile_completeness_score", 0.0) or 0.0

        # Reliability
        signals["interview_completion_rate"] = getattr(redrob, "interview_completion_rate", None) or None
        signals["offer_acceptance_rate"] = getattr(redrob, "offer_acceptance_rate", None) or None

        # Verified identity
        verified_count = 0
        if getattr(redrob, "verified_email", False):
            verified_count += 1
        if getattr(redrob, "verified_phone", False):
            verified_count += 1
        if getattr(redrob, "linkedin_connected", False):
            verified_count += 1
        signals["verified_count"] = verified_count

        # Skill assessment scores
        skill_scores = getattr(redrob, "skill_assessment_scores", None)
        if isinstance(skill_scores, dict):
            signals["skill_assessment_scores"] = {
                k.lower(): v for k, v in skill_scores.items()
            }

    # Scan headline and summary for leadership keywords
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
    """Score a candidate's availability and behavioral signals.

    Fallback #4 fix: Incorporates 13+ Redrob signals that were previously
    ignored. Availability is now the primary signal — an unavailable
    candidate loses score regardless of technical profile strength.

    Args:
        candidate: The candidate to score.

    Returns:
        A ``ScoreResult`` with score in [0, 100].
    """
    reasons: list[str] = []
    score = 0.0
    penalties = 0.0

    signals = _extract_behavior_signals(candidate)

    # ================================================================
    # PART 1: AVAILABILITY SIGNALS (can heavily penalise)
    # ================================================================

    # --- Last active date ---
    months_inactive = signals["months_since_active"]
    if months_inactive is not None:
        if months_inactive > 12:
            penalties += 25.0
            reasons.append(
                f"Last active {months_inactive:.0f} months ago (>12 months): -25.0 (likely unavailable)"
            )
        elif months_inactive > 6:
            penalties += 15.0
            reasons.append(
                f"Last active {months_inactive:.0f} months ago (>6 months): -15.0 (inactive)"
            )
        elif months_inactive > 3:
            penalties += 5.0
            reasons.append(
                f"Last active {months_inactive:.0f} months ago (>3 months): -5.0"
            )
        else:
            score += 10.0
            reasons.append(
                f"Recently active ({months_inactive:.0f} months ago): +10.0"
            )
    else:
        # No activity date — mild negative signal
        penalties += 3.0
        reasons.append("No last active date recorded: -3.0")

    # --- Open to work flag ---
    if not signals["open_to_work"]:
        penalties += 10.0
        reasons.append("Not open to work (flag=false): -10.0")
    else:
        score += 8.0
        reasons.append("Open to work: +8.0")

    # --- Recruiter response rate ---
    rr = signals["recruiter_response_rate"]
    if rr is not None:
        if rr < 0.1:
            penalties += 20.0
            reasons.append(f"Very low recruiter response rate ({rr:.0%}): -20.0")
        elif rr < 0.3:
            penalties += 10.0
            reasons.append(f"Low recruiter response rate ({rr:.0%}): -10.0")
        elif rr >= 0.7:
            score += 8.0
            reasons.append(f"High recruiter response rate ({rr:.0%}): +8.0")
        elif rr >= 0.4:
            score += 4.0
            reasons.append(f"Moderate recruiter response rate ({rr:.0%}): +4.0")
    else:
        reasons.append("No recruiter response rate data")

    # --- Average response time ---
    art = signals["avg_response_time_hours"]
    if art is not None:
        if art > 168:  # > 1 week
            penalties += 5.0
            reasons.append(f"Very slow response time ({art:.0f}h): -5.0")
        elif art <= 24:
            score += 3.0
            reasons.append(f"Fast response time ({art:.0f}h): +3.0")

    # --- Notice period ---
    notice = signals["notice_period_days"]
    if notice is not None:
        if notice > 90:
            penalties += 10.0
            reasons.append(f"Long notice period ({notice} days): -10.0")
        elif notice > 60:
            penalties += 5.0
            reasons.append(f"Moderate notice period ({notice} days): -5.0")
        elif notice <= 30:
            score += 5.0
            reasons.append(f"Short notice period ({notice} days): +5.0")
        else:
            reasons.append(f"Notice period {notice} days: neutral")
    else:
        reasons.append("Notice period not specified: neutral")

    # --- Work mode alignment ---
    work_mode = signals["preferred_work_mode"]
    if work_mode in ("hybrid", "flexible", "hybrid/flexible"):
        score += 5.0
        reasons.append(f"Work mode '{work_mode}' matches JD hybrid requirement: +5.0")
    elif work_mode in ("remote", "fully remote", "wfh"):
        penalties += 5.0
        reasons.append(f"Work mode '{work_mode}' conflicts with JD hybrid requirement: -5.0")

    # ================================================================
    # PART 2: ACTIVE SEARCH SIGNALS (positive bonuses)
    # ================================================================

    apps = signals["applications_submitted_30d"]
    if apps > 5:
        score += 5.0
        reasons.append(f"Actively applying ({apps} applications in 30d): +5.0")
    elif apps > 0:
        score += 3.0
        reasons.append(f"Some application activity ({apps} in 30d): +3.0")

    saved = signals["saved_by_recruiters_30d"]
    if saved > 10:
        score += 5.0
        reasons.append(f"High recruiter demand ({saved} saves in 30d): +5.0")
    elif saved > 3:
        score += 2.0
        reasons.append(f"Moderate recruiter interest ({saved} saves in 30d): +2.0")

    # ================================================================
    # PART 3: PLATFORM ENGAGEMENT & COMMUNITY
    # ================================================================

    # --- GitHub activity ---
    github = signals["github_activity"]
    if github > 0:
        github_score = min(github * 2.5, 15.0)
        score += github_score
        reasons.append(f"GitHub activity score {github:.1f}: +{github_score:.1f}")
    else:
        reasons.append("No GitHub activity signal")

    # --- Leadership / community keywords ---
    if signals["has_leadership_keywords"]:
        score += 8.0
        reasons.append("Leadership/community keywords found: +8.0")

    # --- Network engagement ---
    connections = signals["connection_count"]
    if connections > 500:
        score += 8.0
        reasons.append(f"Large network ({connections} connections): +8.0")
    elif connections > 100:
        network_score = min((connections - 100) * 0.02, 8.0)
        score += network_score
        reasons.append(f"Network ({connections} connections): +{network_score:.1f}")

    # --- Endorsements ---
    endorsements = signals["endorsements_received"]
    if endorsements > 10:
        endorsement_score = min((endorsements - 10) * 0.1, 8.0)
        score += endorsement_score
        reasons.append(f"Endorsements ({endorsements}): +{endorsement_score:.1f}")

    # ================================================================
    # PART 4: SKILL ASSESSMENT SCORES (verified skills bonus)
    # ================================================================

    assessment_scores = signals["skill_assessment_scores"]
    if assessment_scores and JD_REQUIRED_SKILLS:
        matched_assessments = {
            skill: score_val
            for skill, score_val in assessment_scores.items()
            if any(req in skill or skill in req for req in JD_REQUIRED_SKILLS)
        }
        if matched_assessments:
            avg_score = sum(matched_assessments.values()) / len(matched_assessments)
            # Scale: avg_score in [0,1] → up to +10 pts
            assessment_bonus = min(avg_score * 10.0, 10.0)
            score += assessment_bonus
            reasons.append(
                f"Verified skill assessments for {len(matched_assessments)} JD-relevant skills "
                f"(avg={avg_score:.2f}): +{assessment_bonus:.1f}"
            )

    # ================================================================
    # PART 5: RELIABILITY + IDENTITY
    # ================================================================

    # --- Interview completion rate ---
    icr = signals["interview_completion_rate"]
    if icr is not None:
        if icr < 0.5:
            penalties += 5.0
            reasons.append(f"Low interview completion rate ({icr:.0%}): -5.0")
        elif icr >= 0.8:
            score += 3.0
            reasons.append(f"High interview completion rate ({icr:.0%}): +3.0")

    # --- Offer acceptance rate ---
    oar = signals["offer_acceptance_rate"]
    if oar is not None and oar > 0.7:
        score += 3.0
        reasons.append(f"High offer acceptance rate ({oar:.0%}): +3.0")

    # --- Profile completeness ---
    completeness = signals["profile_completeness"]
    if completeness > 80:
        completeness_score = (completeness - 80) * 0.1
        score += completeness_score
        reasons.append(f"Profile completeness {completeness:.1f}%: +{completeness_score:.1f}")

    # --- Verified identity ---
    verified = signals["verified_count"]
    if verified >= 3:
        score += 7.0
        reasons.append(f"Fully verified identity ({verified}/3): +7.0")
    elif verified >= 1:
        score += 3.0
        reasons.append(f"Partially verified identity ({verified}/3): +3.0")
    else:
        reasons.append("No verified identity signals")

    # ================================================================
    # Apply penalties and clamp
    # ================================================================
    final_score = score - penalties
    final_score = max(0.0, min(100.0, round(final_score, 2)))

    metadata = {
        "github_activity": github,
        "has_leadership_keywords": signals["has_leadership_keywords"],
        "connections": connections,
        "endorsements": endorsements,
        "verified_count": verified,
        "profile_completeness": completeness,
        "months_since_active": months_inactive,
        "open_to_work": signals["open_to_work"],
        "recruiter_response_rate": rr,
        "notice_period_days": notice,
        "work_mode": work_mode,
        "raw_penalty": penalties,
    }

    return ScoreResult(score=final_score, reasons=reasons, metadata=metadata)

"""Reason generator for the REDROB candidate ranking system.

Produces concise, human-readable, JD-specific explanations for a candidate's
final score using metadata produced by upstream scoring modules.

Fallback #8 fix: Rewritten to be JD-aware and profile-specific:
  - References actual JD title and matched skill names by name
  - Cites actual companies and YoE from the profile
  - Acknowledges specific gaps (e.g., "No vector DB experience")
  - Varies phrasing per candidate using metadata, not canned templates
  - References Redrob availability signals (last active, response rate)
  - References location and work-mode alignment
  - Rank-consistent tone (rank-1 ≠ rank-100 language)

Output: a :class:`Reason` object containing at most 5 reasons, each
truncated to 120 characters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.scoring import FinalScore, HoneypotResult, PenaltyResult, ScoreResult

if TYPE_CHECKING:
    from src.models.candidate import Candidate

# Hard limits per the spec.
MAX_REASONS: int = 5
MAX_REASON_LENGTH: int = 120

# JD context (loaded once)
from src.jd_config import JD

_TARGET_TITLE: str = JD.get("target_title") or "Senior AI Engineer"
_REQUIRED_SKILLS: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("required_skills") or [])
)
_PREFERRED_SKILLS: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("preferred_skills") or [])
)


@dataclass(slots=True)
class Reason:
    """A set of concise explanations for a candidate's ranking.

    Attributes:
        candidate_id: Identifier of the explained candidate.
        reasons: Ordered list of short reason strings (<=120 chars each).
    """

    candidate_id: str
    reasons: list[str] = field(default_factory=list)


def _truncate(text: str, limit: int = MAX_REASON_LENGTH) -> str:
    """Truncate ``text`` to ``limit`` characters, appending an ellipsis if cut."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


# Score tier labels — rank-consistent tone
def _score_tier(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 55:
        return "good"
    if score >= 35:
        return "moderate"
    return "weak"


# ---------------------------------------------------------------------------
# JD-aware reason builders
# ---------------------------------------------------------------------------

def _build_title_reason(
    title_result: ScoreResult | None,
    candidate: "Candidate | None",
) -> str | None:
    """Build a reason about title match — cites actual title and JD target."""
    if title_result is None:
        return None
    meta = title_result.metadata or {}
    title = meta.get("title", "")
    seniority = meta.get("seniority", "")
    is_ai = meta.get("is_ai_related", False)
    is_manager = meta.get("is_manager", False)
    jd_aware = meta.get("jd_aware", False)
    score = title_result.score

    if not title:
        return None

    if jd_aware:
        if score >= 65:
            return f"Title '{title}' aligns well with target '{_TARGET_TITLE}'."
        elif is_manager:
            return f"Title '{title}' is management/leadership — JD requires hands-on IC engineer."
        elif seniority in ("principal", "staff"):
            return f"Title '{title}' (over-senior): JD prefers Senior IC, not {seniority}-level."
        elif is_ai and score >= 40:
            return f"AI/ML relevant title '{title}' but partial match with '{_TARGET_TITLE}'."
        elif not is_ai:
            return f"Title '{title}' is not AI/ML relevant — JD requires '{_TARGET_TITLE}'."
    else:
        if is_ai and score >= 50:
            return f"AI/ML-relevant title: '{title}'."
        elif is_manager:
            return "Leadership/management title."

    return None


def _build_skill_reason(
    skill_result: ScoreResult | None,
    candidate: "Candidate | None",
) -> str | None:
    """Build a reason summarising matched JD skills by name."""
    if skill_result is None:
        return None
    meta = skill_result.metadata or {}
    matched_required = meta.get("matched_required_skills") or []
    matched_preferred = meta.get("matched_preferred_skills") or []
    required_count = meta.get("required_count", 0)
    unique_skills = meta.get("unique_skills", 0)
    anti_penalty = meta.get("anti_skill_penalty", 0)

    if anti_penalty > 0:
        return (
            f"Domain mismatch: primary skills are CV/speech/robotics, "
            f"lacking required NLP/retrieval stack."
        )

    if matched_required:
        # Show top matched skill names (up to 4)
        sample = ", ".join(sorted(matched_required)[:4])
        ratio = f"{len(matched_required)}/{required_count}" if required_count else str(len(matched_required))
        if len(matched_required) >= required_count * 0.7 and required_count > 0:
            return f"Matched {ratio} required skills incl. {sample}."
        elif len(matched_required) > 0:
            missing_count = required_count - len(matched_required) if required_count else 0
            gap_str = f" Missing {missing_count} required skills." if missing_count > 0 else ""
            return f"Matched {ratio} required skills ({sample}).{gap_str}"
    else:
        if required_count > 0:
            # Identify most important missing skills
            missing = sorted(_REQUIRED_SKILLS)[:3]
            missing_str = ", ".join(missing)
            return f"No required JD skills matched (needs: {missing_str}…)."

    if matched_preferred:
        pref_sample = ", ".join(sorted(matched_preferred)[:3])
        return f"Preferred skills present: {pref_sample} ({unique_skills} total skills)."

    return f"Skill breadth: {unique_skills} unique skills but limited JD overlap."


def _build_experience_reason(
    exp_result: ScoreResult | None,
    candidate: "Candidate | None",
) -> str | None:
    """Build a reason citing actual companies, YoE, and production signals."""
    if exp_result is None:
        return None
    meta = exp_result.metadata or {}
    yoe = meta.get("yoe", 0.0)
    progression = meta.get("progression", 0.0)
    it_fraction = meta.get("it_services_fraction", 0.0)
    top_tier = meta.get("top_tier_count", 0)
    production = meta.get("production_ml_signal", False)
    domain_rel = meta.get("domain_relevance_signal", False)
    companies = meta.get("companies", [])

    # Build company snippet
    company_str = ""
    if companies:
        # Show up to 2 company names
        top_companies = companies[:2]
        company_str = f" ({', '.join(top_companies)})" if top_companies else ""

    if top_tier > 0 and production:
        return f"{yoe:.0f} yrs incl. top-tier company{company_str} with production ML deployment."
    elif it_fraction >= 0.8 and yoe > 0:
        return f"{yoe:.0f} yrs but consulting-only career{company_str} — JD requires product companies."
    elif production and domain_rel:
        return (
            f"{yoe:.0f} yrs with production ML + ranking/retrieval domain work{company_str}."
        )
    elif production:
        return f"{yoe:.0f} yrs of production ML experience{company_str}."
    elif domain_rel:
        return f"{yoe:.0f} yrs with search/ranking/retrieval domain experience{company_str}."
    elif progression >= 0.5:
        return f"Strong upward progression over {yoe:.0f} yrs{company_str}."
    elif yoe > 0:
        return f"{yoe:.0f} yrs experience{company_str}."

    return None


def _build_availability_reason(
    behavior_result: ScoreResult | None,
    candidate: "Candidate | None",
) -> str | None:
    """Build a reason citing actual Redrob availability signals."""
    if behavior_result is None:
        return None
    meta = behavior_result.metadata or {}
    months_inactive = meta.get("months_since_active")
    open_to_work = meta.get("open_to_work", True)
    response_rate = meta.get("recruiter_response_rate")
    notice = meta.get("notice_period_days")
    work_mode = meta.get("work_mode", "")
    raw_penalty = meta.get("raw_penalty", 0.0)

    # Build availability string from actual signals
    issues = []
    positives = []

    if months_inactive is not None:
        if months_inactive > 6:
            issues.append(f"inactive {months_inactive:.0f}mo")
        elif months_inactive <= 1:
            positives.append("recently active")

    if not open_to_work:
        issues.append("not open to work")
    elif open_to_work:
        positives.append("open to work")

    if response_rate is not None:
        if response_rate < 0.3:
            issues.append(f"low response rate ({response_rate:.0%})")
        elif response_rate >= 0.7:
            positives.append(f"high response rate ({response_rate:.0%})")

    if notice is not None:
        if notice > 90:
            issues.append(f"{notice}d notice")
        elif notice <= 30:
            positives.append(f"short notice ({notice}d)")

    if issues:
        issues_str = ", ".join(issues)
        return f"Availability concerns: {issues_str}."

    if positives:
        pos_str = ", ".join(positives[:2])
        return f"Strong availability: {pos_str}."

    return None


def _build_location_reason(
    location_result: ScoreResult | None,
    candidate: "Candidate | None",
) -> str | None:
    """Build a reason about location alignment."""
    if location_result is None:
        return None
    meta = location_result.metadata or {}
    city = meta.get("city", "")
    country = meta.get("country", "")
    work_mode = meta.get("work_mode", "")
    raw_score = meta.get("raw_location_score", 0.0)

    if raw_score >= 15:
        loc_str = city if city else country
        return f"Location '{loc_str}' is a preferred city for this role."
    elif raw_score <= -10:
        loc_str = country or city or "unknown"
        return f"Location '{loc_str}' outside India — no visa sponsorship."
    elif raw_score >= 8:
        loc_str = city or country
        mode_str = f", {work_mode} mode" if work_mode else ""
        return f"India-based ({loc_str}{mode_str}) — acceptable location."

    return None


def _build_gap_reason(
    skill_result: ScoreResult | None,
    exp_result: ScoreResult | None,
    candidate: "Candidate | None",
) -> str | None:
    """Build a reason explicitly calling out the most critical gap."""
    skill_meta = (skill_result.metadata or {}) if skill_result else {}
    exp_meta = (exp_result.metadata or {}) if exp_result else {}

    matched_required = set(skill_meta.get("matched_required_skills") or [])
    required_count = skill_meta.get("required_count", 0)

    if required_count > 0 and len(matched_required) < required_count * 0.5:
        # Identify the most important unmatched skills
        unmatched = _REQUIRED_SKILLS - matched_required
        important_missing = sorted(unmatched)[:3]
        if important_missing:
            missing_str = ", ".join(important_missing)
            return f"Gaps: missing critical JD skills — {missing_str}."

    production = exp_meta.get("production_ml_signal", False)
    domain_rel = exp_meta.get("domain_relevance_signal", False)
    it_fraction = exp_meta.get("it_services_fraction", 0.0)

    if not production and not domain_rel:
        return "No production ML or retrieval/ranking experience evident."

    if it_fraction >= 0.6:
        return f"Majority of experience at IT services firms ({it_fraction:.0%}) — JD prefers product companies."

    return None


def _build_penalty_reason(penalty: PenaltyResult | None) -> str | None:
    """Build a reason describing the dominant JD-specific penalty (if any)."""
    if penalty is None or penalty.penalty_score <= 0:
        return None
    breakdown = (penalty.metadata or {}).get("breakdown", {})
    if not breakdown:
        return None

    # Prioritise JD-specific penalties
    jd_specific_order = [
        "consulting_only", "research_only", "domain_mismatch",
        "management_no_code", "framework_enthusiast"
    ]
    for key in jd_specific_order:
        val = breakdown.get(key, 0)
        if val > 0:
            label = key.replace("_", " ")
            return f"JD penalty: {label} ({val:.0f}pts)."

    # Generic penalties
    dominant = max(breakdown.items(), key=lambda kv: kv[1])
    name, value = dominant
    if value > 0:
        label = name.replace("_", " ")
        severity = "Major" if value >= 10.0 else "Minor"
        return f"{severity} penalty: {label}."

    return None


def _build_honeypot_reason(honeypot: HoneypotResult | None) -> str | None:
    """Build a reason noting honeypot suspicion (if any)."""
    if honeypot is None or honeypot.suspicion_score <= 0:
        return None
    meta = honeypot.metadata or {}
    is_hard_filtered = meta.get("is_hard_filtered", False)
    if is_hard_filtered:
        return f"⚠️ Profile flagged as suspicious (score {honeypot.suspicion_score:.0f}) — hard-filtered."
    if honeypot.suspicion_score >= 20.0:
        return f"Elevated profile suspicion ({honeypot.suspicion_score:.0f}/100) — verify profile."
    return None


def _build_confidence_reason(final: FinalScore) -> str | None:
    """Build a reason noting evidence confidence."""
    confidence = final.component_scores.get("confidence", 0.0)
    consistency = final.component_scores.get("consistency", 0.0)
    if confidence >= 0.7 and consistency >= 0.8:
        return "Well-corroborated, highly consistent profile."
    if confidence < 0.3:
        return "Limited cross-source evidence for declared skills."
    return None


# ---------------------------------------------------------------------------
# A typed bundle of optional component results
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _ComponentBundle:
    title: ScoreResult | None = None
    skills: ScoreResult | None = None
    experience: ScoreResult | None = None
    education: ScoreResult | None = None
    behavior: ScoreResult | None = None
    location: ScoreResult | None = None


def generate_reason(
    final: FinalScore,
    component_results: dict[str, ScoreResult] | None = None,
    candidate: "Candidate | None" = None,
) -> Reason:
    """Generate a concise, JD-aware :class:`Reason` for a single candidate.

    Fallback #8 fix: Now profile-specific. References actual titles, company
    names, matched skill names, availability signals, and JD gaps.

    Args:
        final: The composite ``FinalScore`` (carries penalty & honeypot).
        component_results: Optional mapping of component name -> its
            ``ScoreResult``.
        candidate: Optional candidate object for profile-specific reasoning.

    Returns:
        A :class:`Reason` with at most ``MAX_REASONS`` short strings.
    """
    bundle = _ComponentBundle()
    if component_results:
        bundle.title = component_results.get("title")
        bundle.skills = component_results.get("skills")
        bundle.experience = component_results.get("experience")
        bundle.education = component_results.get("education")
        bundle.behavior = component_results.get("behavior")
        bundle.location = component_results.get("location")

    reasons: list[str] = []

    # --- 1. Title reason (JD-aware) ---
    title_r = _build_title_reason(bundle.title, candidate)
    if title_r:
        reasons.append(title_r)

    # --- 2. Skills reason (JD-aware, names actual matched skills) ---
    skill_r = _build_skill_reason(bundle.skills, candidate)
    if skill_r:
        reasons.append(skill_r)

    # --- 3. Experience reason (cites companies + production signals) ---
    exp_r = _build_experience_reason(bundle.experience, candidate)
    if exp_r:
        reasons.append(exp_r)

    # --- 4. Availability reason (Redrob signals) or Location ---
    avail_r = _build_availability_reason(bundle.behavior, candidate)
    if avail_r:
        reasons.append(avail_r)
    else:
        loc_r = _build_location_reason(bundle.location, candidate)
        if loc_r:
            reasons.append(loc_r)

    # --- 5. Gaps / penalties / concerns (always last as a caveat) ---
    # First try a specific gap, then JD penalty, then honeypot
    gap_r = _build_gap_reason(bundle.skills, bundle.experience, candidate)
    pen_r = _build_penalty_reason(final.penalty_result)
    hp_r = _build_honeypot_reason(final.honeypot_result)

    # Fill remaining slots
    for r in [gap_r, pen_r, hp_r, _build_confidence_reason(final)]:
        if r and len(reasons) < MAX_REASONS:
            reasons.append(r)

    # If we still have < 2 reasons, add location or confidence
    if len(reasons) < 2:
        loc_r = _build_location_reason(bundle.location, candidate)
        if loc_r and loc_r not in reasons:
            reasons.append(loc_r)
        conf_r = _build_confidence_reason(final)
        if conf_r and conf_r not in reasons:
            reasons.append(conf_r)

    # Fallback: score tier summary if still empty
    if not reasons:
        tier = _score_tier(final.score)
        reasons.append(f"Overall {tier} match for '{_TARGET_TITLE}' (score={final.score:.1f}).")

    # Enforce limits: at most MAX_REASONS, each <= MAX_REASON_LENGTH.
    truncated = [_truncate(r) for r in reasons[:MAX_REASONS]]

    return Reason(candidate_id=final.candidate_id, reasons=truncated)


class ReasonGenerator:
    """Stateless wrapper around :func:`generate_reason`."""

    def generate(
        self,
        final: FinalScore,
        component_results: dict[str, ScoreResult] | None = None,
        candidate: "Candidate | None" = None,
    ) -> Reason:
        """Generate a reason. Delegates to :func:`generate_reason`."""
        return generate_reason(final, component_results, candidate=candidate)


__all__ = [
    "Reason",
    "ReasonGenerator",
    "generate_reason",
    "MAX_REASONS",
    "MAX_REASON_LENGTH",
]

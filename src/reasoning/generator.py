"""Reason generator for the REDROB candidate ranking system.

Produces concise, human-readable explanations for a candidate's final
score using **only** the metadata already produced by upstream scoring
modules. It never recomputes features and never inspects raw candidate
text.

Output: a :class:`Reason` object containing at most 5 reasons, each
truncated to 120 characters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.scoring import FinalScore, HoneypotResult, PenaltyResult, ScoreResult

# Hard limits per the spec.
MAX_REASONS: int = 5
MAX_REASON_LENGTH: int = 120


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
    """Truncate ``text`` to ``limit`` characters, appending an ellipsis if cut.

    Slicing is on characters (not bytes); UTF-8 safety is preserved because
    we never re-encode here. The ellipsis counts toward the limit.
    """
    if len(text) <= limit:
        return text
    # Reserve room for the ellipsis.
    return text[: max(0, limit - 1)] + "…"


# Thresholds for qualitative labels. Kept module-level (not configurable
# via YAML) because they describe presentation, not scoring.
_STRONG_THRESHOLD = 65.0
_SOLID_THRESHOLD = 45.0


def _strength_label(score: float, subject: str) -> str | None:
    """Return a qualitative reason for a component score, or None if too weak."""
    if score >= _STRONG_THRESHOLD:
        return f"Strong {subject} profile."
    if score >= _SOLID_THRESHOLD:
        return f"Solid {subject} profile."
    return None


def _build_skill_reason(skill_result: ScoreResult | None) -> str | None:
    """Build a reason summarizing matched skills with evidence backing."""
    if skill_result is None:
        return None
    meta = skill_result.metadata or {}
    match_count = meta.get("match_count", 0)
    if match_count <= 0:
        return None

    # Pull up to 2 matched skill names from the skill features if available.
    # The skill engine stores matched skills implicitly; we use the
    # ``synergy`` flag and match_count to describe evidence quality.
    synergy = meta.get("synergy", 0.0)
    if synergy > 0:
        return f"Evidence-backed expertise across {match_count} key skills with synergy."
    return f"Evidence-backed expertise across {match_count} key skills."


def _build_experience_reason(exp_result: ScoreResult | None) -> str | None:
    """Build a reason describing career progression / stability."""
    if exp_result is None:
        return None
    meta = exp_result.metadata or {}
    progression = meta.get("progression", 0.0)
    yoe = meta.get("yoe", 0.0)
    if progression >= 0.5:
        return f"Strong upward career progression over {yoe:.1f} years."
    if progression >= 0.2:
        return f"Stable career progression over {yoe:.1f} years."
    return None


def _build_education_reason(edu_result: ScoreResult | None) -> str | None:
    """Build a reason for education quality."""
    if edu_result is None:
        return None
    meta = edu_result.metadata or {}
    degree_bucket = meta.get("degree_bucket", "")
    tier = meta.get("tier", "")
    if degree_bucket in ("phd", "masters") and tier in ("tier_1", "tier_2"):
        return f"Strong academic background ({degree_bucket}, {tier})."
    return None


def _build_penalty_reason(penalty: PenaltyResult | None) -> str | None:
    """Build a reason describing the dominant penalty (if any)."""
    if penalty is None or penalty.penalty_score <= 0:
        return None
    breakdown = (penalty.metadata or {}).get("breakdown", {})
    # Find the largest contributing penalty.
    if breakdown:
        dominant = max(breakdown.items(), key=lambda kv: kv[1])
        name, value = dominant
        if value > 0:
            label = name.replace("_", " ")
            severity = "Major" if value >= 10.0 else "Minor"
            return f"{severity} penalty for {label}."
    return f"Minor penalty applied ({penalty.penalty_score:.1f} points)."


def _build_honeypot_reason(honeypot: HoneypotResult | None) -> str | None:
    """Build a reason noting honeypot suspicion (if any)."""
    if honeypot is None or honeypot.suspicion_score <= 0:
        return None
    if honeypot.suspicion_score >= 30.0:
        return f"Elevated suspicion signals detected ({honeypot.suspicion_score:.0f})."
    return None


def _build_confidence_reason(final: FinalScore) -> str | None:
    """Build a reason noting high or low evidence confidence."""
    confidence = final.component_scores.get("confidence", 0.0)
    consistency = final.component_scores.get("consistency", 0.0)
    if confidence >= 0.7 and consistency >= 0.8:
        return "Well-corroborated, highly consistent profile."
    if confidence < 0.3:
        return "Limited cross-source evidence for declared skills."
    return None


# A typed bundle of optional component results used for reason generation.
# This decouples the generator from how callers gather per-engine outputs.
@dataclass(slots=True)
class _ComponentBundle:
    title: ScoreResult | None = None
    skills: ScoreResult | None = None
    experience: ScoreResult | None = None
    education: ScoreResult | None = None
    behavior: ScoreResult | None = None


def generate_reason(
    final: FinalScore,
    component_results: dict[str, ScoreResult] | None = None,
) -> Reason:
    """Generate a concise :class:`Reason` for a single candidate.

    Args:
        final: The composite ``FinalScore`` (carries penalty & honeypot).
        component_results: Optional mapping of component name -> its
            ``ScoreResult``. When omitted, reasons are derived from the
            component score *values* in ``final.component_scores`` only.

    Returns:
        A :class:`Reason` with at most ``MAX_REASONS`` short strings.

    Notes:
        - Deterministic: identical inputs produce identical output.
        - Never inspects raw candidate text or recomputes features.
    """
    bundle = _ComponentBundle()
    if component_results:
        bundle.title = component_results.get("title")
        bundle.skills = component_results.get("skills")
        bundle.experience = component_results.get("experience")
        bundle.education = component_results.get("education")
        bundle.behavior = component_results.get("behavior")

    components = final.component_scores or {}
    reasons: list[str] = []

    # --- Title ---
    title_score = components.get("title", 0.0)
    title_label = _strength_label(title_score, "title match")
    if title_label:
        reasons.append(title_label)
    elif bundle.title:
        # Fallback: mention AI relevance or seniority if available.
        meta = bundle.title.metadata or {}
        if meta.get("is_ai_related"):
            reasons.append("AI/ML-relevant title.")
        elif meta.get("is_manager"):
            reasons.append("Leadership/management title.")

    # --- Skills (prefer rich metadata, fall back to qualitative label) ---
    skill_reason = _build_skill_reason(bundle.skills)
    if skill_reason:
        reasons.append(skill_reason)
    else:
        skill_label = _strength_label(components.get("skills", 0.0), "skill")
        if skill_label:
            reasons.append(skill_label)

    # --- Experience ---
    exp_reason = _build_experience_reason(bundle.experience)
    if exp_reason:
        reasons.append(exp_reason)
    else:
        exp_label = _strength_label(components.get("experience", 0.0), "experience")
        if exp_label:
            reasons.append(exp_label)

    # --- Education ---
    edu_reason = _build_education_reason(bundle.education)
    if edu_reason:
        reasons.append(edu_reason)

    # --- Confidence / consistency note ---
    conf_reason = _build_confidence_reason(final)
    if conf_reason:
        reasons.append(conf_reason)

    # --- Penalty (always last so it reads as a caveat) ---
    pen_reason = _build_penalty_reason(final.penalty_result)
    if pen_reason:
        reasons.append(pen_reason)

    # --- Honeypot (caveat) ---
    hp_reason = _build_honeypot_reason(final.honeypot_result)
    if hp_reason:
        reasons.append(hp_reason)

    # Enforce limits: at most MAX_REASONS, each <= MAX_REASON_LENGTH.
    truncated = [_truncate(r) for r in reasons[:MAX_REASONS]]

    return Reason(candidate_id=final.candidate_id, reasons=truncated)


class ReasonGenerator:
    """Stateless wrapper around :func:`generate_reason`."""

    def generate(
        self,
        final: FinalScore,
        component_results: dict[str, ScoreResult] | None = None,
    ) -> Reason:
        """Generate a reason. Delegates to :func:`generate_reason`."""
        return generate_reason(final, component_results)


__all__ = [
    "Reason",
    "ReasonGenerator",
    "generate_reason",
    "MAX_REASONS",
    "MAX_REASON_LENGTH",
]

"""Skill scoring engine for the REDROB candidate ranking system.

Evaluates a candidate's skill set against required and preferred skills,
with JD-specific must-have vs nice-to-have differentiation, anti-skill
detection, proficiency-weighted matching, and JD-relevant synergy groups.

Fallback #5 fix:
  - required_skills / preferred_skills now driven by JD config (called from main.py)
  - Anti-skills: CV/speech/robotics without NLP/retrieval → penalty
  - Synergy groups redefined for retrieval/ranking/LLM stacks
  - Proficiency weighting: beginner on a required skill = 50% credit
  - Skill assessment scores from Redrob → verified bonus

This module is purely responsible for skill scoring. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import ScoreResult
from src.jd_config import JD_ANTI_SKILLS, JD_REQUIRED_SKILLS


# JD-relevant complementary skill groups (Fallback #5 fix).
# If a candidate has 2+ skills from the same group, they earn a synergy bonus.
_SYNERGY_GROUPS: list[list[str]] = [
    # Core retrieval stack (most important for this JD)
    ["embeddings", "vector db", "vector database", "pinecone", "weaviate", "faiss",
     "chroma", "qdrant", "milvus", "retrieval", "semantic search", "dense retrieval",
     "sparse retrieval", "information retrieval", "rag", "retrieval augmented generation"],
    # Applied ML stack
    ["python", "pytorch", "tensorflow", "machine learning", "deep learning", "ml",
     "nlp", "transformers", "hugging face", "scikit-learn"],
    # LLM / fine-tuning stack
    ["llm", "fine-tuning", "lora", "rlhf", "instruction tuning", "prompt engineering",
     "openai", "gpt", "llama", "mistral"],
    # Evaluation & ranking stack
    ["ndcg", "mrr", "map", "evaluation", "ranking", "learning to rank", "a/b testing",
     "experimentation", "offline evaluation", "online evaluation"],
    # MLOps / infrastructure
    ["mlflow", "kubeflow", "airflow", "docker", "kubernetes", "aws", "gcp", "azure",
     "feast", "spark", "distributed systems"],
]

# Primary anti-skill categories (CV/speech/robotics without NLP presence)
_CV_SKILLS: frozenset[str] = frozenset([
    "computer vision", "opencv", "object detection", "image segmentation",
    "image classification", "yolo", "cnn", "convolutional", "face recognition",
    "pose estimation",
])
_SPEECH_SKILLS: frozenset[str] = frozenset([
    "speech recognition", "asr", "tts", "text to speech", "speech synthesis",
    "speech processing", "kaldi", "wav2vec",
])
_ROBOTICS_SKILLS: frozenset[str] = frozenset([
    "robotics", "ros", "slam", "autonomous vehicles", "self-driving",
    "motion planning", "kinematics",
])

# NLP/retrieval anchor skills — split into two tiers (Fallback #19 fix).
#
# STRONG anchors: domain-specific — fully mitigate anti-skill penalty.
# Only these prove the candidate genuinely bridges into NLP/retrieval.
_STRONG_NLP_ANCHORS: frozenset[str] = frozenset([
    "nlp", "natural language processing",
    "embeddings", "vector db", "vector database",
    "retrieval", "information retrieval", "dense retrieval", "sparse retrieval",
    "semantic search", "rag", "retrieval augmented generation",
    "ranking", "learning to rank",
    "transformers",
    "llm",
])

# WEAK anchors: generic ML terms that any ML engineer might list.
# Their presence only *reduces* the anti-skill penalty (60% off), never eliminates it.
# A pure computer vision PhD who also lists "python" and "machine learning"
# is still a domain mismatch for this JD.
_WEAK_NLP_ANCHORS: frozenset[str] = frozenset([
    "machine learning", "ml", "deep learning",
    "python", "pytorch", "tensorflow", "scikit-learn",
    "data science", "artificial intelligence", "ai",
])

# Combined for backwards-compat references (not used in _check_anti_skills).
_NLP_ANCHOR_SKILLS: frozenset[str] = _STRONG_NLP_ANCHORS | _WEAK_NLP_ANCHORS

# Proficiency tiers — used to weight required-skill credit
_PROFICIENCY_WEIGHTS: dict[str, float] = {
    "expert": 1.0,
    "advanced": 0.85,
    "intermediate": 0.7,
    "beginner": 0.5,
    "": 0.8,  # no proficiency declared → assume intermediate-ish
}


def _count_synergy(candidate_skills: set[str]) -> tuple[float, list[str]]:
    """Count JD-relevant synergy bonuses from complementary skill groups.

    For each group, if a candidate has 2+ skills from that group, award a bonus.
    The first group (retrieval stack) gives double weight as it is the core JD need.

    Returns:
        Tuple of (total bonus points, list of reason strings).
    """
    total_bonus = 0.0
    synergy_reasons: list[str] = []
    seen_skills: set[str] = set()

    for group_idx, group in enumerate(_SYNERGY_GROUPS):
        matched = set()
        for skill in candidate_skills:
            if skill in group and skill not in seen_skills:
                matched.add(skill)

        if len(matched) >= 2:
            # Retrieval stack (index 0) is worth double synergy
            multiplier = 2.0 if group_idx == 0 else 1.0
            bonus = min(len(matched) * 3.0 * multiplier, 18.0 if group_idx == 0 else 12.0)
            total_bonus += bonus
            synergy_reasons.append(
                f"Synergy bonus for {sorted(matched)}: +{bonus:.1f}"
            )
            seen_skills.update(matched)

    return total_bonus, synergy_reasons


def _check_anti_skills(
    candidate_skills: set[str],
    skill_features: dict,
) -> tuple[float, str]:
    """Detect domain mismatch — CV/speech/robotics without NLP/retrieval presence.

    Fallback #19 fix: Tiered anchor mitigation.
      - STRONG anchors (retrieval, embeddings, vector db, rag, etc.):
        fully mitigate the penalty — candidate genuinely bridges domains.
      - WEAK anchors (machine learning, ml, python, etc.):
        reduce the penalty by 60% but do not eliminate it — a pure CV
        specialist who lists 'machine learning' is still a domain mismatch.
      - No anchors: full penalty applies.

    Returns:
        Tuple of (penalty points, reason string).
    """
    # --- Tier 1: strong domain-specific anchors → full mitigation ---
    has_strong_anchor = bool(candidate_skills & _STRONG_NLP_ANCHORS)
    if has_strong_anchor:
        return 0.0, ""  # Candidate genuinely bridges into NLP/retrieval

    # Count anti-skill hits
    cv_hits = candidate_skills & _CV_SKILLS
    speech_hits = candidate_skills & _SPEECH_SKILLS
    robotics_hits = candidate_skills & _ROBOTICS_SKILLS
    total_anti = len(cv_hits) + len(speech_hits) + len(robotics_hits)

    if total_anti == 0:
        return 0.0, ""

    # Fraction of skills that are anti-skills
    total_skills = len(candidate_skills) if candidate_skills else 1
    anti_fraction = total_anti / total_skills

    # --- Tier 2: weak generic anchors → partial mitigation (60% off) ---
    has_weak_anchor = bool(candidate_skills & _WEAK_NLP_ANCHORS)
    weak_reduction = 0.6 if has_weak_anchor else 0.0  # fraction to reduce penalty by

    if anti_fraction >= 0.4:
        base_penalty = 20.0
        penalty = base_penalty * (1.0 - weak_reduction)
        anchor_note = f", partial mitigation (weak anchor, -{weak_reduction:.0%})" if has_weak_anchor else ""
        return penalty, (
            f"Primary expertise is CV/speech/robotics without NLP/retrieval anchor "
            f"({total_anti} anti-skills, {anti_fraction:.0%} of profile"
            f"{anchor_note}): -{penalty:.0f}"
        )
    elif anti_fraction >= 0.2 or total_anti >= 3:
        base_penalty = 10.0
        penalty = base_penalty * (1.0 - weak_reduction)
        anchor_note = f", partial mitigation (weak anchor, -{weak_reduction:.0%})" if has_weak_anchor else ""
        return penalty, (
            f"Significant CV/speech/robotics focus without NLP anchor "
            f"({total_anti} anti-skills{anchor_note}): -{penalty:.0f}"
        )

    return 0.0, ""


def _get_proficiency_weight(skill_name_lower: str, skill_features: dict) -> float:
    """Look up proficiency weight for a specific skill.

    Uses multi_source_skills data from SkillsExtractor to find proficiency level.
    Defaults to 0.8 (intermediate-ish) if not found.
    """
    multi_source = skill_features.get("multi_source_skills") or []
    for s in multi_source:
        if (s.get("name") or "").lower() == skill_name_lower:
            prof = (s.get("proficiency") or "").lower()
            return _PROFICIENCY_WEIGHTS.get(prof, _PROFICIENCY_WEIGHTS[""])
    return _PROFICIENCY_WEIGHTS[""]  # default


def score_skills(
    candidate: Candidate,
    skill_features: dict,
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
) -> ScoreResult:
    """Score a candidate's skills with JD-aware hierarchy.

    Fallback #5 fix: required/preferred skills are now passed from main.py
    (loaded from JD config). Anti-skills, proficiency weighting, synergy
    rebalancing, and Redrob assessment score bonuses are all included.

    Args:
        candidate: The candidate to score.
        skill_features: Output from ``SkillsExtractor.extract()``.
        required_skills: Skills mandatory for the JD role (from jd_config).
        preferred_skills: Nice-to-have skills for the JD role.

    Returns:
        A ``ScoreResult`` with score in [0, 100].
    """
    reasons: list[str] = []
    score = 0.0
    candidate_skills = set(skill_features.get("skills", []))
    unique_count = skill_features.get("unique_skills", 0)
    endorsement_total = skill_features.get("endorsement_total", 0)

    # --- Base score for having skills ---
    if unique_count == 0:
        reasons.append("No skills found")
        return ScoreResult(
            score=0.0,
            reasons=reasons,
            metadata={"match_count": 0, "required_count": 0, "preferred_count": 0, "synergy": 0.0},
        )

    # Volume score: diminishing returns after 10 skills
    volume_score = min(unique_count * 1.5, 15.0)
    score += volume_score
    reasons.append(f"Skill volume ({unique_count} unique): +{volume_score:.1f}")

    # --- Required skill matching (40-point lever) ---
    required_score = 0.0
    matched_required: set[str] = set()
    if required_skills:
        required_lower = {s.lower() for s in required_skills}
        for req_skill in required_lower:
            if req_skill in candidate_skills:
                # Weight by proficiency
                prof_weight = _get_proficiency_weight(req_skill, skill_features)
                required_score += prof_weight
                matched_required.add(req_skill)

        # Normalise to 40 points
        max_possible = len(required_lower)
        required_ratio = required_score / max_possible if max_possible > 0 else 0.0
        required_pts = required_ratio * 40.0
        score += required_pts
        reasons.append(
            f"Required skills: {len(matched_required)}/{len(required_lower)} matched "
            f"(proficiency-weighted): +{required_pts:.1f}"
        )
    else:
        required_lower = set()

    # --- Preferred skill matching (15-point lever) ---
    matched_preferred: set[str] = set()
    if preferred_skills:
        preferred_lower = {s.lower() for s in preferred_skills}
        matched_preferred = candidate_skills & preferred_lower
        preferred_ratio = len(matched_preferred) / len(preferred_lower) if preferred_lower else 0.0
        preferred_score = preferred_ratio * 15.0
        score += preferred_score
        reasons.append(
            f"Preferred skills: {len(matched_preferred)}/{len(preferred_lower)} matched: +{preferred_score:.1f}"
        )
    else:
        matched_preferred = set()

    # --- Anti-skill domain mismatch penalty (Fallback #5) ---
    anti_penalty, anti_reason = _check_anti_skills(candidate_skills, skill_features)
    if anti_penalty > 0:
        score -= anti_penalty
        reasons.append(anti_reason)

    # --- Endorsement bonus ---
    if endorsement_total > 0:
        endorsement_bonus = min(endorsement_total * 0.1, 8.0)
        score += endorsement_bonus
        reasons.append(f"Endorsements ({endorsement_total} total): +{endorsement_bonus:.1f}")

    # --- JD-relevant synergy bonuses ---
    synergy_bonus, synergy_reasons = _count_synergy(candidate_skills)
    if synergy_bonus > 0:
        score += synergy_bonus
        reasons.extend(synergy_reasons)

    # --- Career-backing quality bonus ---
    # Skills found in career_history descriptions carry stronger evidence.
    multi_source = skill_features.get("multi_source_skills") or []
    career_ratio = 0.0
    if multi_source:
        career_backed = sum(
            1 for s in multi_source
            if "career_history" in (s.get("sources") or [])
        )
        career_ratio = career_backed / len(multi_source)
        if career_ratio >= 0.5:
            context_bonus = round(career_ratio * 10.0, 2)
            score += context_bonus
            reasons.append(
                f"Career-backed skills ({career_backed}/{len(multi_source)}, "
                f"{career_ratio:.0%}): +{context_bonus:.1f}"
            )
        elif career_ratio > 0:
            context_bonus = round(career_ratio * 5.0, 2)
            score += context_bonus
            reasons.append(
                f"Partially career-backed skills ({career_backed}/{len(multi_source)}): "
                f"+{context_bonus:.1f}"
            )
        else:
            reasons.append("Skills not backed by career history (self-reported only)")

    # --- Redrob skill assessment scores (verified bonus) ---
    redrob = candidate.redrob_signals
    assessment_bonus = 0.0
    if redrob:
        assessment_scores = getattr(redrob, "skill_assessment_scores", None) or {}
        if assessment_scores and required_skills:
            req_set = {s.lower() for s in required_skills}
            matched_assessments = {}
            for skill_name, score_val in assessment_scores.items():
                s_lower = skill_name.lower()
                if any(req in s_lower or s_lower in req for req in req_set):
                    matched_assessments[s_lower] = score_val
            if matched_assessments:
                avg_val = sum(matched_assessments.values()) / len(matched_assessments)
                assessment_bonus = min(avg_val * 10.0, 10.0)
                score += assessment_bonus
                reasons.append(
                    f"Redrob-verified JD skill assessments ({len(matched_assessments)} skills, "
                    f"avg={avg_val:.2f}): +{assessment_bonus:.1f}"
                )

    # Clamp to [0, 100].
    score = max(0.0, min(100.0, round(score, 2)))

    metadata = {
        "match_count": len(matched_required),
        "required_count": len(required_skills or []),
        "preferred_count": len(matched_preferred),
        "synergy": round(synergy_bonus, 2),
        "unique_skills": unique_count,
        "career_backed_ratio": round(career_ratio if multi_source else 0.0, 2),
        "anti_skill_penalty": anti_penalty,
        "matched_required_skills": sorted(matched_required),
        "matched_preferred_skills": sorted(matched_preferred),
        "assessment_bonus": assessment_bonus,
    }

    return ScoreResult(score=score, reasons=reasons, metadata=metadata)

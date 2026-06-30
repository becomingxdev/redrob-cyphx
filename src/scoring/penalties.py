"""Penalty engine for the REDROB candidate ranking system.

Applies soft, non-rejecting penalties for profile quality issues AND
JD-specific disqualifier patterns.

Fallback #3 fix: Adds JD-specific negative filters:
  - Consulting-only career (TCS, Infosys, etc.)
  - Research-only profile (no production deployment)
  - Domain mismatch (CV/speech/robotics without NLP/retrieval)
  - Management/architecture-only without recent IC coding
  - Framework-enthusiast-only (LangChain with no underlying skills)

Penalties reduce the final score but never eliminate a candidate.

This module is purely responsible for penalty assessment. It never computes
rankings or accesses other scoring components.
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import PenaltyResult
from src.jd_config import JD, JD_IT_SERVICES, JD_REQUIRED_SKILLS


# ============================================================
# Original generic penalty checks
# ============================================================

def _check_inflated_title(
    title_features: dict,
    career_features: dict,
    experience_features: dict,
) -> tuple[float, str]:
    """Detect potentially inflated titles relative to career history."""
    seniority = title_features.get("seniority", "mid")
    yoe = experience_features.get("years_of_experience", 0.0)
    total_positions = experience_features.get("total_positions", 0)
    has_promotion = career_features.get("promotion_indicators", False)

    if seniority in ("senior", "lead", "principal", "staff") and yoe < 2.0:
        penalty = min((2.0 - yoe) * 5.0, 10.0)
        return penalty, (
            f"Title '{seniority}' with only {yoe:.1f} years of experience"
        )

    if seniority in ("principal", "staff") and not has_promotion and total_positions <= 2:
        penalty = 8.0
        return penalty, (
            f"Title '{seniority}' with no promotion history and only "
            f"{total_positions} positions"
        )

    return 0.0, ""


def _check_weak_evidence(evidence_result: dict) -> tuple[float, str]:
    """Detect candidates with weak overall evidence support."""
    if not evidence_result:
        return 5.0, "No evidence available"

    weak_skills = 0
    total_skills = len(evidence_result)
    for skill_evidence in evidence_result.values():
        support = skill_evidence.get("source_support_count", 0)
        if support <= 1:
            weak_skills += 1

    if total_skills > 0:
        weak_ratio = weak_skills / total_skills
        if weak_ratio > 0.8:
            penalty = min(weak_ratio * 8.0, 10.0)
            return penalty, (
                f"Weak evidence for {weak_skills}/{total_skills} skills "
                f"({weak_ratio:.0%} with single-source support)"
            )

    return 0.0, ""


def _check_inconsistent_profile(consistency_result: dict) -> tuple[float, str]:
    """Detect profile inconsistency based on consistency analysis.

    FIX 5: Max penalty reduced from 10 → 5 points.
    """
    consistency_score = consistency_result.get("consistency_score", 1.0)
    conflicts = consistency_result.get("conflicts", [])

    if consistency_score < 0.5:
        penalty = min((0.5 - consistency_score) * 10.0, 5.0)
        return penalty, (
            f"Low consistency score ({consistency_score:.2f}): "
            f"{len(conflicts)} conflict(s)"
        )

    return 0.0, ""


def _check_missing_information(
    candidate: Candidate,
    experience_features: dict,
    education_features: dict,
) -> tuple[float, str]:
    """Detect missing or incomplete profile information."""
    missing: list[str] = []
    penalty = 0.0

    if not candidate.profile or not candidate.profile.summary:
        missing.append("summary")
        penalty += 2.0

    if not candidate.career_history or len(candidate.career_history) == 0:
        missing.append("career_history")
        penalty += 3.0

    if not candidate.skills or len(candidate.skills) == 0:
        missing.append("skills")
        penalty += 3.0

    if education_features.get("highest_degree", "") == "" and education_features.get("education_count", 0) == 0:
        missing.append("education")
        penalty += 1.0

    if not candidate.redrob_signals or not getattr(candidate.redrob_signals, "verified_email", False):
        missing.append("unverified_email")
        penalty += 1.0

    if missing:
        penalty = min(penalty, 10.0)
        return penalty, f"Missing profile sections: {', '.join(missing)}"

    return 0.0, ""


# ============================================================
# JD-specific negative filters (Fallback #3)
# ============================================================

def _check_consulting_only_career(career_features: dict) -> tuple[float, str]:
    """Detect consulting-only careers at IT services firms.

    JD disqualifier: 'Consulting-only career (TCS, Wipro, Infosys...)'.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    it_fraction = career_features.get("it_services_fraction", 0.0)
    it_count = career_features.get("it_services_count", 0)
    consulting_threshold = JD.get("consulting_only_fraction", 0.8)
    companies = career_features.get("companies_worked_for", [])
    total_companies = len(companies) if companies else 1

    if it_fraction >= consulting_threshold and it_count >= 1 and total_companies >= 1:
        penalty = 25.0
        it_companies = [c for c in companies if any(
            it_name in c.lower() for it_name in JD_IT_SERVICES
        )]
        company_str = ", ".join(it_companies[:3]) if it_companies else "IT services firms"
        return penalty, (
            f"Consulting-only career ({it_fraction:.0%} IT services: {company_str}): -{penalty:.0f} "
            f"(JD: 'product companies only, not pure services')"
        )

    return 0.0, ""


def _check_research_only_profile(
    candidate: Candidate,
    career_features: dict,
    title_features: dict,
) -> tuple[float, str]:
    """Detect pure research profiles with no production deployment.

    JD disqualifier: 'Pure research only, no production deployment'.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    # Research-only signal from career description analysis
    research_only = career_features.get("research_only_signal", False)
    production_signals = career_features.get("total_production_signals", 0)

    # Also check title and summary for research keywords
    title = (title_features.get("title") or "").lower()
    profile = candidate.profile
    summary = (profile.summary or "").lower() if profile else ""

    research_title_keywords = ["research scientist", "researcher", "research engineer",
                                "research intern", "postdoc", "phd candidate"]
    has_research_title = any(kw in title for kw in research_title_keywords)

    if research_only or (has_research_title and production_signals == 0):
        penalty = 15.0
        return penalty, (
            f"Research-only profile detected (no production deployment evidence): -{penalty:.0f} "
            f"(JD: 'pure research candidates will not move forward')"
        )

    return 0.0, ""


def _check_domain_mismatch(skill_features: dict) -> tuple[float, str]:
    """Detect domain mismatch — CV/speech/robotics primary expertise without NLP.

    JD disqualifier: 'Primary expertise is CV/speech/robotics without NLP/retrieval'.

    Fallback #19 fix: Tiered anchor mitigation. A candidate who only lists
    generic ML skills ('machine learning', 'python') but has a CV-heavy profile
    is still a domain mismatch for this retrieval-focused JD. Only domain-specific
    anchors (retrieval, vector db, embeddings, rag, etc.) fully clear the penalty.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    candidate_skills = set(skill_features.get("skills", []))
    if not candidate_skills:
        return 0.0, ""

    # CV/speech/robotics anti-skills
    cv_keywords = {"computer vision", "opencv", "object detection", "image segmentation",
                   "image classification", "yolo", "cnn", "convolutional"}
    speech_keywords = {"speech recognition", "asr", "tts", "text to speech",
                       "speech synthesis", "speech processing"}
    robotics_keywords = {"robotics", "ros", "slam", "autonomous vehicles"}

    anti_hits = len(
        (candidate_skills & cv_keywords) |
        (candidate_skills & speech_keywords) |
        (candidate_skills & robotics_keywords)
    )

    # --- Tier 1: strong domain-specific anchors → full mitigation ---
    # Only retrieval/NLP-specific terms prove real domain overlap.
    strong_anchors = {
        "nlp", "embeddings", "retrieval", "information retrieval",
        "vector db", "vector database", "semantic search", "transformers",
        "llm", "rag", "retrieval augmented generation", "ranking",
        "dense retrieval", "sparse retrieval", "learning to rank",
    }
    has_strong_anchor = bool(candidate_skills & strong_anchors)
    if has_strong_anchor:
        return 0.0, ""  # Genuinely bridges domains — no penalty

    # --- Tier 2: weak generic anchors → partial mitigation (60% off) ---
    weak_anchors = {
        "machine learning", "ml", "deep learning", "python",
        "pytorch", "tensorflow", "data science", "artificial intelligence", "ai",
    }
    has_weak_anchor = bool(candidate_skills & weak_anchors)
    weak_reduction = 0.6 if has_weak_anchor else 0.0

    mismatch_threshold = JD.get("domain_mismatch_fraction", 0.5)
    total_skills = len(candidate_skills)
    anti_fraction = anti_hits / total_skills if total_skills > 0 else 0.0

    if anti_fraction >= mismatch_threshold:
        base_penalty = 15.0
        penalty = base_penalty * (1.0 - weak_reduction)
        anchor_note = f", partial mitigation (generic ML anchor, -{weak_reduction:.0%})" if has_weak_anchor else ""
        return penalty, (
            f"Primary domain is CV/speech/robotics ({anti_hits} anti-skills, "
            f"{anti_fraction:.0%} of profile) without NLP/retrieval anchor"
            f"{anchor_note}: -{penalty:.0f}"
        )
    elif anti_fraction >= 0.3 or anti_hits >= 3:
        base_penalty = 8.0
        penalty = base_penalty * (1.0 - weak_reduction)
        anchor_note = f", partial mitigation (generic ML anchor, -{weak_reduction:.0%})" if has_weak_anchor else ""
        return penalty, (
            f"Significant CV/speech/robotics focus without NLP presence "
            f"({anti_hits} anti-skills{anchor_note}): -{penalty:.0f}"
        )

    return 0.0, ""



def _check_management_no_code(
    title_features: dict,
    career_features: dict,
    experience_features: dict,
) -> tuple[float, str]:
    """Detect management/architecture roles with no recent IC coding signals.

    JD disqualifier: 'People who haven't written production code in the last
    18 months because they've moved into architecture or tech lead roles.'

    Returns:
        Tuple of (penalty amount, reason string).
    """
    seniority = title_features.get("seniority", "mid")
    is_manager = title_features.get("is_manager", False)
    title = (title_features.get("title") or "").lower()
    yoe = experience_features.get("years_of_experience", 0.0)

    # Only penalise clearly management/architecture roles
    management_terms = ["manager", "director", "vp", "chief", "head of", "cto"]
    architect_terms = ["architect", "solution architect"]
    is_management = is_manager or any(t in title for t in management_terms)
    is_architect = any(t in title for t in architect_terms) and "engineer" not in title

    if not (is_management or is_architect or seniority in ("principal",)):
        return 0.0, ""

    # Check if career description shows recent production coding
    production_ml = career_features.get("production_ml_signal", False)
    prod_signals = career_features.get("total_production_signals", 0)

    if not production_ml and prod_signals == 0:
        penalty = 10.0
        role_type = "management" if is_management else ("architect" if is_architect else "principal")
        return penalty, (
            f"{role_type.capitalize()} role ('{title_features.get('title')}') with "
            f"no production coding signals: -{penalty:.0f} "
            f"(JD wants IC engineers, not pure {role_type})"
        )

    return 0.0, ""


def _check_framework_enthusiast_only(skill_features: dict) -> tuple[float, str]:
    """Detect LangChain/wrapper-only profiles without underlying ML skills.

    JD disqualifier: 'Framework enthusiasts (LangChain tutorials only)'.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    candidate_skills = set(skill_features.get("skills", []))
    if not candidate_skills:
        return 0.0, ""

    framework_only_skills = set(JD.get("framework_only_skills") or [])
    framework_hits = candidate_skills & framework_only_skills

    if not framework_hits:
        return 0.0, ""

    # If candidate has underlying required skills, no penalty
    has_underlying = bool(candidate_skills & JD_REQUIRED_SKILLS)
    if has_underlying:
        return 0.0, ""

    # Framework-only without underlying skills
    penalty = 10.0
    return penalty, (
        f"Framework-enthusiast pattern: {sorted(framework_hits)} without "
        f"underlying embeddings/retrieval/ML skills: -{penalty:.0f}"
    )


def _check_salary_misalignment(candidate: Candidate) -> tuple[float, str]:
    """Detect wildly misaligned salary expectations.

    Fallback #14: Candidates expecting far above Series A band → mild penalty.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    redrob = candidate.redrob_signals
    if not redrob:
        return 0.0, ""

    salary_range = getattr(redrob, "expected_salary_range_inr_lpa", None)
    if salary_range is None:
        return 0.0, ""

    # Get the minimum of the expected range
    salary_min = getattr(salary_range, "min", None)
    salary_max = getattr(salary_range, "max", None)

    extreme_threshold = JD.get("salary_extreme_threshold_lpa", 80)
    jd_max = JD.get("salary_band_max_lpa", 50)

    if salary_min is not None and salary_min > extreme_threshold:
        penalty = 8.0
        return penalty, (
            f"Salary expectation (min ₹{salary_min}L) far exceeds "
            f"Series A band (₹{jd_max}L max): -{penalty:.0f}"
        )
    elif salary_min is not None and salary_min > jd_max * 1.5:
        penalty = 4.0
        return penalty, (
            f"Salary expectation (min ₹{salary_min}L) above typical band "
            f"(₹{jd_max}L max): -{penalty:.0f}"
        )

    return 0.0, ""


def _check_industry_mismatch(career_features: dict) -> tuple[float, str]:
    """Detect careers spent entirely in non-tech industries.

    Fallback #15: Manufacturing/retail/FMCG careers should be penalised.

    Returns:
        Tuple of (penalty amount, reason string).
    """
    industries = [i.lower() for i in (career_features.get("industries_worked_in") or [])]
    if not industries:
        return 0.0, ""

    non_tech_industries = [
        "manufacturing", "paper", "packaging", "retail", "consumer goods",
        "fmcg", "automotive", "construction", "real estate", "hospitality",
        "textile", "agriculture", "mining", "oil", "gas", "pharmaceutical",
        "food", "beverage",
    ]
    tech_industries = [
        "technology", "software", "information technology", "internet",
        "saas", "fintech", "edtech", "healthtech", "ai", "artificial intelligence",
        "e-commerce", "media", "telecom",
    ]

    non_tech_count = sum(1 for ind in industries if any(nt in ind for nt in non_tech_industries))
    tech_count = sum(1 for ind in industries if any(t in ind for t in tech_industries))

    # If majority of career is in non-tech industries with no tech presence
    if non_tech_count > 0 and tech_count == 0 and non_tech_count >= len(industries) * 0.7:
        penalty = 8.0
        return penalty, (
            f"Career primarily in non-tech industries ({non_tech_count}/{len(industries)}): -{penalty:.0f}"
        )

    return 0.0, ""


# ============================================================
# Main penalty orchestrator
# ============================================================

def apply_penalties(
    candidate: Candidate,
    title_features: dict,
    experience_features: dict,
    education_features: dict,
    career_features: dict,
    evidence_result: dict,
    consistency_result: dict,
    skill_features: dict | None = None,
) -> PenaltyResult:
    """Apply soft penalties to a candidate's profile.

    Fallback #3 fix: Added JD-specific negative filters for consulting-only
    careers, research-only profiles, domain mismatch, management roles without
    coding, and framework-enthusiast patterns.

    Penalties are additive and non-rejecting. They reduce the final score
    via a multiplier in the composite engine, but never eliminate candidates.

    Args:
        candidate: The candidate to assess.
        title_features: Output from ``TitleExtractor.extract()``.
        experience_features: Output from ``ExperienceExtractor.extract()``.
        education_features: Output from ``EducationExtractor.extract()``.
        career_features: Output from ``CareerExtractor.extract()``.
        evidence_result: Output from ``EvidenceVerifier.verify()``.
        consistency_result: Output from ``ConsistencyAnalyzer.analyze()``.
        skill_features: Output from ``SkillsExtractor.extract()`` (optional).

    Returns:
        A ``PenaltyResult`` with the total penalty score and breakdown.
    """
    total_penalty = 0.0
    reasons: list[str] = []
    breakdown: dict[str, float] = {}

    # --- Generic checks ---
    generic_checks = [
        ("inflated_title", _check_inflated_title(title_features, career_features, experience_features)),
        ("weak_evidence", _check_weak_evidence(evidence_result)),
        ("inconsistent_profile", _check_inconsistent_profile(consistency_result)),
        ("missing_information", _check_missing_information(candidate, experience_features, education_features)),
        ("industry_mismatch", _check_industry_mismatch(career_features)),
    ]

    # --- JD-specific checks (Fallback #3) ---
    jd_checks = [
        ("consulting_only", _check_consulting_only_career(career_features)),
        ("research_only", _check_research_only_profile(candidate, career_features, title_features)),
        ("management_no_code", _check_management_no_code(title_features, career_features, experience_features)),
        ("salary_misalignment", _check_salary_misalignment(candidate)),
    ]
    if skill_features is not None:
        jd_checks.append(("domain_mismatch", _check_domain_mismatch(skill_features)))
        jd_checks.append(("framework_enthusiast", _check_framework_enthusiast_only(skill_features)))

    all_checks = generic_checks + jd_checks

    for name, (penalty, reason) in all_checks:
        if penalty > 0 and reason:
            total_penalty += penalty
            reasons.append(f"[{name}] {reason}: -{penalty:.1f}")
            breakdown[name] = penalty
        else:
            breakdown[name] = 0.0

    # Cap total penalty — JD-specific penalties can be severe but not total annihilation.
    # Individual checks already carry their own weight; composite uses this as a multiplier.
    total_penalty = min(total_penalty, 50.0)
    total_penalty = round(total_penalty, 2)

    metadata = {"breakdown": breakdown}

    return PenaltyResult(
        penalty_score=total_penalty,
        reasons=reasons,
        metadata=metadata,
    )

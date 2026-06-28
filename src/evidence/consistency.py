"""Cross-signal consistency analysis for the evidence layer.

The :class:`ConsistencyAnalyzer` measures how well the extracted signals on a
candidate *agree with one another*. It compares pairs of independent signals —
for example the headline versus the current title, or the seniority implied by
the title versus the seniority trajectory of the career history — and combines
the per-check agreement flags into a single normalized ``consistency_score``
in ``[0.0, 1.0]``.

This module is strictly observational:

- It computes its own consistency score from raw candidate signals.
- It **never** reads or modifies any score produced by other layers.
- It produces metadata describing each agreement and conflict so downstream
  consumers can explain the result.

Weights for each pairwise check are configurable via ``config/evidence_rules.yaml``
(loaded with the shared ``load_rules_yaml`` helper). Missing keys fall back to
sensible defaults so the analyzer works without any configuration file.
"""

from __future__ import annotations

import re

from src.features.base import load_rules_yaml
from src.models.candidate import Candidate

# Default per-check weights. They are normalized at computation time, so their
# relative magnitudes are all that matter. Each check contributes its weight to
# the denominator, and (weight * agreement_flag) to the numerator.
DEFAULT_WEIGHTS: dict[str, float] = {
    "headline_title": 1.0,        # headline seniority vs current-title seniority
    "title_career": 1.0,          # current-title seniority vs career seniority ceiling
    "title_skills_domain": 1.0,   # title domain keywords vs skill domain keywords
    "experience_yoe": 1.5,        # summed career tenure vs declared years of experience
    "education_skills": 0.5,      # education field-of-study vs skill domain keywords
    "assessment_skills": 1.0,     # skill-assessment subjects vs declared skills
}

# Domain keyword groups used for title-vs-skills and education-vs-skills checks.
# A candidate is considered "in agreement" when the title's domain group(s) and
# the skill set's domain group(s) overlap.
DOMAIN_GROUPS: dict[str, list[str]] = {
    "backend": ["backend", "back-end", "server", "api", "django", "flask", "fastapi",
                "spring", "node", "express", "rails", "golang", "java", "c#"],
    "frontend": ["frontend", "front-end", "react", "vue", "angular", "javascript",
                 "typescript", "css", "html", "nextjs", "next.js"],
    "data": ["data", "sql", "spark", "airflow", "etl", "warehouse", "dbt", "snowflake",
             "pipeline", "analytics", "bigquery"],
    "ml": ["ml", "machine learning", "deep learning", "nlp", "computer vision",
           "pytorch", "tensorflow", "transformer", "llm", "ai", "model"],
    "devops": ["devops", "kubernetes", "docker", "aws", "gcp", "azure", "terraform",
               "ci/cd", "jenkins", "cloud"],
    "mobile": ["mobile", "android", "ios", "swift", "kotlin", "flutter", "react native"],
}


class ConsistencyAnalyzer:
    """Compute a normalized consistency score and explanatory metadata.

    Parameters
    ----------
    config_path:
        Path to the YAML weight configuration. If the file is absent or a key
        is missing, :data:`DEFAULT_WEIGHTS` is used as the fallback.
    """

    DEFAULT_WEIGHTS: dict[str, float] = DEFAULT_WEIGHTS
    DOMAIN_GROUPS: dict[str, list[str]] = DOMAIN_GROUPS

    def __init__(self, config_path: str = "config/evidence_rules.yaml") -> None:
        rules = load_rules_yaml(config_path)
        configured = rules.get("consistency_weights", {})
        # Merge configured weights over defaults; only known keys are honored.
        self.weights: dict[str, float] = {
            **DEFAULT_WEIGHTS,
            **{k: float(v) for k, v in configured.items() if k in DEFAULT_WEIGHTS},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, candidate: Candidate) -> dict:
        """Return the consistency score and per-check metadata.

        The returned dict has the shape::

            {
                "consistency_score": float,     # in [0.0, 1.0]
                "checks": {
                    "<check_name>": {
                        "agreed": bool,
                        "weight": float,
                        "detail": str,           # human-readable explanation
                    }, ...
                },
                "agreements": [str, ...],        # detail for agreed checks
                "conflicts":  [str, ...],        # detail for conflicting checks
            }
        """
        checks = self._run_checks(candidate)
        total_weight = sum(c["weight"] for c in checks.values())
        if total_weight <= 0:
            score = 0.0
        else:
            score = sum(
                c["weight"] * (1.0 if c["agreed"] else 0.0)
                for c in checks.values()
            ) / total_weight

        # Clamp to [0, 1] to guard against floating point edge cases.
        score = max(0.0, min(1.0, round(score, 4)))

        agreements = [c["detail"] for c in checks.values() if c["agreed"]]
        conflicts = [c["detail"] for c in checks.values() if not c["agreed"]]

        return {
            "consistency_score": score,
            "checks": checks,
            "agreements": agreements,
            "conflicts": conflicts,
        }

    # ------------------------------------------------------------------
    # Check orchestration
    # ------------------------------------------------------------------
    def _run_checks(self, candidate: Candidate) -> dict[str, dict]:
        """Run every pairwise check and return the per-check metadata."""
        ctx = self._build_context(candidate)
        w = self.weights

        return {
            "headline_title": self._check_headline_title(ctx, w["headline_title"]),
            "title_career": self._check_title_career(ctx, w["title_career"]),
            "title_skills_domain": self._check_title_skills_domain(ctx, w["title_skills_domain"]),
            "experience_yoe": self._check_experience_yoe(ctx, candidate, w["experience_yoe"]),
            "education_skills": self._check_education_skills(ctx, w["education_skills"]),
            "assessment_skills": self._check_assessment_skills(ctx, candidate, w["assessment_skills"]),
        }

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------
    def _build_context(self, candidate: Candidate) -> dict:
        """Pre-compute normalized signals reused across multiple checks."""
        profile = candidate.profile
        headline = (profile.headline or "").lower() if profile else ""
        current_title = (profile.current_title or "").lower() if profile else ""

        career_titles = [
            (r.title or "").lower() for r in (candidate.career_history or [])
        ]

        skill_names = [
            (s.name or "").strip().lower() for s in (candidate.skills or [])
        ]
        skill_blob = " ".join(skill_names)

        edu_fields = [
            (e.field_of_study or "").lower() for e in (candidate.education or [])
        ]

        return {
            "headline": headline,
            "current_title": current_title,
            "career_titles": career_titles,
            "skill_names": skill_names,
            "skill_blob": skill_blob,
            "edu_fields": edu_fields,
            "headline_seniority": self._seniority(headline),
            "title_seniority": self._seniority(current_title),
            "title_domains": self._domains(current_title),
            "headline_domains": self._domains(headline),
            "skill_domains": self._domains(skill_blob),
        }

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    def _check_headline_title(self, ctx: dict, weight: float) -> dict:
        """Agree when headline and current title share a seniority band.

        If neither contains a detectable seniority token, the check is treated
        as neutral-agreement (``True``) since there is no evidence of conflict.
        """
        h, t = ctx["headline_seniority"], ctx["title_seniority"]
        if h is None and t is None:
            agreed = True
            detail = "Neither headline nor title declares a seniority level."
        elif h is None or t is None:
            agreed = True
            detail = "Seniority present in only one of headline/title; no conflict."
        else:
            agreed = h == t
            detail = (
                f"Headline seniority '{h}' vs title seniority '{t}': "
                f"{'match' if agreed else 'mismatch'}."
            )
        return {"agreed": agreed, "weight": weight, "detail": detail}

    def _check_title_career(self, ctx: dict, weight: float) -> dict:
        """Agree when current-title seniority does not exceed career ceiling.

        A current title more senior than every prior role (e.g. claiming
        "Director" with a history of only junior roles) is treated as a
        potential inconsistency.

        If either the current title seniority or all career history seniorities
        are unknown (None), the check is skipped (treated as neutral agreement).
        """
        if not ctx["career_titles"]:
            return {
                "agreed": True,
                "weight": weight,
                "detail": "No prior roles to compare against current title.",
            }

        title_sen = ctx["title_seniority"]
        if title_sen is None:
            return {
                "agreed": True,
                "weight": weight,
                "detail": "Current title seniority is unknown; skipping check.",
            }

        # Filter out unknown prior role seniorities.
        career_sens = [self._seniority(t) for t in ctx["career_titles"]]
        known_career_sens = [s for s in career_sens if s is not None]

        if not known_career_sens:
            return {
                "agreed": True,
                "weight": weight,
                "detail": "No prior roles with detectable seniority; skipping check.",
            }

        # Map seniority bands to ordinals for comparison.
        order = {"junior": 0, "mid": 1, "senior": 2, "lead": 3, "director": 4}
        title_rank = order.get(title_sen, 1)

        ceiling_rank = max(order.get(s, 1) for s in known_career_sens)

        # Map rank back to name for the detail string
        rank_to_sen = {v: k for k, v in order.items()}
        career_ceiling = rank_to_sen.get(ceiling_rank, "mid")

        # Agree when the current title is at or below the career ceiling.
        agreed = title_rank <= ceiling_rank
        detail = (
            f"Current title seniority '{title_sen}' vs career ceiling "
            f"'{career_ceiling}': {'consistent' if agreed else 'title above history'}."
        )
        return {"agreed": agreed, "weight": weight, "detail": detail}

    def _check_title_skills_domain(self, ctx: dict, weight: float) -> dict:
        """Agree when title and skill set share at least one domain group."""
        title_d = set(ctx["title_domains"])
        skill_d = set(ctx["skill_domains"])

        if not title_d or not skill_d:
            agreed = True
            detail = "Insufficient domain signal in title or skills to compare."
        else:
            overlap = title_d & skill_d
            agreed = bool(overlap)
            detail = (
                f"Title domains {sorted(title_d)} vs skill domains {sorted(skill_d)}: "
                f"overlap={sorted(overlap) or 'none'}."
            )
        return {"agreed": agreed, "weight": weight, "detail": detail}

    def _check_experience_yoe(self, ctx: dict, candidate: Candidate, weight: float) -> dict:
        """Agree when summed career tenure is within tolerance of declared YoE.

        Tolerance is generous (the declared ``years_of_experience`` often
        includes partial years, internships, or rounding). We treat anything
        within ±50% of the larger of the two values as consistent, with an
        absolute floor so very short careers are not penalized for small
        absolute differences.
        """
        profile_yoe = (
            float(candidate.profile.years_of_experience)
            if candidate.profile and candidate.profile.years_of_experience is not None
            else 0.0
        )
        total_months = sum(
            (r.duration_months or 0) for r in (candidate.career_history or [])
        )
        tenure_years = total_months / 12.0

        if profile_yoe <= 0 and tenure_years <= 0:
            agreed = True
            detail = "No experience signal on either side; nothing to compare."
        else:
            base = max(profile_yoe, tenure_years, 1.0)
            # 50% relative tolerance, with a 1-year absolute cushion.
            diff = abs(profile_yoe - tenure_years)
            agreed = diff <= max(base * 0.5, 1.0)
            detail = (
                f"Declared YoE {profile_yoe:.1f} vs summed tenure "
                f"{tenure_years:.1f} years (diff {diff:.1f}): "
                f"{'consistent' if agreed else 'mismatch'}."
            )
        return {"agreed": agreed, "weight": weight, "detail": detail}

    def _check_education_skills(self, ctx: dict, weight: float) -> dict:
        """Agree when education field-of-study shares a domain with skills.

        Lower-weighted because education and current skills commonly diverge
        legitimately (career changers). No education record => neutral agreement.
        """
        fields = ctx["edu_fields"]
        if not fields:
            agreed = True
            detail = "No education records to compare against skills."
        else:
            edu_domains: set[str] = set()
            for f in fields:
                edu_domains |= set(self._domains(f))
            skill_domains = set(ctx["skill_domains"])
            if not edu_domains or not skill_domains:
                agreed = True
                detail = "Insufficient domain signal to compare education and skills."
            else:
                overlap = edu_domains & skill_domains
                agreed = bool(overlap)
                detail = (
                    f"Education domains {sorted(edu_domains)} vs skill domains "
                    f"{sorted(skill_domains)}: overlap={sorted(overlap) or 'none'}."
                )
        return {"agreed": agreed, "weight": weight, "detail": detail}

    def _check_assessment_skills(self, ctx: dict, candidate: Candidate, weight: float) -> dict:
        """Agree when skill-assessment subjects overlap declared skills.

        Uses the platform's ``skill_assessment_scores`` map (subject -> score).
        No assessments => neutral agreement.
        """
        assessments = (
            candidate.redrob_signals.skill_assessment_scores
            if candidate.redrob_signals and candidate.redrob_signals.skill_assessment_scores
            else {}
        )
        if not assessments:
            agreed = True
            detail = "No skill-assessment scores recorded."
        else:
            assessed = {(k or "").strip().lower() for k in assessments.keys()}
            declared = set(ctx["skill_names"])
            overlap = assessed & declared
            agreed = bool(overlap)
            detail = (
                f"Assessed subjects {sorted(assessed)} vs declared skills: "
                f"overlap={sorted(overlap) or 'none'}."
            )
        return {"agreed": agreed, "weight": weight, "detail": detail}

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _seniority(text: str) -> str | None:
        """Return a coarse seniority band for ``text`` or ``None`` if undetectable.

        Bands (highest to lowest): ``director``, ``lead``, ``senior``, ``mid``,
        ``junior``. The first matching band wins.
        """
        if not text:
            return None
        t = text.lower()
        if any(_word(t, k) for k in ("director", "vp", "chief", "cto", "ceo", "cpo", "head", "principal", "staff")):
            return "director"
        if any(_word(t, k) for k in ("lead", "manager", "architect")):
            return "lead"
        if any(_word(t, k) for k in ("senior", "sr", "sr.")):
            return "senior"
        if any(_word(t, k) for k in ("junior", "jr", "jr.", "intern", "trainee", "associate", "entry")):
            return "junior"
        return None

    @staticmethod
    def _domains(text: str) -> list[str]:
        """Return the list of domain groups whose keywords appear in ``text``."""
        if not text:
            return []
        lowered = text.lower()
        matched: list[str] = []
        for group, keywords in DOMAIN_GROUPS.items():
            if any(_word(lowered, k) for k in keywords):
                matched.append(group)
        return matched


def _word(text: str, keyword: str) -> bool:
    """Whole-word, case-insensitive membership test for ``keyword`` in ``text``.

    Handles keywords with trailing or leading special characters (like 'c#', 'c++', or 'next.js')
    by using custom lookaround assertions instead of simple '\\b' word boundaries.
    """
    if not text or not keyword:
        return False
    start_boundary = r"\b" if keyword[0].isalnum() or keyword[0] == '_' else r"(?<!\w)"
    end_boundary = r"\b" if keyword[-1].isalnum() or keyword[-1] == '_' else r"(?!\w)"
    pattern = start_boundary + re.escape(keyword) + end_boundary
    return re.search(pattern, text) is not None

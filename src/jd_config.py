"""JD Configuration loader for the REDROB candidate ranking engine.

Loads config/jd_config.yaml once at module-import time and exposes a
frozen-dict singleton ``JD``. All scoring engines that need JD-specific
context import from here.

Usage::

    from src.jd_config import JD

    required = JD["required_skills"]
    target   = JD["target_title"]
"""

from __future__ import annotations

import yaml
from pathlib import Path

_JD_CONFIG_PATH = Path("config/jd_config.yaml")

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load_jd_config(path: Path) -> dict:
    """Load and return the raw JD config dict from YAML.

    Falls back to a minimal safe default if the file is missing or malformed,
    so the rest of the pipeline never crashes due to a missing config.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
    except (OSError, yaml.YAMLError):
        pass

    # Minimal fallback so scoring engines degrade gracefully.
    return {
        "target_title": None,
        "target_yoe": None,
        "ideal_yoe_min": None,
        "ideal_yoe_max": None,
        "required_skills": [],
        "preferred_skills": [],
        "anti_skills": [],
        "framework_only_skills": [],
        "production_keywords": [],
        "research_keywords": [],
        "domain_relevance_keywords": [],
        "preferred_cities": [],
        "acceptable_cities": [],
        "required_country": None,
        "required_work_mode": None,
        "it_services_companies": [],
        "top_tier_companies": [],
        "salary_band_min_lpa": 0,
        "salary_band_max_lpa": 999,
        "salary_extreme_threshold_lpa": 999,
        "honeypot_hard_filter_threshold": 40,
        "consulting_only_fraction": 0.8,
        "domain_mismatch_fraction": 0.5,
    }


# ---------------------------------------------------------------------------
# Module-level singleton — loaded ONCE per process
# ---------------------------------------------------------------------------

JD: dict = _load_jd_config(_JD_CONFIG_PATH)

# Pre-compute normalised lower-case sets for O(1) lookups at scoring time.
JD_REQUIRED_SKILLS: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("required_skills") or [])
)
JD_PREFERRED_SKILLS: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("preferred_skills") or [])
)
JD_ANTI_SKILLS: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("anti_skills") or [])
)
JD_IT_SERVICES: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("it_services_companies") or [])
)
JD_TOP_TIER: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("top_tier_companies") or [])
)
JD_PREFERRED_CITIES: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("preferred_cities") or [])
)
JD_ACCEPTABLE_CITIES: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("acceptable_cities") or [])
)
JD_PRODUCTION_KEYWORDS: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("production_keywords") or [])
)
JD_RESEARCH_KEYWORDS: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("research_keywords") or [])
)
JD_DOMAIN_KEYWORDS: frozenset[str] = frozenset(
    s.lower() for s in (JD.get("domain_relevance_keywords") or [])
)


__all__ = [
    "JD",
    "JD_REQUIRED_SKILLS",
    "JD_PREFERRED_SKILLS",
    "JD_ANTI_SKILLS",
    "JD_IT_SERVICES",
    "JD_TOP_TIER",
    "JD_PREFERRED_CITIES",
    "JD_ACCEPTABLE_CITIES",
    "JD_PRODUCTION_KEYWORDS",
    "JD_RESEARCH_KEYWORDS",
    "JD_DOMAIN_KEYWORDS",
]

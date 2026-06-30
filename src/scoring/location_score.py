"""Location scoring engine for the REDROB candidate ranking system.

Evaluates a candidate's location alignment with the JD's geographic
requirements:  India-based, preferably Pune/Noida, hybrid work mode.

This module is purely responsible for location scoring. It never computes
rankings or accesses other scoring components.

Scoring logic (JD: Pune/Noida hybrid, India only, no visa sponsorship):
  - India + preferred city (Pune/Noida)     → +15 base + possible work-mode bonus
  - India + acceptable city (Hyd/MBai/etc.) → +10 base
  - India + other city + willing to relocate → +5 base
  - India + other city, no relocation        → +3 base
  - Non-India candidate                      → heavy penalty (-20)
  - preferred_work_mode == hybrid/flexible   → +5 bonus
  - preferred_work_mode == remote only       → -5
"""

from __future__ import annotations

from src.models.candidate import Candidate
from src.scoring import ScoreResult
from src.jd_config import JD, JD_PREFERRED_CITIES, JD_ACCEPTABLE_CITIES

# Required country from JD config (normalised lower-case)
_REQUIRED_COUNTRY: str = (JD.get("required_country") or "india").lower()


def score_location(
    candidate: Candidate,
    location_features: dict,
) -> ScoreResult:
    """Score a candidate's location alignment with the JD.

    Args:
        candidate: The candidate to score.
        location_features: Output from ``LocationExtractor.extract()``.

    Returns:
        A ``ScoreResult`` with score in [0, 100]. Naturally bounded to
        ~35 for a perfect match (preferred city + hybrid mode).
    """
    reasons: list[str] = []
    score = 0.0

    city = (location_features.get("city") or "").lower().strip()
    country = (location_features.get("country") or "").lower().strip()
    work_mode = (location_features.get("preferred_work_mode") or "").lower().strip()
    willing_to_relocate = location_features.get("relocation_willingness", False)

    # --- Country check ---
    # Normalise country — handle common variants
    is_india = (
        "india" in country
        or country in ("in", "ind")
        or not country  # If no country declared, assume India (common in dataset)
    )

    if not is_india and country:
        # Non-India candidate — JD says no visa sponsorship
        score -= 20.0
        reasons.append(f"Non-India candidate ('{country}'): -20.0 (no visa sponsorship)")
    elif is_india and country:
        reasons.append("India-based candidate: country check passed")
    else:
        reasons.append("Country not specified: assuming India (neutral)")

    # --- City check (only meaningful if India-based or country unspecified) ---
    if is_india or not country:
        if city in JD_PREFERRED_CITIES:
            score += 15.0
            reasons.append(f"Preferred city ('{city}'): +15.0")
        elif city in JD_ACCEPTABLE_CITIES:
            score += 10.0
            reasons.append(f"Acceptable city ('{city}'): +10.0")
        elif city:
            if willing_to_relocate:
                score += 5.0
                reasons.append(f"City '{city}' not preferred but willing to relocate: +5.0")
            else:
                score += 3.0
                reasons.append(f"City '{city}' not in preferred list, no relocation: +3.0")
        else:
            # No city — minor bonus for being India-based
            score += 5.0
            reasons.append("City not specified, assumed India-based: +5.0")

    # --- Work mode alignment ---
    if work_mode in ("hybrid", "flexible", "hybrid/flexible"):
        score += 5.0
        reasons.append(f"Work mode '{work_mode}' aligns with JD hybrid requirement: +5.0")
    elif work_mode in ("remote", "fully remote", "wfh"):
        score -= 5.0
        reasons.append(f"Work mode '{work_mode}' misaligns with JD hybrid requirement: -5.0")
    elif work_mode in ("on-site", "onsite", "in-office"):
        score += 2.0
        reasons.append(f"Work mode '{work_mode}' (on-site, close to hybrid): +2.0")
    else:
        reasons.append(f"Work mode not specified ('{work_mode}'): neutral")

    # --- Relocation willingness bonus ---
    if willing_to_relocate and city not in JD_PREFERRED_CITIES:
        score += 2.0
        reasons.append("Willing to relocate: +2.0")

    # Clamp raw score to [-30, 35]
    score = max(-30.0, min(35.0, round(score, 2)))

    # Normalise to [0, 100] for the composite engine
    # Map [-30, 35] → [0, 100]
    normalised = (score + 30.0) / 65.0 * 100.0
    normalised = max(0.0, min(100.0, round(normalised, 2)))

    metadata = {
        "city": city,
        "country": country,
        "work_mode": work_mode,
        "willing_to_relocate": willing_to_relocate,
        "raw_location_score": score,
    }

    return ScoreResult(score=normalised, reasons=reasons, metadata=metadata)

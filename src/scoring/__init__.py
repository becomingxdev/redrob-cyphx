"""Modular scoring system for the REDROB candidate ranking engine.

Every scoring component works independently and returns structured results.
Only ``composite.py`` combines components into a final score. No module
should import ``main.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ScoreResult:
    """Uniform return type for all scoring engines.

    Attributes:
        score: Numeric score (0-100 scale for individual engines).
        reasons: Human-readable explanations for the score.
        metadata: Arbitrary engine-specific data (counts, flags, etc.).
    """

    score: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class PenaltyResult:
    """Return type for the penalty engine.

    Attributes:
        penalty_score: Non-negative penalty value (0 = no penalty).
        reasons: Human-readable penalty descriptions.
        metadata: Per-penalty breakdown.
    """

    penalty_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class HoneypotResult:
    """Return type for the honeypot detection engine.

    Attributes:
        suspicion_score: 0-100 suspicion level (0 = no suspicion).
        reasons: Human-readable suspicion descriptions.
        metadata: Per-check breakdown.
    """

    suspicion_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class FinalScore:
    """Composite score produced by the composite engine.

    Attributes:
        candidate_id: Identifier of the scored candidate.
        score: Final composite score (0-100).
        component_scores: Individual engine scores keyed by component name.
        penalty_result: Penalty engine output.
        honeypot_result: Honeypot engine output.
        reasons: Aggregated human-readable explanations.
    """

    candidate_id: str
    score: float
    component_scores: dict = field(default_factory=dict)
    penalty_result: PenaltyResult | None = None
    honeypot_result: HoneypotResult | None = None
    reasons: list[str] = field(default_factory=list)


# Default weight values (mirrors config/weights.yaml).
DEFAULT_WEIGHTS: dict[str, float] = {
    "title": 0.25,
    "skills": 0.30,
    "experience": 0.20,
    "education": 0.10,
    "behavior": 0.05,
    "confidence": 0.05,
    "consistency": 0.05,
}

DEFAULT_PENALTY_MULTIPLIER: float = 1.0
DEFAULT_HONEYPOT_MULTIPLIER: float = 1.0


def load_weights(path: str = "config/weights.yaml") -> dict:
    """Load scoring weights from a YAML file.

    Returns a dict with keys matching DEFAULT_WEIGHTS plus
    ``penalty_multiplier`` and ``honeypot_multiplier``.
    Falls back to defaults when the file is missing or incomplete.
    """
    from src.features.base import load_rules_yaml

    rules = load_rules_yaml(path)
    weights = dict(DEFAULT_WEIGHTS)
    for k, v in rules.items():
        if k in weights and isinstance(v, (int, float)):
            weights[k] = float(v)
    return weights


__all__ = [
    "ScoreResult",
    "PenaltyResult",
    "HoneypotResult",
    "FinalScore",
    "DEFAULT_WEIGHTS",
    "DEFAULT_PENALTY_MULTIPLIER",
    "DEFAULT_HONEYPOT_MULTIPLIER",
    "load_weights",
]

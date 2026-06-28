"""Confidence calculation for the evidence layer.

The :class:`ConfidenceCalculator` converts evidence metadata produced by the
:class:`~src.evidence.verifier.EvidenceVerifier` (and optionally the
:class:`~src.evidence.consistency.ConsistencyAnalyzer`) into a normalized
confidence value in ``[0.0, 1.0]``.

Principle
---------
Confidence reflects *how many independent sources corroborate* a candidate's
extracted features. A skill listed only in the skills section is less
trustworthy than one also described in the summary, demonstrated in past roles,
and exercised in projects. More independent, mutually-corroborating sources
should raise confidence.

This module computes confidence **only**. It performs no ranking and does not
assign any candidate-level score that could be mistaken for a ranking output.

Weights are configurable via ``config/evidence_rules.yaml``.
"""

from __future__ import annotations

from src.evidence.verifier import EvidenceVerifier, SOURCE_NAMES
from src.features.base import load_rules_yaml
from src.models.candidate import Candidate

# Per-source weights applied when a source supports a feature. Relative
# magnitudes only — they are normalized at computation time.
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "skills": 1.0,          # explicit declaration
    "experience": 1.2,      # demonstrated in role descriptions/titles
    "career_history": 1.0,  # appears in company/title/industry history
    "summary": 0.8,         # self-described in headline/summary
    "projects": 1.3,        # tangible project evidence (highest signal)
}

# Aggregate confidence is scaled by how many features (relative to the total)
# are corroborated by more than one independent source. These tunables shape
# that curve and can be overridden via config.
DEFAULT_PARAMS: dict[str, float] = {
    # Floor applied when there are features but none are corroborated.
    "min_confidence": 0.0,
    # Ceiling applied when every feature is maximally corroborated.
    "max_confidence": 1.0,
    # Multiplier on consistency score used to modulate final confidence.
    # 0.0 means consistency is ignored; 1.0 means it fully gates confidence.
    "consistency_weight": 0.15,
}


class ConfidenceCalculator:
    """Convert evidence metadata into a normalized confidence value.

    Parameters
    ----------
    config_path:
        Path to the YAML weight configuration. Missing file/keys fall back to
        :data:`DEFAULT_SOURCE_WEIGHTS` and :data:`DEFAULT_PARAMS`.
    """

    DEFAULT_SOURCE_WEIGHTS: dict[str, float] = DEFAULT_SOURCE_WEIGHTS
    DEFAULT_PARAMS: dict[str, float] = DEFAULT_PARAMS
    SOURCE_NAMES: tuple[str, ...] = SOURCE_NAMES

    def __init__(self, config_path: str = "config/evidence_rules.yaml") -> None:
        rules = load_rules_yaml(config_path)
        configured_weights = rules.get("confidence_source_weights", {})
        configured_params = rules.get("confidence_params", {})

        self.source_weights: dict[str, float] = {
            **DEFAULT_SOURCE_WEIGHTS,
            **{
                k: float(v)
                for k, v in configured_weights.items()
                if k in DEFAULT_SOURCE_WEIGHTS
            },
        }
        self.params: dict[str, float] = {
            **DEFAULT_PARAMS,
            **{
                k: float(v)
                for k, v in configured_params.items()
                if k in DEFAULT_PARAMS
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calculate(
        self,
        candidate: Candidate,
        evidence: dict | None = None,
        consistency: dict | None = None,
    ) -> dict:
        """Compute confidence for ``candidate`` from evidence metadata.

        Args:
            candidate: The extracted candidate (used to count features when
                ``evidence`` is not supplied).
            evidence: Optional pre-computed verifier output. If ``None``, the
                verifier is run internally.
            consistency: Optional pre-computed consistency output. If supplied,
                its ``consistency_score`` modulates the final confidence per
                :attr:`params` ``consistency_weight``.

        Returns:
            A dict of the form::

                {
                    "confidence": float,             # in [0.0, 1.0]
                    "feature_count": int,
                    "corroborated_feature_count": int,
                    "avg_support_per_feature": float,
                    "per_feature_confidence": {feature: float, ...},
                }
        """
        if evidence is None:
            evidence = EvidenceVerifier().verify(candidate)

        feature_count = len(evidence)
        per_feature = self._per_feature_confidence(evidence)

        if feature_count == 0:
            # No features to corroborate => we have no basis for confidence.
            raw = 0.0
            avg_support = 0.0
            corroborated = 0
        else:
            corroborated = sum(1 for v in per_feature.values() if v > 0.0)
            avg_support = sum(per_feature.values()) / feature_count
            # Final raw confidence = mean of per-feature confidences.
            raw = sum(per_feature.values()) / feature_count

        # Modulate by consistency if provided (rewards mutually-agreeing signals).
        consistency_factor = 1.0
        if consistency is not None:
            cs = consistency.get("consistency_score")
            if isinstance(cs, (int, float)):
                cw = self.params["consistency_weight"]
                # Blend toward the consistency score by `cw`.
                # consistency_factor in [1 - cw, 1] when cs in [0, 1].
                consistency_factor = (1.0 - cw) + cw * float(cs)

        confidence = raw * consistency_factor

        lo = self.params["min_confidence"]
        hi = self.params["max_confidence"]
        confidence = max(lo, min(hi, round(confidence, 4)))

        return {
            "confidence": confidence,
            "feature_count": feature_count,
            "corroborated_feature_count": corroborated,
            "avg_support_per_feature": round(avg_support, 4),
            "per_feature_confidence": {k: round(v, 4) for k, v in per_feature.items()},
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _per_feature_confidence(self, evidence: dict) -> dict[str, float]:
        """Compute a normalized confidence for each feature from its sources.

        Each feature's confidence is the sum of weights of the sources that
        support it, divided by the sum of weights of *all* possible sources.
        This is a strictly increasing function of the number of corroborating
        sources, which is the intended behavior.
        """
        total_possible = sum(self.source_weights.values())
        if total_possible <= 0:
            # Degenerate config: fall back to uniform counting.
            total_possible = float(len(SOURCE_NAMES))

        per_feature: dict[str, float] = {}
        for feature, support in evidence.items():
            supported_weight = 0.0
            for source in SOURCE_NAMES:
                if support.get(source):
                    supported_weight += self.source_weights.get(source, 0.0)
            per_feature[feature] = supported_weight / total_possible
        return per_feature

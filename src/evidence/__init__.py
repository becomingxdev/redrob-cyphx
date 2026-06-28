"""Evidence layer for the REDROB ranker.

This package collects and quantifies *evidence* only. It must never compute
ranking scores, apply penalties, or generate reasoning text. The three public
components are:

- :class:`~src.evidence.verifier.EvidenceVerifier` - determines where each
  extracted feature is supported across the candidate's sources.
- :class:`~src.evidence.consistency.ConsistencyAnalyzer` - measures how well
  extracted signals agree with one another.
- :class:`~src.evidence.confidence.ConfidenceCalculator` - converts evidence
  metadata into normalized confidence values.
"""

from src.evidence.confidence import ConfidenceCalculator
from src.evidence.consistency import ConsistencyAnalyzer
from src.evidence.verifier import EvidenceVerifier

__all__ = [
    "EvidenceVerifier",
    "ConsistencyAnalyzer",
    "ConfidenceCalculator",
]

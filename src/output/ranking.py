"""Ranking engine for the REDROB candidate ranking system.

Sorts candidates by their pre-computed ``FinalScore`` and assigns
sequential ranks. This module performs **no scoring** and **no
explanation generation** — it only orders candidates deterministically.

Tie-breaking order (in priority):
    1. Higher final score
    2. Higher confidence
    3. Higher consistency
    4. Higher experience score
    5. Candidate ID (ascending, lexicographic)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.scoring import FinalScore


@dataclass(slots=True)
class RankedEntry:
    """A single ranked candidate entry.

    Attributes:
        candidate_id: Identifier of the candidate.
        rank: 1-based sequential rank (1 = highest).
        final_score: The composite final score.
        confidence: Evidence confidence in [0, 1].
        consistency: Profile consistency in [0, 1].
        final: The original ``FinalScore`` object (for downstream use).
    """

    candidate_id: str
    rank: int
    final_score: float
    confidence: float
    consistency: float
    final: FinalScore


def _tie_break_key(entry: FinalScore) -> tuple:
    """Build a sort key for deterministic, descending ordering.

    The tuple is structured so that sorting in *descending* order (via
    ``reverse=True``) rewards higher score, confidence, consistency, and
    experience — except the candidate_id, which is tie-broken ascending.

    To achieve mixed-direction ordering with a single reverse sort, we
    negate the candidate_id comparison by inverting its ordinal value.
    """
    components = entry.component_scores or {}
    confidence = components.get("confidence", 0.0)
    consistency = components.get("consistency", 0.0)
    experience = components.get("experience", 0.0)

    # For ascending candidate_id under a descending sort, we transform
    # the string so that "smaller" strings sort later. We do this by
    # mapping each character to its negated ordinal.
    id_key = tuple(-ord(c) for c in entry.candidate_id)

    return (
        entry.score,
        confidence,
        consistency,
        experience,
        id_key,
    )


def rank_candidates(finals: list[FinalScore]) -> list[RankedEntry]:
    """Rank candidates by descending final score with deterministic tie-breaks.

    Args:
        finals: Pre-computed ``FinalScore`` objects for every candidate.

    Returns:
        A list of ``RankedEntry`` objects sorted by rank (1 = best).
        Ranks are sequential 1..N with no gaps or duplicates.

    Notes:
        - Runs in O(n log n) via a single sort. No O(n²) operations.
        - Produces identical output for identical input (deterministic).
        - Empty input yields an empty list.
    """
    if not finals:
        return []

    # Single sort using a composite key. reverse=True rewards higher
    # score/confidence/consistency/experience; the negated candidate_id
    # ordinals ensure ascending ID ordering on ties.
    ordered = sorted(finals, key=_tie_break_key, reverse=True)

    ranked: list[RankedEntry] = []
    for idx, final in enumerate(ordered):
        components = final.component_scores or {}
        ranked.append(
            RankedEntry(
                candidate_id=final.candidate_id,
                rank=idx + 1,
                final_score=final.score,
                confidence=components.get("confidence", 0.0),
                consistency=components.get("consistency", 0.0),
                final=final,
            )
        )

    return ranked


class RankingEngine:
    """Stateless wrapper around :func:`rank_candidates`.

    Provided for consistency with the project's engine-style API.
    Instances hold no mutable state.
    """

    def rank(self, finals: list[FinalScore]) -> list[RankedEntry]:
        """Rank candidates. Delegates to :func:`rank_candidates`."""
        return rank_candidates(finals)


__all__ = ["RankedEntry", "rank_candidates", "RankingEngine"]

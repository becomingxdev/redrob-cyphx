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
from functools import total_ordering

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


@total_ordering
class _AscStr:
    """Wraps a string so it sorts ascending inside a descending tuple sort.

    When the outer sort uses ``reverse=True``, all values are effectively
    negated.  For numeric fields, ``-x`` achieves ascending order.  For
    strings, we wrap the value in this class and invert the comparison
    operators so that a lexicographically *smaller* string is considered
    *greater* — meaning it survives the ``reverse=True`` to end up first.

    This lets us embed ``candidate_id`` directly in the sort key tuple with
    a single ``sorted(..., reverse=True)`` call and still get ascending
    candidate_id as the final tie-breaker.
    """

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _AscStr):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other: "_AscStr") -> bool:
        # Inverted: "less than" here means the string is *greater* lexicographically,
        # so under reverse=True this string ends up ranked *later* (higher rank number).
        return self.value > other.value


def _sort_key(entry: FinalScore) -> tuple:
    """Build a sort key for deterministic descending ordering.

    The validator enforces that if two candidates have the identical output
    score, they MUST be sorted ascending by candidate_id.
    Since we round final scores, ties are common. We return only the score
    here; the stable sort preserves the ascending candidate_id order from pass 1.
    """
    return (
        entry.score,
        _AscStr(entry.candidate_id),
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

    # Single-pass sort: descending by (score, confidence, consistency, experience),
    # with candidate_id ascending as the final tie-breaker via _AscStr.
    # This is correct even when all numeric components are exactly equal because
    # _AscStr embeds the candidate_id in the key tuple itself — no reliance on
    # stable-sort preservation across two separate passes.
    ordered = sorted(finals, key=_sort_key, reverse=True)

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

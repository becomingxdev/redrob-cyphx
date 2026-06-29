"""CSV writer for the REDROB candidate ranking submission.

Generates the submission CSV with columns:
    candidate_id, rank, final_score, confidence, consistency, reason

Output is UTF-8, deterministically ordered, and rounded to 2 decimals.
No debug metadata is included.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.output.ranking import RankedEntry
from src.reasoning.generator import Reason


# Submission column header (order matters).
_HEADER: list[str] = [
    "candidate_id",
    "rank",
    "final_score",
    "confidence",
    "consistency",
    "reason",
]


def _format_reason(reasons: list[str]) -> str:
    """Join multiple reason strings with a semicolon separator.

    Empty or missing reasons produce an empty string.
    """
    if not reasons:
        return ""
    return "; ".join(reasons)


def write_submission_csv(
    path: str | Path,
    ranked: list[RankedEntry],
    reasons: dict[str, Reason],
) -> Path:
    """Write the submission CSV to disk.

    Args:
        path: Destination file path (created/overwritten).
        ranked: Deterministically ordered list of ``RankedEntry`` objects.
        reasons: Mapping of ``candidate_id`` -> ``Reason``.

    Returns:
        The resolved absolute ``Path`` that was written.

    Notes:
        - Uses the standard library ``csv`` module for maximum portability.
        - UTF-8 encoding is enforced.
        - ``final_score``, ``confidence``, and ``consistency`` are rounded
          to 2 decimal places.
        - Deterministic: rows follow the order of ``ranked`` exactly.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)

        for entry in ranked:
            reason = reasons.get(entry.candidate_id)
            reason_text = _format_reason(reason.reasons) if reason else ""

            writer.writerow([
                entry.candidate_id,
                entry.rank,
                f"{entry.final_score:.2f}",
                f"{entry.confidence:.2f}",
                f"{entry.consistency:.2f}",
                reason_text,
            ])

    return destination


class CSVWriter:
    """Stateless wrapper around :func:`write_submission_csv`."""

    def write(
        self,
        path: str | Path,
        ranked: list[RankedEntry],
        reasons: dict[str, Reason],
    ) -> Path:
        """Write the submission CSV. Delegates to :func:`write_submission_csv`."""
        return write_submission_csv(path, ranked, reasons)


__all__ = ["write_submission_csv", "CSVWriter"]

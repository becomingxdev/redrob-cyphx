"""Streaming loader for candidate JSONL and JSONL.GZ datasets."""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from pathlib import Path

from src.models.candidate import Candidate
from src.models.factory import CandidateFactory
from src.parser.normalizer import normalize_candidate
from src.parser.validator import validate_candidate

logger = logging.getLogger(__name__)


def load_candidates(path: str) -> Iterator[Candidate]:
    """Stream Candidate objects from a .jsonl or .jsonl.gz file.

    Each line is parsed, normalized, validated, and converted into a
    ``Candidate`` dataclass via ``CandidateFactory``.  Invalid or malformed
    records are skipped with a warning log message — processing never
    terminates because of a single bad record.

    Args:
        path: Filesystem path to a ``.jsonl`` or ``.jsonl.gz`` file.

    Yields:
        Fully-constructed ``Candidate`` instances for every valid record.
    """
    file_path = Path(path)

    if not file_path.exists():
        logger.error("File not found: %s", path)
        return

    opener = _open_func(file_path)

    with opener(file_path, "rt", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            # --- parse JSON ---
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping line %d: malformed JSON", line_number
                )
                continue

            # --- validate structure ---
            try:
                is_valid, errors = validate_candidate(record)
            except Exception:
                logger.exception(
                    "Skipping line %d: unexpected validation error", line_number
                )
                continue

            if not is_valid:
                reason = "; ".join(errors)
                logger.warning(
                    "Skipping line %d (id=%s): %s",
                    line_number,
                    record.get("candidate_id", "unknown"),
                    reason,
                )
                continue

            # --- normalize ---
            try:
                cleaned = normalize_candidate(record)
            except Exception:
                logger.exception(
                    "Skipping line %d: unexpected normalization error", line_number
                )
                continue

            # --- build Candidate ---
            try:
                candidate = CandidateFactory.create(cleaned)
            except Exception:
                logger.exception(
                    "Skipping line %d (id=%s): factory error",
                    line_number,
                    record.get("candidate_id", "unknown"),
                )
                continue

            yield candidate


def _open_func(file_path: Path):
    """Return the correct open callable based on file extension."""
    if file_path.suffixes == [".jsonl", ".gz"] or file_path.name.endswith(".jsonl.gz"):
        return gzip.open
    return open

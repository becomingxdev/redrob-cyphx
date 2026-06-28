"""Feature-source verification for the evidence layer.

The :class:`EvidenceVerifier` reports, for every skill a candidate possesses,
which independent profile sources corroborate that skill.

Design (metadata-driven)
------------------------
The verifier trusts the :class:`~src.features.skills.SkillsExtractor` as the
single source of truth. The extractor already scans every profile section
(skills list, headline, summary, career history, projects, certifications),
normalizes aliases, and records — per skill — the ``sources`` that mentioned
it, the total ``occurrences``, and a few ``contexts`` snippets.

Therefore the verifier **never re-parses profile text**. It simply maps each
extracted skill record into evidence flags. This fixes the previous bug where
``apache kafka`` was reported as ``career_history: false`` even though the
extractor already knew it originated from ``career_history``.

Output shape
------------
Each evidence entry carries two complementary source sets plus the preserved
extractor metadata::

    {
        "apache spark": {
            # --- new fine-grained flags (directly from record["sources"]) ---
            "headline": True,
            "summary": True,
            "career_history": True,
            "projects": False,
            "certifications": False,
            "skills": True,
            # --- legacy flag (kept for the unmodifiable confidence engine) ---
            "experience": True,
            # --- aggregates + preserved metadata ---
            "source_support_count": 4,
            "occurrences": 5,
            "contexts": ["...", "...", "..."],
        }
    }

Backward compatibility
----------------------
- The module-level :data:`SOURCE_NAMES` tuple (``skills``, ``experience``,
  ``career_history``, ``summary``, ``projects``) is **unchanged** so that
  :class:`~src.evidence.confidence.ConfidenceCalculator` (which imports it and
  weights ``experience`` separately) keeps working without modification.
- The ``experience`` flag mirrors ``career_history`` because the extractor
  emits a single ``career_history`` source for both role titles and role
  descriptions. Legacy ``summary`` is identical to the new ``summary`` flag.
- :meth:`verify` accepts an optional pre-computed ``multi_source_skills`` list
  so callers may skip recompute; ``main.py`` calls ``verify(candidate)`` with
  no second argument and works unchanged (the verifier runs the extractor
  internally).

Constraints
-----------
* Deterministic, CPU-only, O(number of skills). No LLMs, no embeddings.
* Produces evidence metadata only — never scores, ranks, or penalizes.
"""

from __future__ import annotations

from src.features.skills import SkillsExtractor
from src.models.candidate import Candidate

# Canonical, ordered list of *legacy* evidence source names. Preserved verbatim
# so the unmodifiable ConfidenceCalculator (which imports SOURCE_NAMES and
# weights "experience" separately) continues to work.
SOURCE_NAMES: tuple[str, ...] = (
    "skills",
    "experience",
    "career_history",
    "summary",
    "projects",
)

# The new, fine-grained source flags reported by the Skills Extractor. These
# are the six independent sources the verifier derives directly from each skill
# record's ``sources`` list. Order is fixed for deterministic iteration.
NEW_SOURCE_NAMES: tuple[str, ...] = (
    "headline",
    "summary",
    "career_history",
    "projects",
    "certifications",
    "skills",
)

# Legacy "experience" mirrors the extractor's "career_history" source (the
# extractor records a single career source for both role titles and role
# descriptions). This alias documents that mapping in one place.
_LEGACY_EXPERIENCE_MIRRORS = "career_history"


class EvidenceVerifier:
    """Determine where each extracted skill is supported on a candidate.

    The verifier consumes structured skill records produced by the
    :class:`SkillsExtractor`. It does **not** re-scan profile text; it trusts
    the extractor's ``sources`` / ``occurrences`` / ``contexts`` metadata and
    only projects that metadata into evidence flags.

    A single :class:`SkillsExtractor` is instantiated lazily and cached on the
    instance so repeated ``verify`` calls reuse the same extractor (its rules
    are loaded once from YAML).
    """

    SOURCE_NAMES: tuple[str, ...] = SOURCE_NAMES
    NEW_SOURCE_NAMES: tuple[str, ...] = NEW_SOURCE_NAMES

    def __init__(self) -> None:
        # Lazily constructed on first use. Cached thereafter so the YAML rules
        # are parsed only once, regardless of how many candidates are verified.
        self._skills_extractor: SkillsExtractor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def verify(
        self,
        candidate: Candidate,
        multi_source_skills: list[dict] | None = None,
    ) -> dict:
        """Build per-skill evidence metadata for ``candidate``.

        Args:
            candidate: An extracted :class:`Candidate` object.
            multi_source_skills: Optional pre-computed list of skill records as
                produced by ``SkillsExtractor.extract(...)["multi_source_skills"]``.
                When ``None`` (the default — and how ``main.py`` calls us), the
                verifier runs the extractor internally to obtain the records.
                Passing it explicitly lets a caller reuse work already done.

        Returns:
            A mapping ``{skill_name: {source_flag: bool, ...,
            "source_support_count": int, "occurrences": int,
            "contexts": [str, ...]}}`` keyed by normalized, lower-case skill
            name (the record's ``name`` field). One entry is produced per
            extracted skill; a candidate with no skills yields ``{}``.
        """
        records = self._get_skill_records(candidate, multi_source_skills)

        # O(number of skills): one projection per record. Deterministic output
        # because the extractor sorts records by canonical name.
        return {
            record.get("name", "").lower(): self._record_to_evidence(record)
            for record in records
            if record.get("name")  # skip malformed records missing a name
        }

    # ------------------------------------------------------------------
    # Record acquisition
    # ------------------------------------------------------------------
    def _get_skill_records(
        self,
        candidate: Candidate,
        multi_source_skills: list[dict] | None,
    ) -> list[dict]:
        """Return the list of skill-record dicts to project into evidence.

        Trusts a caller-supplied ``multi_source_skills`` when provided;
        otherwise runs the :class:`SkillsExtractor` on ``candidate``. We never
        re-parse profile text here — the extractor is the single source of
        truth.
        """
        if multi_source_skills is not None:
            return multi_source_skills
        if self._skills_extractor is None:
            self._skills_extractor = SkillsExtractor()
        extracted = self._skills_extractor.extract(candidate)
        return extracted.get("multi_source_skills", [])

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _record_to_evidence(record: dict) -> dict:
        """Project one skill record into its evidence metadata dict.

        Reads defensively (``.get`` with defaults) so a partial record never
        raises; missing sections simply report ``False``.
        """
        sources = record.get("sources") or []

        # --- new fine-grained flags: one per extractor source ---
        evidence: dict = {source: (source in sources) for source in NEW_SOURCE_NAMES}

        # --- legacy flag: "experience" mirrors the extractor's career source.
        # The extractor emits a single "career_history" source for both role
        # titles and role descriptions, so legacy experience == career_history.
        # Kept so the unmodifiable ConfidenceCalculator (which weights
        # "experience" separately) keeps working unchanged.
        evidence["experience"] = _LEGACY_EXPERIENCE_MIRRORS in sources

        # --- aggregate: count of UNIQUE supporting sources (new flags only).
        # Per the spec this is source diversity, not keyword frequency.
        evidence["source_support_count"] = sum(
            1 for source in NEW_SOURCE_NAMES if evidence[source]
        )

        # --- preserved metadata: echo occurrences and contexts verbatim ---
        evidence["occurrences"] = record.get("occurrences", 0)
        # Copy the list so callers cannot mutate the extractor's record, and to
        # preserve the snippets exactly as provided (no truncation).
        evidence["contexts"] = list(record.get("contexts") or [])

        return evidence

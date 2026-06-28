"""Feature-source verification for the evidence layer.

The :class:`EvidenceVerifier` inspects an extracted :class:`Candidate` and, for
each feature (currently each skill), determines which independent sources on the
profile also support that feature.

It returns **structured evidence metadata only**. It never assigns scores,
penalties, or ranks.

Sources considered
------------------
The verifier consults the following candidate fields. Each is treated as an
*independent* source of evidence:

- ``skills``            - the explicit skills list.
- ``experience``        - free-text descriptions of past/current roles, plus
                          role titles.
- ``career_history``    - company names, role titles, and industries held.
- ``summary``           - the profile headline and summary text.
- ``projects``          - project records, *if present*. The current domain
                          schema does not model projects, so this source is read
                          defensively from ``candidate.raw`` and silently
                          treated as empty when absent.

The set of source names is exposed as :attr:`EvidenceVerifier.SOURCE_NAMES` so
downstream consumers (e.g. the confidence calculator) can iterate deterministically.
"""

from __future__ import annotations
import re
from src.models.candidate import Candidate

# Canonical, ordered list of evidence source names. Order is fixed so that
# downstream consumers iterate deterministically regardless of dict ordering.
SOURCE_NAMES: tuple[str, ...] = (
    "skills",
    "experience",
    "career_history",
    "summary",
    "projects",
)


class EvidenceVerifier:
    """Determine where each extracted feature is supported on a candidate.

    A feature is a normalized, lower-case string (typically a skill name). For
    each feature the verifier reports a ``bool`` per source indicating whether
    that source mentions the feature.

    Example output::

        {
            "python": {
                "skills": True,
                "experience": True,
                "career_history": False,
                "summary": False,
                "projects": False,
                "source_support_count": 2,
            }
        }
    """

    SOURCE_NAMES: tuple[str, ...] = SOURCE_NAMES

    def __init__(self) -> None:
        # The verifier is intentionally stateless: it holds no mutable data so
        # that results are fully determined by its input candidate.
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def verify(self, candidate: Candidate) -> dict:
        """Build per-feature evidence metadata for ``candidate``.

        Args:
            candidate: An extracted :class:`Candidate` object.

        Returns:
            A mapping ``{feature: {source: bool, ..., "source_support_count": int}}``
            keyed by normalized feature name. One entry is produced per unique
            skill on the candidate (features with no skills yield ``{}``).
        """
        source_blobs = self._build_source_blobs(candidate)
        skills_set = source_blobs.pop("__skills_set")
        features = self._extract_features(candidate)

        evidence: dict[str, dict] = {}
        for feature in features:
            # Text-based sources are matched via tokenized substring logic.
            per_source = {
                source: self._feature_in_blob(feature, blob)
                for source, blob in source_blobs.items()
            }
            # ``skills`` is a structured list, not free text, so it is verified
            # via direct membership rather than substring matching.
            per_source["skills"] = feature in skills_set
            per_source["source_support_count"] = sum(1 for v in per_source.values() if v)
            evidence[feature] = per_source
        return evidence

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features(candidate: Candidate) -> list[str]:
        """Return the sorted, de-duplicated, normalized feature (skill) names."""
        skills = candidate.skills if candidate.skills else []
        names = {
            s.name.strip().lower()
            for s in skills
            if s.name and s.name.strip()
        }
        return sorted(names)

    # ------------------------------------------------------------------
    # Source blob construction
    # ------------------------------------------------------------------
    def _build_source_blobs(self, candidate: Candidate) -> dict:
        """Collect a lower-case text blob for each text-based source.

        Returns a dict containing both the text blobs (keyed by source name) and
        a special ``__skills_set`` entry holding the normalized skill-name set
        for direct membership checks.
        """
        skills_set = {
            s.name.strip().lower()
            for s in (candidate.skills or [])
            if s.name and s.name.strip()
        }

        return {
            # Text-based sources (matched via tokenized substring logic).
            "experience": self._experience_blob(candidate),
            "career_history": self._career_blob(candidate),
            "summary": self._summary_blob(candidate),
            "projects": self._projects_blob(candidate),
            # Membership-based source.
            "__skills_set": skills_set,
        }

    @staticmethod
    def _experience_blob(candidate: Candidate) -> str:
        """Concatenate role titles and descriptions from career history."""
        parts: list[str] = []
        for role in candidate.career_history or []:
            if role.title:
                parts.append(role.title)
            if role.description:
                parts.append(role.description)
        return " ".join(parts).lower()

    @staticmethod
    def _career_blob(candidate: Candidate) -> str:
        """Concatenate company names, titles, and industries from career history."""
        parts: list[str] = []
        for role in candidate.career_history or []:
            if role.company:
                parts.append(role.company)
            if role.title:
                parts.append(role.title)
            if role.industry:
                parts.append(role.industry)
        return " ".join(parts).lower()

    @staticmethod
    def _summary_blob(candidate: Candidate) -> str:
        """Concatenate the profile headline and summary."""
        profile = candidate.profile
        if profile is None:
            return ""
        parts: list[str] = []
        if profile.headline:
            parts.append(profile.headline)
        if profile.summary:
            parts.append(profile.summary)
        return " ".join(parts).lower()

    @staticmethod
    def _projects_blob(candidate: Candidate) -> str:
        """Concatenate project text, if projects are present in ``candidate.raw``.

        The current domain schema does not model projects, so this reads
        defensively from the raw payload. Each project may be a string or a
        dict with ``name``/``description``/``tech``/``technologies`` fields.
        """
        raw = candidate.raw or {}
        projects = raw.get("projects")
        if not projects:
            return ""
        parts: list[str] = []
        for project in projects:
            if isinstance(project, str):
                parts.append(project)
            elif isinstance(project, dict):
                for key in ("name", "title", "description", "tech", "technologies"):
                    val = project.get(key)
                    if isinstance(val, str):
                        parts.append(val)
                    elif isinstance(val, list):
                        parts.extend(v for v in val if isinstance(v, str))
        return " ".join(parts).lower()

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------
    def _feature_in_blob(self, feature: str, blob: str) -> bool:
        """Return ``True`` if ``feature`` is mentioned in ``blob``.

        Matching is deterministic and case-insensitive (blobs are pre-lowered).

        - Single-token features are matched as whole words to avoid false
          positives (e.g. ``"sql"`` must not match inside ``"mysql"``).
        - Multi-token features (e.g. ``"machine learning"``) are matched as a
          contiguous, whitespace-normalized phrase.
        - Supports special character endings (e.g. ``"c#"`` or ``"c++"``) by
          using custom lookarounds instead of trailing word boundaries.
        """
        if not feature or not blob:
            return False

        # Normalize internal whitespace in both feature and blob.
        feature_norm = " ".join(feature.split())
        if not feature_norm:
            return False

        # Build boundaries that handle trailing/leading non-word characters
        # like '#' in 'c#' or '+' in 'c++'.
        start_boundary = r"\b" if feature_norm[0].isalnum() or feature_norm[0] == "_" else r"(?<!\w)"
        end_boundary = r"\b" if feature_norm[-1].isalnum() or feature_norm[-1] == "_" else r"(?!\w)"
        pattern = start_boundary + re.escape(feature_norm) + end_boundary

        collapsed = re.sub(r"\s+", " ", blob)
        return re.search(pattern, collapsed) is not None

"""Multi-source skill collection and normalization for the REDROB feature pipeline.

Architecture
------------
The :class:`SkillsExtractor` collects skills from six independent sources in
priority order:

1. **Skills list** — structured ``candidate.skills`` entries (explicit declaration).
2. **Headline**    — pipe- / comma-separated tokens in the profile headline.
3. **Summary**     — comma- / newline-tokenized text in the profile summary.
4. **Career history** — every role ``description`` and ``title`` field.
5. **Projects**    — raw project records in ``candidate.raw["projects"]``.
6. **Certifications** — certification names from ``candidate.certifications``.

For each source the extractor:

* Tokenizes or reads structured fields.
* Applies the alias → canonical normalization map from
  ``config/skill_normalization.yaml``.
* Filters soft skills using the blocklist in the same file.
* Merges duplicates across sources, accumulating ``sources``,
  ``occurrences``, and ``contexts`` lists in a :class:`_SkillRecord`.

Backward Compatibility
----------------------
The public :meth:`SkillsExtractor.extract` return shape is **fully
preserved**. All keys produced by the original extractor (``skills``,
``unique_skills``, ``proficiency_distribution``, ``endorsement_total``,
``duration_statistics``, ``expert_count``, …) are still present. A new key
``multi_source_skills`` is appended, containing the richer per-skill objects
with source and context metadata.

Constraints
-----------
* Deterministic — given the same ``Candidate`` the output is identical.
* CPU-only — no embeddings, no LLMs, no external API calls.
* No scoring, ranking, weights, or confidence values are computed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.features.base import FeatureExtractor, load_rules_yaml
from src.models.candidate import Candidate

# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

@dataclass
class _SkillRecord:
    """Accumulates evidence for a single normalized skill across all sources."""

    name: str                       # lower-cased canonical key (used as dict key)
    normalized_name: str            # display form from the alias map or title-cased raw
    sources: list[str] = field(default_factory=list)      # ordered, deduplicated source names
    occurrences: int = 0            # total mentions across all sources
    contexts: list[str] = field(default_factory=list)     # up to N short snippets

    def add(self, source: str, context: str = "", max_contexts: int = 3) -> None:
        """Record one occurrence from ``source`` with an optional context snippet."""
        self.occurrences += 1
        if source not in self.sources:
            self.sources.append(source)
        if context and len(self.contexts) < max_contexts:
            snippet = context.strip()[:120]
            if snippet and snippet not in self.contexts:
                self.contexts.append(snippet)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "normalized_name": self.normalized_name,
            "sources": list(self.sources),
            "occurrences": self.occurrences,
            "contexts": list(self.contexts),
        }


# ---------------------------------------------------------------------------
# Internal skill collector
# ---------------------------------------------------------------------------

class _SkillCollector:
    """Orchestrates multi-source skill extraction for a single candidate.

    Parameters
    ----------
    alias_map:
        Lower-cased alias → canonical display name. Built once from YAML.
    soft_skills:
        Set of lower-cased strings to ignore during free-text extraction.
    max_contexts:
        Maximum number of context snippets retained per skill.
    min_token_len:
        Minimum character length for a single free-text token.
    max_token_words:
        Maximum number of space-separated words in a multi-word skill phrase
        considered when scanning free text against the alias map.
    """

    # Source name constants — used as dict keys and in ``sources`` lists.
    SRC_SKILLS      = "skills"
    SRC_HEADLINE    = "headline"
    SRC_SUMMARY     = "summary"
    SRC_CAREER      = "career_history"
    SRC_PROJECTS    = "projects"
    SRC_CERTS       = "certifications"

    def __init__(
        self,
        alias_map: dict[str, str],
        soft_skills: set[str],
        max_contexts: int = 3,
        min_token_len: int = 2,
        max_token_words: int = 4,
    ) -> None:
        self._alias_map     = alias_map      # lower raw → canonical display
        self._soft_skills   = soft_skills
        self._max_contexts  = max_contexts
        self._min_token_len = min_token_len
        self._max_token_words = max_token_words
        # registry: lower canonical name → _SkillRecord
        self._registry: dict[str, _SkillRecord] = {}
        # Lookup set: union of all alias *keys* plus lower-cased canonical
        # *values* (e.g. "airflow", "dbt", "snowflake").
        # This lets free-text extraction match canonical names even when they
        # are not yet in the registry (e.g. empty explicit skills list).
        self._lookup_set: set[str] = set(alias_map.keys()) | {
            v.lower() for v in alias_map.values()
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def collect(self, candidate: Candidate) -> list[_SkillRecord]:
        """Run all sources and return deduplicated, merged skill records."""
        self._registry.clear()
        self._collect_from_skills_list(candidate)
        self._collect_from_headline(candidate)
        self._collect_from_summary(candidate)
        self._collect_from_career_history(candidate)
        self._collect_from_projects(candidate)
        self._collect_from_certifications(candidate)
        # Sort deterministically by canonical name
        return sorted(self._registry.values(), key=lambda r: r.name)

    # ------------------------------------------------------------------
    # Source handlers
    # ------------------------------------------------------------------

    def _collect_from_skills_list(self, candidate: Candidate) -> None:
        """Read structured ``candidate.skills`` entries — highest priority source."""
        for skill in (candidate.skills or []):
            raw = (skill.name or "").strip()
            if not raw:
                continue
            self._record(raw, self.SRC_SKILLS, context=raw)

    def _collect_from_headline(self, candidate: Candidate) -> None:
        """Extract pipe- / comma- / space-separated tokens from the headline.

        Each pipe/comma segment is tried as a multi-word phrase against the
        lookup set first. Unmatched segments are passed through _extract_from_text
        so that alias phrases inside a segment are also caught.
        """
        headline = (candidate.profile.headline or "") if candidate.profile else ""
        if not headline.strip():
            return
        parts = re.split(r"[|,;\n]+", headline)
        for part in parts:
            token = part.strip()
            if not token or self._is_soft_skill(token):
                continue
            # Try the whole segment as a phrase first (handles "Machine Learning" etc.)
            phrase_key = " ".join(token.lower().split())
            if phrase_key in self._lookup_set or phrase_key in self._alias_map:
                self._record(token, self.SRC_HEADLINE, context=headline[:120])
            else:
                # Fall back to token-by-token scan for aliases within the segment
                self._extract_from_text(token, self.SRC_HEADLINE)

    def _collect_from_summary(self, candidate: Candidate) -> None:
        """Extract skills from the profile summary using tokenization + alias matching."""
        summary = (candidate.profile.summary or "") if candidate.profile else ""
        if not summary.strip():
            return
        self._extract_from_text(summary, self.SRC_SUMMARY)

    def _collect_from_career_history(self, candidate: Candidate) -> None:
        """Scan every role description and title for known skills."""
        for role in (candidate.career_history or []):
            # Role title can carry implicit skill signal (e.g. "Python Engineer")
            if role.title:
                self._extract_from_text(role.title, self.SRC_CAREER)
            # Role description is the richest source
            if role.description:
                self._extract_from_text(role.description, self.SRC_CAREER)

    def _collect_from_projects(self, candidate: Candidate) -> None:
        """Read projects defensively from ``candidate.raw`` and extract skills."""
        raw = candidate.raw or {}
        projects = raw.get("projects")
        if not projects:
            return
        for project in projects:
            if isinstance(project, str):
                self._extract_from_text(project, self.SRC_PROJECTS)
            elif isinstance(project, dict):
                for key in ("name", "title", "description", "tech", "technologies", "tools"):
                    val = project.get(key)
                    if isinstance(val, str):
                        self._extract_from_text(val, self.SRC_PROJECTS)
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                self._extract_from_text(item, self.SRC_PROJECTS)

    def _collect_from_certifications(self, candidate: Candidate) -> None:
        """Extract technology names from certification names and issuers."""
        for cert in (candidate.certifications or []):
            if cert.name:
                self._extract_from_text(cert.name, self.SRC_CERTS)

    # ------------------------------------------------------------------
    # Core extraction helpers
    # ------------------------------------------------------------------

    def _extract_from_text(self, text: str, source: str) -> None:
        """Scan ``text`` for known alias keys (greedy, longest-match first).

        Strategy
        --------
        1. Tokenize ``text`` into a word list (lower-cased).
        2. Slide a window of length ``max_token_words`` down to 1.
           For each window, check if the joined phrase appears in the alias map
           or in the skills-list-seeded registry.
        3. If a match is found, consume those tokens (skip them for shorter
           windows) and record the skill.

        This greedy longest-match ensures "Spark Streaming" normalizes to
        "Apache Spark" rather than being split into "Spark" + "Streaming".
        """
        if not text or not text.strip():
            return

        # Tokenize: split on whitespace, punctuation, bullets
        raw_tokens = re.split(r"[\s,;|\n•\-–—/\\]+", text.lower())
        tokens = [t.strip("().\"':!?") for t in raw_tokens if t.strip()]

        i = 0
        while i < len(tokens):
            matched = False
            # Try longest phrase first
            for window in range(
                min(self._max_token_words, len(tokens) - i), 0, -1
            ):
                phrase = " ".join(tokens[i : i + window])
                if len(phrase) < self._min_token_len:
                    continue
                if self._is_soft_skill(phrase):
                    break
                # Match against: alias map keys, canonical value names, or
                # skills already recorded (handles explicit skills list entries).
                if phrase in self._lookup_set or phrase in self._registry:
                    context = self._extract_context(text, phrase)
                    self._record(phrase, source, context=context)
                    i += window
                    matched = True
                    break
            if not matched:
                i += 1

    def _record(self, raw: str, source: str, context: str = "") -> None:
        """Normalize ``raw`` and upsert into the registry."""
        canonical_display = self._normalize(raw)
        key = canonical_display.lower()
        if key not in self._registry:
            self._registry[key] = _SkillRecord(
                name=key,
                normalized_name=canonical_display,
            )
        self._registry[key].add(source, context=context, max_contexts=self._max_contexts)

    def _normalize(self, raw: str) -> str:
        """Map ``raw`` to its canonical display name.

        Lookup is case-insensitive. If no alias is found, the raw value is
        returned with its original casing preserved (stripped).
        """
        key = " ".join(raw.strip().lower().split())  # collapse internal whitespace
        if key in self._alias_map:
            return self._alias_map[key]
        return raw.strip()

    def _is_soft_skill(self, token: str) -> bool:
        """Return True if ``token`` (lower-cased) is on the soft-skill blocklist."""
        return token.lower() in self._soft_skills

    @staticmethod
    def _extract_context(text: str, phrase: str) -> str:
        """Return a short snippet of ``text`` surrounding the first occurrence of ``phrase``."""
        lo = text.lower()
        idx = lo.find(phrase.lower())
        if idx == -1:
            return text.strip()[:120]
        start = max(0, idx - 30)
        end   = min(len(text), idx + len(phrase) + 60)
        snippet = text[start:end].strip()
        return snippet[:120]


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------

class SkillsExtractor(FeatureExtractor):
    """Extract structured skills features from a candidate using multiple sources.

    The extractor reads from:

    * ``candidate.skills``       — explicit skills list (highest priority)
    * ``candidate.profile.headline``
    * ``candidate.profile.summary``
    * ``candidate.career_history[*].description`` and ``.title``
    * ``candidate.raw["projects"]``  (defensive read)
    * ``candidate.certifications``

    Parameters
    ----------
    config_path:
        Path to the legacy proficiency-bucket YAML (``config/skill_rules.yaml``).
    normalization_config_path:
        Path to the alias-map / soft-skill / extraction-tuning YAML
        (``config/skill_normalization.yaml``).
    """

    def __init__(
        self,
        config_path: str = "config/skill_rules.yaml",
        normalization_config_path: str = "config/skill_normalization.yaml",
    ) -> None:
        # Legacy proficiency rules
        self.rules = load_rules_yaml(config_path)
        self.proficiency_buckets: dict[str, list[str]] = self.rules.get("proficiency_buckets", {})

        # Normalization / extraction rules
        norm_rules = load_rules_yaml(normalization_config_path)
        raw_aliases: dict[str, str] = norm_rules.get("aliases", {})
        # Lower-case both sides of the alias map for case-insensitive lookup
        self._alias_map: dict[str, str] = {
            " ".join(k.strip().lower().split()): v
            for k, v in raw_aliases.items()
        }

        raw_soft: list[str] = norm_rules.get("soft_skills", [])
        self._soft_skills: set[str] = {s.strip().lower() for s in raw_soft}

        extraction_cfg: dict = norm_rules.get("extraction", {})
        self._max_contexts  = int(extraction_cfg.get("max_contexts", 3))
        self._min_token_len = int(extraction_cfg.get("min_token_length", 2))
        self._max_token_words = int(extraction_cfg.get("max_tokens_per_skill", 4))

    # ------------------------------------------------------------------
    # Proficiency helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _normalize_proficiency(self, proficiency: str) -> str:
        """Map raw proficiency string to a standard bucket."""
        if not proficiency:
            return "unknown"
        p_low = proficiency.strip().lower()
        for bucket, keywords in self.proficiency_buckets.items():
            if p_low == bucket or p_low in keywords:
                return bucket
        return p_low

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, candidate: Candidate) -> dict:
        """Extract skills features from Candidate using all available sources.

        Returns
        -------
        dict
            All keys from the original extractor are preserved
            (``skills``, ``unique_skills``, ``proficiency_distribution``,
            ``endorsement_total``, ``duration_statistics``, ``expert_count``,
            ``advanced_count``, ``intermediate_count``, ``beginner_count``).

            A new key ``multi_source_skills`` contains a list of enriched
            skill dicts, each with ``name``, ``normalized_name``, ``sources``,
            ``occurrences``, and ``contexts``.
        """
        # ── Multi-source collection ─────────────────────────────────────────
        collector = _SkillCollector(
            alias_map=self._alias_map,
            soft_skills=self._soft_skills,
            max_contexts=self._max_contexts,
            min_token_len=self._min_token_len,
            max_token_words=self._max_token_words,
        )
        records: list[_SkillRecord] = collector.collect(candidate)

        # Flat, sorted, lower-cased unique skill names (union of all sources)
        all_skill_names: list[str] = sorted(r.name for r in records)
        unique_skills_count = len(all_skill_names)

        # ── Legacy stats (from structured skills list only) ─────────────────
        skills_list = candidate.skills if candidate.skills else []

        endorsement_total = sum(s.endorsements for s in skills_list if s.endorsements)

        proficiency_dist: dict[str, int] = {b: 0 for b in self.proficiency_buckets}
        proficiency_dist["unknown"] = 0

        for s in skills_list:
            norm_p = self._normalize_proficiency(s.proficiency)
            proficiency_dist.setdefault(norm_p, 0)
            proficiency_dist[norm_p] += 1

        durations = [
            s.duration_months
            for s in skills_list
            if s.duration_months is not None
        ]
        if durations:
            duration_stats = {
                "min": min(durations),
                "max": max(durations),
                "avg": round(sum(durations) / len(durations), 2),
                "total": sum(durations),
            }
        else:
            duration_stats = {"min": 0, "max": 0, "avg": 0.0, "total": 0}

        # ── Build result dict ───────────────────────────────────────────────
        res: dict = {
            "skills": all_skill_names,
            "unique_skills": unique_skills_count,
            "proficiency_distribution": {
                k: v for k, v in proficiency_dist.items() if v > 0
            },
            "endorsement_total": endorsement_total,
            "duration_statistics": duration_stats,
            "multi_source_skills": [r.to_dict() for r in records],
        }

        # Explicit bucket count keys (unchanged from original)
        for bucket in self.proficiency_buckets:
            res[f"{bucket}_count"] = proficiency_dist.get(bucket, 0)

        return res

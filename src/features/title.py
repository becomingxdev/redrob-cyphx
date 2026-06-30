"""PERF 4: Pre-compiled regex patterns for all keyword groups.
All re.compile calls now happen once at __init__ time, not per candidate.
"""

import re
from src.models.candidate import Candidate
from src.features.base import FeatureExtractor, load_rules_yaml


class TitleExtractor(FeatureExtractor):
    """Extracts structured title information and metadata from a candidate."""

    def __init__(self, config_path: str = "config/title_rules.yaml") -> None:
        self.rules = load_rules_yaml(config_path)
        self.seniority_levels: dict = self.rules.get("seniority_levels", {})
        self.management_keywords: list = self.rules.get("management_keywords", [])
        self.ai_ml_keywords: list = self.rules.get("ai_ml_keywords", [])

        # PERF 4: Pre-compile one combined pattern per seniority level so
        # extract() never calls re.compile at runtime.
        self._seniority_patterns: dict[str, re.Pattern] = {}
        for level, kws in self.seniority_levels.items():
            if kws:
                pat = r"\b(?:" + "|".join(re.escape(str(kw)) for kw in kws) + r")\b"
                self._seniority_patterns[level] = re.compile(pat, re.IGNORECASE)

        if self.management_keywords:
            mgmt_pat = r"\b(?:" + "|".join(re.escape(str(kw)) for kw in self.management_keywords) + r")\b"
            self._management_pattern: re.Pattern | None = re.compile(mgmt_pat, re.IGNORECASE)
        else:
            self._management_pattern = None

        if self.ai_ml_keywords:
            ai_pat = r"\b(?:" + "|".join(re.escape(str(kw)) for kw in self.ai_ml_keywords) + r")\b"
            self._ai_pattern: re.Pattern | None = re.compile(ai_pat, re.IGNORECASE)
        else:
            self._ai_pattern = None

    def extract(self, candidate: Candidate) -> dict:
        """Extract title facts from Candidate."""
        title = ""
        if candidate.profile and candidate.profile.current_title:
            title = candidate.profile.current_title.strip()

        normalized_title = title.lower()

        # Tokenize title: extract all word characters (alphanumeric)
        tokens = [t for t in re.findall(r"\b\w+\b", normalized_title) if t]

        # Determine seniority level using pre-compiled patterns (PERF 4).
        # Match in hierarchical order: principal, staff, lead, senior, junior, default to mid.
        seniority = "mid"
        for level in ["principal", "staff", "lead", "senior", "junior"]:
            pattern = self._seniority_patterns.get(level)
            if pattern and pattern.search(normalized_title):
                seniority = level
                break

        # Management indicator: pre-compiled pattern (PERF 4).
        is_manager = bool(self._management_pattern and self._management_pattern.search(normalized_title))

        # AI/ML relevance indicator: pre-compiled pattern (PERF 4).
        is_ai_related = bool(self._ai_pattern and self._ai_pattern.search(normalized_title))

        return {
            "title": title,
            "normalized_title": normalized_title,
            "seniority": seniority,
            "is_manager": is_manager,
            "is_ai_related": is_ai_related,
            "tokens": tokens,
        }

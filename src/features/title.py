import re
from src.models.candidate import Candidate
from src.features.base import FeatureExtractor, load_rules_yaml


class TitleExtractor(FeatureExtractor):
    """Extracts structured title information and metadata from a candidate."""

    def __init__(self, config_path: str = "config/title_rules.yaml") -> None:
        self.rules = load_rules_yaml(config_path)
        self.seniority_levels = self.rules.get("seniority_levels", {})
        self.management_keywords = self.rules.get("management_keywords", [])
        self.ai_ml_keywords = self.rules.get("ai_ml_keywords", [])

    def extract(self, candidate: Candidate) -> dict:
        """Extract title facts from Candidate."""
        title = ""
        if candidate.profile and candidate.profile.current_title:
            title = candidate.profile.current_title.strip()

        normalized_title = title.lower()

        # Tokenize title: extract all word characters (alphanumeric)
        tokens = [t for t in re.findall(r"\b\w+\b", normalized_title) if t]

        # Determine seniority level based on keyword presence in normalized title
        # Match in hierarchical order: principal, staff, lead, senior, junior, default to mid
        seniority = "mid"
        for level in ["principal", "staff", "lead", "senior", "junior"]:
            keywords = self.seniority_levels.get(level, [])
            # Search for keyword matches as whole words or substring matching where appropriate
            if any(re.search(rf"\b{re.escape(kw)}\b", normalized_title) for kw in keywords):
                seniority = level
                break

        # Management indicator: checks if any management keywords are present in the title
        is_manager = any(
            re.search(rf"\b{re.escape(kw)}\b", normalized_title) for kw in self.management_keywords
        )

        # AI/ML relevance indicator: checks if any AI/ML keywords are present in the title
        is_ai_related = any(
            re.search(rf"\b{re.escape(kw)}\b", normalized_title) for kw in self.ai_ml_keywords
        )

        return {
            "title": title,
            "normalized_title": normalized_title,
            "seniority": seniority,
            "is_manager": is_manager,
            "is_ai_related": is_ai_related,
            "tokens": tokens,
        }

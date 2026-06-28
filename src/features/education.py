import re

from src.models.candidate import Candidate
from src.features.base import FeatureExtractor


class EducationExtractor(FeatureExtractor):
    """Extracts structured education facts from a candidate."""

    # Word-boundary patterns for degree classification.
    # Using regex prevents false positives from substring matching
    # (e.g., "be" inside "degree", "ba" inside "diploma").
    _PHD_PATTERNS: list[str] = ["phd", r"ph\.?d", "doctor", "doctorate"]
    _MASTERS_PATTERNS: list[str] = [
        "master", r"m\.s", r"\bms\b", "mba", r"m\.tech", r"m\.sc", "mca", "pgdm",
    ]
    _BACHELORS_PATTERNS: list[str] = [
        "bachelor", r"b\.s", r"\bbs\b", r"b\.tech", r"b\.e", r"\bbe\b",
        r"b\.sc", r"\bbsc\b", r"b\.a", r"\bba\b", r"b\.com", r"\bbcom\b", r"\bbba\b",
    ]
    _DIPLOMA_PATTERNS: list[str] = ["diploma", "associate", "foundation"]
    _SCHOOL_PATTERNS: list[str] = ["school", r"high\s+school", "ssc", "hsc", "12th", "10th"]

    @staticmethod
    def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
        """Compile a list of regex patterns into compiled pattern objects."""
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def __init__(self) -> None:
        self._phd_res = self._compile_patterns(self._PHD_PATTERNS)
        self._masters_res = self._compile_patterns(self._MASTERS_PATTERNS)
        self._bachelors_res = self._compile_patterns(self._BACHELORS_PATTERNS)
        self._diploma_res = self._compile_patterns(self._DIPLOMA_PATTERNS)
        self._school_res = self._compile_patterns(self._SCHOOL_PATTERNS)

    def _degree_score(self, degree: str | None) -> int:
        """Map a degree string to a hierarchy level for comparison.

        Returns an ordinal so that higher degrees have higher scores.
        This is used only to determine the *highest* degree; no actual
        scoring points are assigned.
        """
        if not degree:
            return 0
        text = degree.strip()

        if any(p.search(text) for p in self._phd_res):
            return 5
        if any(p.search(text) for p in self._masters_res):
            return 4
        if any(p.search(text) for p in self._bachelors_res):
            return 3
        if any(p.search(text) for p in self._diploma_res):
            return 2
        if any(p.search(text) for p in self._school_res):
            return 1

        return 0

    def extract(self, candidate: Candidate) -> dict:
        """Extract education features from Candidate."""
        edu_list = candidate.education if candidate.education else []
        education_count = len(edu_list)

        highest_edu = None
        highest_score = -1

        for edu in edu_list:
            score = self._degree_score(edu.degree)
            if score > highest_score:
                highest_score = score
                highest_edu = edu
            elif score == highest_score and highest_edu is not None:
                # Tie breaker: choose the one with the later graduation year
                if edu.end_year > highest_edu.end_year:
                    highest_edu = edu

        if highest_edu:
            highest_degree = highest_edu.degree.strip() if highest_edu.degree else ""
            field_of_study = highest_edu.field_of_study.strip() if highest_edu.field_of_study else ""
            education_tier = highest_edu.tier.strip() if highest_edu.tier else ""
            graduation_year = highest_edu.end_year
        else:
            highest_degree = ""
            field_of_study = ""
            education_tier = ""
            graduation_year = 0

        return {
            "highest_degree": highest_degree,
            "field_of_study": field_of_study,
            "education_tier": education_tier,
            "graduation_year": graduation_year,
            "education_count": education_count,
        }

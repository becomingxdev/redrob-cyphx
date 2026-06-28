from src.models.candidate import Candidate
from src.features.base import FeatureExtractor, load_rules_yaml


class SkillsExtractor(FeatureExtractor):
    """Extracts structured skills facts from a candidate."""

    def __init__(self, config_path: str = "config/skill_rules.yaml") -> None:
        self.rules = load_rules_yaml(config_path)
        self.proficiency_buckets = self.rules.get("proficiency_buckets", {})

    def _normalize_proficiency(self, proficiency: str) -> str:
        """Map raw proficiency string to a standard bucket."""
        if not proficiency:
            return "unknown"
        p_low = proficiency.strip().lower()
        for bucket, keywords in self.proficiency_buckets.items():
            if p_low == bucket or p_low in keywords:
                return bucket
        return p_low

    def extract(self, candidate: Candidate) -> dict:
        """Extract skills features from Candidate."""
        skills_list = candidate.skills if candidate.skills else []

        normalized_names = sorted(list({s.name.strip().lower() for s in skills_list if s.name}))
        unique_skills_count = len(normalized_names)

        endorsement_total = sum(s.endorsements for s in skills_list if s.endorsements)

        # Proficiency distribution and counts
        proficiency_dist = {}
        for bucket in self.proficiency_buckets.keys():
            proficiency_dist[bucket] = 0
        proficiency_dist["unknown"] = 0

        for s in skills_list:
            norm_p = self._normalize_proficiency(s.proficiency)
            if norm_p not in proficiency_dist:
                proficiency_dist[norm_p] = 0
            proficiency_dist[norm_p] += 1

        # Duration statistics
        durations = [s.duration_months for s in skills_list if s.duration_months is not None]
        if durations:
            duration_stats = {
                "min": min(durations),
                "max": max(durations),
                "avg": round(sum(durations) / len(durations), 2),
                "total": sum(durations),
            }
        else:
            duration_stats = {"min": 0, "max": 0, "avg": 0.0, "total": 0}

        res = {
            "skills": normalized_names,
            "unique_skills": unique_skills_count,
            "proficiency_distribution": {k: v for k, v in proficiency_dist.items() if v > 0},
            "endorsement_total": endorsement_total,
            "duration_statistics": duration_stats,
        }

        # Add explicit count keys for main buckets
        for bucket in self.proficiency_buckets.keys():
            res[f"{bucket}_count"] = proficiency_dist.get(bucket, 0)

        return res

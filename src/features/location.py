from src.models.candidate import Candidate
from src.features.base import FeatureExtractor


class LocationExtractor(FeatureExtractor):
    """Extracts structured location facts from a candidate."""

    def extract(self, candidate: Candidate) -> dict:
        """Extract location features from Candidate."""
        city = ""
        country = ""
        preferred_work_mode = ""
        relocation_willingness = False

        if candidate.profile:
            if candidate.profile.location:
                city = candidate.profile.location.strip()
            if candidate.profile.country:
                country = candidate.profile.country.strip()

        if candidate.redrob_signals:
            if candidate.redrob_signals.preferred_work_mode:
                preferred_work_mode = candidate.redrob_signals.preferred_work_mode.strip()
            if candidate.redrob_signals.willing_to_relocate is not None:
                relocation_willingness = bool(candidate.redrob_signals.willing_to_relocate)

        return {
            "city": city,
            "country": country,
            "preferred_work_mode": preferred_work_mode,
            "relocation_willingness": relocation_willingness,
        }

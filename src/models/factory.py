from src.models.candidate import Candidate
from src.models.profile import Profile


class CandidateFactory:
    """Creates Candidate objects from validated dictionaries."""

    @staticmethod
    def create(candidate_dict: dict) -> Candidate:
        profile_data = candidate_dict["profile"]

        profile = Profile(
            anonymized_name=profile_data["anonymized_name"],
            headline=profile_data["headline"],
            summary=profile_data["summary"],
            location=profile_data["location"],
            country=profile_data["country"],
            years_of_experience=profile_data["years_of_experience"],
            current_title=profile_data["current_title"],
            current_company=profile_data["current_company"],
            current_company_size=profile_data["current_company_size"],
            current_industry=profile_data["current_industry"],
        )

        return Candidate(
            candidate_id=candidate_dict["candidate_id"],
            profile=profile,
            raw=candidate_dict,
        )
from src.models.candidate import Candidate


class CandidateFactory:
    """Creates Candidate objects from validated dictionaries."""

    @staticmethod
    def create(candidate_dict: dict) -> Candidate:
        return Candidate(raw=candidate_dict)
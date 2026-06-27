class CandidateValidator:
    """Validates the basic structure of candidate data."""

    @staticmethod
    def validate(candidate: dict) -> bool:
        if not isinstance(candidate, dict):
            return False

        return True
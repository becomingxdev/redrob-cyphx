from dataclasses import dataclass
from src.models.profile import Profile

@dataclass(slots=True)
class Candidate:
    candidate_id: str
    profile: Profile
    raw: dict
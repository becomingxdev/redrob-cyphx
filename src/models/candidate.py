from dataclasses import dataclass

from src.models.career import CareerHistory
from src.models.certification import Certification
from src.models.education import Education
from src.models.language import Language
from src.models.profile import Profile
from src.models.redrob import RedrobSignals
from src.models.skill import Skill


@dataclass(slots=True)
class Candidate:
    """Top-level domain model representing a single candidate."""

    candidate_id: str
    profile: Profile
    career_history: list[CareerHistory]
    education: list[Education]
    skills: list[Skill]
    certifications: list[Certification]
    languages: list[Language]
    redrob_signals: RedrobSignals
    raw: dict

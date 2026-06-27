from dataclasses import dataclass


@dataclass(slots=True)
class Language:
    """A language known by the candidate."""

    language: str
    proficiency: str

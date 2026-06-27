from dataclasses import dataclass


@dataclass(slots=True)
class Skill:
    """A skill possessed by the candidate."""

    name: str
    proficiency: str
    endorsements: int
    duration_months: int | None

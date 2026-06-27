from dataclasses import dataclass


@dataclass(slots=True)
class Education:
    """A single education record for the candidate."""

    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str | None
    tier: str

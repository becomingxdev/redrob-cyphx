from dataclasses import dataclass


@dataclass(slots=True)
class Certification:
    """A certification earned by the candidate."""

    name: str
    issuer: str
    year: int

from dataclasses import dataclass


@dataclass(slots=True)
class Candidate:
    """Internal representation of a candidate."""

    raw: dict
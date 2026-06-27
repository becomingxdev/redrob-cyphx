from dataclasses import dataclass


@dataclass(slots=True)
class SalaryRange:
    """Expected salary range in INR Lakhs Per Annum."""

    min: float
    max: float

from dataclasses import dataclass


@dataclass(slots=True)
class CareerHistory:
    """A single role held by the candidate."""

    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

from dataclasses import dataclass


@dataclass(slots=True)
class Profile:
    """Professional profile information for a candidate."""

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

from datetime import date, datetime
from src.models.candidate import Candidate
from src.features.base import FeatureExtractor


class CareerExtractor(FeatureExtractor):
    """Extracts structured career trajectory and stability facts from a candidate."""

    def _parse_date(self, date_str: str | None) -> date | None:
        """Parse YYYY-MM-DD or YYYY-MM date string into date object."""
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m"):
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None

    def _title_seniority_score(self, title: str | None) -> float:
        """Assign a simple numeric score to title seniority for progression mapping."""
        if not title:
            return 1.5
        t_low = title.lower()
        if any(w in t_low for w in ("principal", "distinguished", "fellow", "chief", "cto", "ceo", "cpo", "director", "vp")):
            return 4.0
        if any(w in t_low for w in ("lead", "manager", "head", "architect")):
            return 3.0
        if any(w in t_low for w in ("senior", "sr", "sr.")):
            return 2.0
        if any(w in t_low for w in ("junior", "jr", "jr.", "entry", "associate", "trainee", "intern")):
            return 1.0
        return 1.5  # Mid/default

    def extract(self, candidate: Candidate) -> dict:
        """Extract career progression features from Candidate."""
        history = candidate.career_history if candidate.career_history else []

        # Unique industries, companies, and company sizes
        industries = sorted(list({role.industry.strip() for role in history if role.industry}))
        companies = sorted(list({role.company.strip() for role in history if role.company}))
        company_sizes = sorted(list({role.company_size.strip() for role in history if role.company_size}))

        # Chronological sorting of positions
        parsed_roles = []
        for role in history:
            s_date = self._parse_date(role.start_date)
            parsed_roles.append(
                {
                    "company": role.company.strip() if role.company else "",
                    "title": role.title.strip() if role.title else "",
                    "start_date_obj": s_date or datetime.min.date(),
                    "duration_months": role.duration_months if role.duration_months is not None else 0,
                    "company_size": role.company_size.strip() if role.company_size else "",
                    "industry": role.industry.strip() if role.industry else "",
                }
            )

        # Sort from oldest to newest
        parsed_roles.sort(key=lambda x: x["start_date_obj"])

        title_progression = [role["title"] for role in parsed_roles]

        # Role transitions
        role_transitions = []
        for i in range(len(parsed_roles) - 1):
            role_transitions.append(
                {"from_title": parsed_roles[i]["title"], "to_title": parsed_roles[i + 1]["title"]}
            )

        # Promotion indicators
        # True if there's a title change to a higher seniority score at the same company,
        # or more roles at the same company where the later one is more senior.
        has_promotion = False
        company_roles = {}
        for role in parsed_roles:
            comp = role["company"].lower()
            if comp:
                if comp not in company_roles:
                    company_roles[comp] = []
                company_roles[comp].append(role)

        for comp, roles in company_roles.items():
            if len(roles) > 1:
                # Compare consecutive seniority scores at the same company
                for idx in range(len(roles) - 1):
                    prev_score = self._title_seniority_score(roles[idx]["title"])
                    next_score = self._title_seniority_score(roles[idx + 1]["title"])
                    if next_score > prev_score:
                        has_promotion = True
                        break

        # Career stability metrics
        total_months = sum(role["duration_months"] for role in parsed_roles)
        total_years = total_months / 12.0 if total_months > 0 else 0.0
        num_positions = len(parsed_roles)

        roles_per_year = round(num_positions / total_years, 2) if total_years > 0 else 0.0
        average_tenure_months = round(total_months / num_positions, 2) if num_positions > 0 else 0.0

        stability_metrics = {
            "average_tenure_months": average_tenure_months,
            "roles_per_year": roles_per_year,
            "job_hopping_index": round(num_positions / (total_years + 0.1), 2),  # roles per year offset
        }

        return {
            "industries_worked_in": industries,
            "companies_worked_for": companies,
            "company_size_history": company_sizes,
            "title_progression": title_progression,
            "role_transitions": role_transitions,
            "promotion_indicators": has_promotion,
            "career_stability_metrics": stability_metrics,
        }

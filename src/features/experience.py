from datetime import date, datetime
from src.models.candidate import Candidate
from src.features.base import FeatureExtractor


class ExperienceExtractor(FeatureExtractor):
    """Extracts structured experience facts and tenure statistics from a candidate."""

    def __init__(self, reference_date: date | None = None) -> None:
        """Initialize with an optional fixed reference date for deterministic gap calculation.

        Args:
            reference_date: Fixed date to use for current role end dates.
                If None, uses the candidate's last_active_date or datetime.now().
        """
        self._reference_date: date | None = reference_date

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

    def extract(self, candidate: Candidate) -> dict:
        """Extract experience features from Candidate."""
        profile_yoe = 0.0
        if candidate.profile and candidate.profile.years_of_experience is not None:
            profile_yoe = float(candidate.profile.years_of_experience)

        history = candidate.career_history if candidate.career_history else []
        total_positions = len(history)

        current_tenures = []
        all_tenures = []
        total_months_worked = 0

        for role in history:
            dur = role.duration_months if role.duration_months is not None else 0
            all_tenures.append(dur)
            total_months_worked += dur
            if role.is_current:
                current_tenures.append(dur)

        current_tenure = current_tenures[0] if current_tenures else 0
        longest_tenure = max(all_tenures) if all_tenures else 0
        average_tenure = round(sum(all_tenures) / len(all_tenures), 2) if all_tenures else 0.0

        # Calculate employment gaps
        # Sort positions by start_date
        parsed_roles = []
        reference_date = self._reference_date
        if reference_date is None:
            if candidate.redrob_signals and candidate.redrob_signals.last_active_date:
                reference_date = self._parse_date(candidate.redrob_signals.last_active_date)
        if reference_date is None:
            reference_date = datetime.now().date()

        for role in history:
            s_date = self._parse_date(role.start_date)
            e_date = self._parse_date(role.end_date)
            if not e_date and (role.is_current or not role.end_date):
                e_date = reference_date
            if s_date:
                parsed_roles.append((s_date, e_date or s_date))

        parsed_roles.sort(key=lambda x: x[0])

        gap_months = 0
        if len(parsed_roles) > 1:
            current_end = parsed_roles[0][1]
            for i in range(1, len(parsed_roles)):
                next_start, next_end = parsed_roles[i]
                if next_start > current_end:
                    # There is a gap
                    days_gap = (next_start - current_end).days
                    gap_months += round(days_gap / 30.44, 2)
                # Keep track of furthest end date (to handle overlapping roles)
                if next_end > current_end:
                    current_end = next_end

        return {
            "years_of_experience": profile_yoe,
            "total_positions": total_positions,
            "current_tenure": current_tenure,
            "average_tenure": average_tenure,
            "longest_tenure": longest_tenure,
            "employment_gaps_months": round(gap_months, 2),
            "total_months_worked": total_months_worked,
        }

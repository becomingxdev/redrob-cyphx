from src.models.candidate import Candidate
from src.models.career import CareerHistory
from src.models.certification import Certification
from src.models.education import Education
from src.models.language import Language
from src.models.profile import Profile
from src.models.redrob import RedrobSignals
from src.models.salary import SalaryRange
from src.models.skill import Skill


class CandidateFactory:
    """Creates Candidate objects from dictionaries."""

    @staticmethod
    def create(candidate_dict: dict) -> Candidate:
        profile_data = candidate_dict["profile"]
        profile = Profile(
            anonymized_name=profile_data["anonymized_name"],
            headline=profile_data["headline"],
            summary=profile_data["summary"],
            location=profile_data["location"],
            country=profile_data["country"],
            years_of_experience=profile_data["years_of_experience"],
            current_title=profile_data["current_title"],
            current_company=profile_data["current_company"],
            current_company_size=profile_data["current_company_size"],
            current_industry=profile_data["current_industry"],
        )

        career_history = [
            CareerHistory(
                company=item["company"],
                title=item["title"],
                start_date=item["start_date"],
                end_date=item.get("end_date"),
                duration_months=item["duration_months"],
                is_current=item["is_current"],
                industry=item["industry"],
                company_size=item["company_size"],
                description=item["description"],
            )
            for item in candidate_dict["career_history"]
        ]

        education = [
            Education(
                institution=item["institution"],
                degree=item["degree"],
                field_of_study=item["field_of_study"],
                start_year=item["start_year"],
                end_year=item["end_year"],
                grade=item.get("grade"),
                tier=item["tier"],
            )
            for item in candidate_dict["education"]
        ]

        skills = [
            Skill(
                name=item["name"],
                proficiency=item["proficiency"],
                endorsements=item["endorsements"],
                duration_months=item.get("duration_months"),
            )
            for item in candidate_dict["skills"]
        ]

        certifications = [
            Certification(
                name=item["name"],
                issuer=item["issuer"],
                year=item["year"],
            )
            for item in candidate_dict.get("certifications", [])
        ]

        languages = [
            Language(
                language=item["language"],
                proficiency=item["proficiency"],
            )
            for item in candidate_dict.get("languages", [])
        ]

        signals_data = candidate_dict["redrob_signals"]
        salary_data = signals_data["expected_salary_range_inr_lpa"]
        expected_salary = SalaryRange(
            min=salary_data["min"],
            max=salary_data["max"],
        )

        redrob_signals = RedrobSignals(
            profile_completeness_score=signals_data["profile_completeness_score"],
            signup_date=signals_data["signup_date"],
            last_active_date=signals_data["last_active_date"],
            open_to_work_flag=signals_data["open_to_work_flag"],
            profile_views_received_30d=signals_data["profile_views_received_30d"],
            applications_submitted_30d=signals_data["applications_submitted_30d"],
            recruiter_response_rate=signals_data["recruiter_response_rate"],
            avg_response_time_hours=signals_data["avg_response_time_hours"],
            skill_assessment_scores=signals_data["skill_assessment_scores"],
            connection_count=signals_data["connection_count"],
            endorsements_received=signals_data["endorsements_received"],
            notice_period_days=signals_data["notice_period_days"],
            expected_salary_range_inr_lpa=expected_salary,
            preferred_work_mode=signals_data["preferred_work_mode"],
            willing_to_relocate=signals_data["willing_to_relocate"],
            github_activity_score=signals_data["github_activity_score"],
            search_appearance_30d=signals_data["search_appearance_30d"],
            saved_by_recruiters_30d=signals_data["saved_by_recruiters_30d"],
            interview_completion_rate=signals_data["interview_completion_rate"],
            offer_acceptance_rate=signals_data["offer_acceptance_rate"],
            verified_email=signals_data["verified_email"],
            verified_phone=signals_data["verified_phone"],
            linkedin_connected=signals_data["linkedin_connected"],
        )

        return Candidate(
            candidate_id=candidate_dict["candidate_id"],
            profile=profile,
            career_history=career_history,
            education=education,
            skills=skills,
            certifications=certifications,
            languages=languages,
            redrob_signals=redrob_signals,
            raw=candidate_dict,
        )

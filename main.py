"""Application entry point for the REDROB AI Candidate Ranking Engine."""

import json
from pathlib import Path

from src.parser.loader import load_candidates
from src.utils.banner import print_banner
from src.features.title import TitleExtractor
from src.features.skills import SkillsExtractor
from src.features.experience import ExperienceExtractor
from src.features.education import EducationExtractor
from src.features.location import LocationExtractor
from src.features.career import CareerExtractor

DATASET_PATH = "data/candidates.jsonl"


def main() -> None:
    """Load, count, extract features for demonstration, and display summary."""

    print_banner()

    dataset = Path(DATASET_PATH)
    if not dataset.exists():
        print(f"\nDataset not found: {DATASET_PATH}")
        return

    print(f"\nLoading dataset: {DATASET_PATH} ...")

    count = 0
    first = None

    for candidate in load_candidates(DATASET_PATH):
        count += 1
        if count == 1:
            first = candidate

    if count == 0 or first is None:
        print("\nNo valid candidates were found.")
        return

    # Initialize all feature extractors
    title_ext = TitleExtractor()
    skills_ext = SkillsExtractor()
    exp_ext = ExperienceExtractor()
    edu_ext = EducationExtractor()
    loc_ext = LocationExtractor()
    career_ext = CareerExtractor()

    # Extract features for the first candidate
    title_features = title_ext.extract(first)
    skills_features = skills_ext.extract(first)
    exp_features = exp_ext.extract(first)
    edu_features = edu_ext.extract(first)
    loc_features = loc_ext.extract(first)
    career_features = career_ext.extract(first)

    # Print the Feature Summary for the first candidate
    print("\n" + "=" * 50)
    print("Feature Summary")
    print("=" * 50)
    print(f"Candidate ID: {first.candidate_id}")
    print(f"Candidate Name: {first.profile.anonymized_name}")

    print("\nTitle Features")
    print("-" * 30)
    print(json.dumps(title_features, indent=2))

    print("\nSkill Features")
    print("-" * 30)
    print(json.dumps(skills_features, indent=2))

    print("\nExperience Features")
    print("-" * 30)
    print(json.dumps(exp_features, indent=2))

    print("\nEducation Features")
    print("-" * 30)
    print(json.dumps(edu_features, indent=2))

    print("\nLocation Features")
    print("-" * 30)
    print(json.dumps(loc_features, indent=2))

    print("\nCareer Features")
    print("-" * 30)
    print(json.dumps(career_features, indent=2))

    print("\n" + "=" * 50)
    print(f"Successfully loaded: {count} candidates\n")


if __name__ == "__main__":
    main()

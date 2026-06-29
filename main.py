"""Application entry point for the REDROB AI Candidate Ranking Engine."""

import json
from pathlib import Path

from src.parser.loader import load_candidates
from src.utils.banner import print_banner

# Feature Extractors
from src.features.title import TitleExtractor
from src.features.skills import SkillsExtractor
from src.features.experience import ExperienceExtractor
from src.features.education import EducationExtractor
from src.features.location import LocationExtractor
from src.features.career import CareerExtractor

# Evidence Layer
from src.evidence.verifier import EvidenceVerifier
from src.evidence.consistency import ConsistencyAnalyzer
from src.evidence.confidence import ConfidenceCalculator

# Scoring Engines
from src.scoring.title_score import score_title
from src.scoring.skill_score import score_skills
from src.scoring.experience_score import score_experience
from src.scoring.education_score import score_education
from src.scoring.behavior_score import score_behavior
from src.scoring.penalties import apply_penalties
from src.scoring.honeypot import detect_honeypot
from src.scoring.composite import compose_score

DATASET_PATH = "data/candidates.jsonl"


def main() -> None:
    """Load dataset and run the complete pipeline up to composite scoring."""

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
        if first is None:
            first = candidate

    if first is None:
        print("\nNo valid candidates were found.")
        return

    # -------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------

    feature_extractors = {
        "title": TitleExtractor(),
        "skills": SkillsExtractor(),
        "experience": ExperienceExtractor(),
        "education": EducationExtractor(),
        "location": LocationExtractor(),
        "career": CareerExtractor(),
    }

    extracted_features = {
        name: extractor.extract(first)
        for name, extractor in feature_extractors.items()
    }

    # -------------------------------------------------------------
    # Evidence Layer
    # -------------------------------------------------------------

    verifier = EvidenceVerifier()
    consistency = ConsistencyAnalyzer()
    confidence = ConfidenceCalculator()

    evidence_result = verifier.verify(first)
    consistency_result = consistency.analyze(first)
    confidence_result = confidence.calculate(
        candidate=first,
        evidence=evidence_result,
        consistency=consistency_result
    )

    # -------------------------------------------------------------
    # Individual Scoring
    # -------------------------------------------------------------

    title_score = score_title(first, extracted_features["title"])
    skill_score = score_skills(first, extracted_features["skills"])
    experience_score = score_experience(
        first,
        extracted_features["experience"],
        extracted_features["career"],
    )
    education_score = score_education(
        first,
        extracted_features["education"],
        candidate_skills=set(extracted_features["skills"].get("skills", [])),
    )
    behavior_score = score_behavior(first)
    penalty_result = apply_penalties(
        candidate=first,
        title_features=extracted_features["title"],
        experience_features=extracted_features["experience"],
        education_features=extracted_features["education"],
        career_features=extracted_features["career"],
        evidence_result=evidence_result,
        consistency_result=consistency_result,
    )
    honeypot_result = detect_honeypot(
        candidate=first,
        experience_features=extracted_features["experience"],
        education_features=extracted_features["education"],
        title_features=extracted_features["title"],
        evidence_result=evidence_result,
        consistency_result=consistency_result,
    )

    # -------------------------------------------------------------
    # Composite Score
    # -------------------------------------------------------------

    final = compose_score(
        candidate_id=first.candidate_id,
        title_score=title_score,
        skill_score=skill_score,
        experience_score=experience_score,
        education_score=education_score,
        behavior_score=behavior_score,
        penalty=penalty_result,
        confidence_result=confidence_result,
        consistency_result=consistency_result,
        honeypot=honeypot_result,
    )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Candidate Summary")
    print("=" * 60)

    print(f"Candidate ID   : {first.candidate_id}")
    print(f"Candidate Name : {first.profile.anonymized_name}")

    print("\n" + "=" * 60)
    print("Feature Extraction")
    print("=" * 60)

    for name, result in extracted_features.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("Evidence Verification")
    print("=" * 60)
    print(json.dumps(evidence_result, indent=2))

    print("\n" + "=" * 60)
    print("Consistency Analysis")
    print("=" * 60)
    print(json.dumps(consistency_result, indent=2))

    print("\n" + "=" * 60)
    print("Confidence Analysis")
    print("=" * 60)
    print(json.dumps(confidence_result, indent=2))

    print("\n" + "=" * 60)
    print("Individual Scores")
    print("=" * 60)

    for engine_name, engine_result in [
        ("Title", title_score),
        ("Skills", skill_score),
        ("Experience", experience_score),
        ("Education", education_score),
        ("Behavior", behavior_score),
    ]:
        print(f"\n{engine_name} Score: {engine_result.score:.2f}")
        for reason in engine_result.reasons:
            print(f"  - {reason}")

    print("\n" + "=" * 60)
    print("Penalty Assessment")
    print("=" * 60)
    print(f"Total Penalty: {penalty_result.penalty_score:.2f}")
    for reason in penalty_result.reasons:
        print(f"  - {reason}")

    print("\n" + "=" * 60)
    print("Honeypot Detection")
    print("=" * 60)
    print(f"Suspicion Score: {honeypot_result.suspicion_score:.2f}")
    for reason in honeypot_result.reasons:
        print(f"  - {reason}")

    print("\n" + "=" * 60)
    print("Composite Score")
    print("=" * 60)
    print(f"Final Score: {final.score:.2f}")
    for reason in final.reasons:
        print(f"  - {reason}")

    print("\n" + "=" * 60)
    print(f"Successfully loaded {count:,} candidates.")
    print("=" * 60)


if __name__ == "__main__":
    main()

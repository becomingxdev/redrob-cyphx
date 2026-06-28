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

DATASET_PATH = "data/candidates.jsonl"


def main() -> None:
    """Load dataset and demonstrate the complete pipeline up to the Evidence Layer."""

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

    # Call the Evidence Layer modules using their correct signatures
    evidence_result = verifier.verify(first)
    consistency_result = consistency.analyze(first)
    confidence_result = confidence.calculate(
        candidate=first,
        evidence=evidence_result,
        consistency=consistency_result
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
    print(f"Successfully loaded {count:,} candidates.")
    print("=" * 60)


if __name__ == "__main__":
    main()

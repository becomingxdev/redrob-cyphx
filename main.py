from src.utils.banner import print_banner
from src.parser.loader import CandidateLoader
from src.parser.validator import CandidateValidator
from src.models.factory import CandidateFactory


def main() -> None:
    """Application entry point."""

    # Display application banner
    print_banner()

    # Load candidate data
    candidates = CandidateLoader.load("data/sample_candidates.json")
    print(f"\nLoaded {len(candidates)} candidates.")

    # Validate candidates
    candidate_objects = []
    for candidate in candidates:
        if CandidateValidator.validate(candidate):
            candidate_objects.append(
                CandidateFactory.create(candidate)
            )
    
    print(f"Candidate objects: {len(candidate_objects)}")
    print("\nFirst candidate:\n")
    print(candidate_objects[0])

if __name__ == "__main__":
    main()
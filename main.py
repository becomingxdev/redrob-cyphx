from src.utils.banner import print_banner
from src.parser.loader import CandidateLoader
from src.parser.validator import CandidateValidator


def main() -> None:
    """Application entry point."""

    # Display application banner
    print_banner()

    # Load candidate data
    candidates = CandidateLoader.load("data/sample.jsonl")

    print(f"\nLoaded {len(candidates)} candidates.")

    # Validate candidates
    valid_candidates = []

    for candidate in candidates:
        if CandidateValidator.validate(candidate):
            valid_candidates.append(candidate)

    print(f"Valid candidates: {len(valid_candidates)}")

    # Print valid candidates (temporary, for testing)
    print("\nCandidates:")
    for candidate in valid_candidates:
        print(candidate)


if __name__ == "__main__":
    main()
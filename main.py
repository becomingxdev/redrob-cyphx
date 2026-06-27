"""Application entry point for the REDROB AI Candidate Ranking Engine."""

from pathlib import Path

from src.parser.loader import load_candidates
from src.utils.banner import print_banner

DATASET_PATH = "data/candidates.jsonl"


def main() -> None:
    """Load, count, and display a summary of parsed candidates."""

    print_banner()

    dataset = Path(DATASET_PATH)
    if not dataset.exists():
        print(f"\nDataset not found: {DATASET_PATH}")
        return

    print(f"\nLoading dataset: {DATASET_PATH} ...")

    count = 0
    first: object | None = None

    for candidate in load_candidates(DATASET_PATH):
        count += 1
        if count == 1:
            first = candidate

    if count == 0:
        print("\nNo valid candidates were found.")
        return

    print("\nFirst Candidate\n")
    print(first)
    print("\n" + "-" * 40)
    print(f"\nSuccessfully loaded: {count} candidates\n")


if __name__ == "__main__":
    main()

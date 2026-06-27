from src.utils.banner import print_banner
from src.parser.loader import CandidateLoader


def main():
    print_banner()
    candidates = CandidateLoader.load("data/sample.jsonl")
    print(f"\nLoaded {len(candidates)} candidates.")
    print(candidates)


if __name__ == "__main__":
    main()
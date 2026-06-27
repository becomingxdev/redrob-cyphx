"""
Explore the Redrob candidate dataset.

This script provides a quick overview of the dataset before
building the ranking pipeline.

Usage:
    python scripts/explore_dataset.py data/sample_candidates.json
"""

from pathlib import Path
import json
import sys
from collections import Counter


def load_candidates(path: Path):
    """
    Load either JSON or JSONL.
    """

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                # Sometimes datasets wrap candidates
                if "candidates" in data:
                    return data["candidates"]

                return [data]

    candidates = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            candidates.append(json.loads(line))

    return candidates


def print_basic_stats(candidates):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Total Candidates : {len(candidates)}")

    field_counter = Counter()

    for candidate in candidates:
        field_counter.update(candidate.keys())

    print("\nMost Common Fields\n")

    for field, count in field_counter.most_common():
        print(f"{field:25} {count}")


def analyze_titles(candidates):
    titles = Counter()

    for candidate in candidates:
        title = (
            candidate.get("headline")
            or candidate.get("title")
            or candidate.get("current_title")
            or ""
        )

        title = title.strip()

        if title:
            titles[title] += 1

    print("\n" + "=" * 60)
    print("TOP JOB TITLES")
    print("=" * 60)

    for title, count in titles.most_common(20):
        print(f"{count:4}  {title}")


def analyze_skills(candidates):
    skills = Counter()

    for candidate in candidates:

        candidate_skills = candidate.get("skills", [])

        if isinstance(candidate_skills, str):
            candidate_skills = [candidate_skills]

        for skill in candidate_skills:
            skills[str(skill).strip()] += 1

    print("\n" + "=" * 60)
    print("TOP SKILLS")
    print("=" * 60)

    for skill, count in skills.most_common(30):
        print(f"{count:4}  {skill}")


def show_sample_candidate(candidates):
    print("\n" + "=" * 60)
    print("FIRST CANDIDATE")
    print("=" * 60)

    print(json.dumps(candidates[0], indent=2))


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python scripts/explore_dataset.py <dataset>")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    candidates = load_candidates(dataset_path)

    print_basic_stats(candidates)
    analyze_titles(candidates)
    analyze_skills(candidates)
    show_sample_candidate(candidates)


if __name__ == "__main__":
    main()
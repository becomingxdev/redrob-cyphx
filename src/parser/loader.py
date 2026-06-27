import json
from pathlib import Path


class CandidateLoader:
    """Loads candidate data from JSON or JSONL files."""

    @staticmethod
    def load(file_path: str) -> list[dict]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of candidates.")

            return data

        if path.suffix == ".jsonl":
            candidates = []

            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    if line:
                        candidates.append(json.loads(line))

            return candidates

        raise ValueError(f"Unsupported file format: {path.suffix}")
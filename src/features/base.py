import os
from abc import ABC, abstractmethod
from src.models.candidate import Candidate

# Try to import yaml, fallback to simple parser if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_rules_yaml(path: str) -> dict:
    """Load configuration from a YAML file. Uses PyYAML if available, falls back to a simple parser."""
    if not os.path.exists(path):
        return {}

    if HAS_YAML:
        with open(path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f) or {}
            except Exception:
                pass

    # Simple fallback parser for our YAML structure
    data = {}
    current_key = None
    current_list = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Strip comments
            line = line.split("#")[0].rstrip()
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            if stripped.startswith("- "):
                val = stripped[2:].strip().strip('"').strip("'")
                if current_list is not None:
                    current_list.append(val)
                continue

            if ":" in stripped:
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.strip()

                if val.startswith("[") and val.endswith("]"):
                    # Inline list: ["a", "b"]
                    items = [x.strip().strip('"').strip("'") for x in val[1:-1].split(",") if x.strip()]
                    if indent == 0:
                        data[key] = items
                        current_key = key
                        current_list = None
                    else:
                        if current_key and isinstance(data.get(current_key), dict):
                            data[current_key][key] = items
                        else:
                            data[key] = items
                elif not val:
                    # Nested block
                    if indent == 0:
                        data[key] = {}
                        current_key = key
                        current_list = None
                    else:
                        if current_key and isinstance(data.get(current_key), dict):
                            data[current_key][key] = []
                            current_list = data[current_key][key]
                        else:
                            data[key] = []
                            current_list = data[key]
                else:
                    # Key-value pair
                    val = val.strip('"').strip("'")
                    if val.lower() == "true":
                        val = True
                    elif val.lower() == "false":
                        val = False
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                    if indent == 0:
                        data[key] = val
                        current_key = key
                        current_list = None
                    else:
                        if current_key and isinstance(data.get(current_key), dict):
                            data[current_key][key] = val
                        else:
                            data[key] = val
    return data


class FeatureExtractor(ABC):
    """Abstract base class for all candidate feature extractors."""

    @abstractmethod
    def extract(self, candidate: Candidate) -> dict:
        """Extract structured facts/features from a Candidate.

        Never computes scores, ranks, or penalties.
        """
        pass

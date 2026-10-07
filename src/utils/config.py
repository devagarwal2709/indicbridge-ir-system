import os
import re
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "retrieval": {
        "top_k": 10,
        "min_similarity": 0.01,
        "alpha": 0.70,   # Cosine similarity weight
        "beta": 0.15,    # Transliteration score weight
        "gamma": 0.10,   # Phonetic score weight
        "delta": 0.05,   # Exact token match weight
    },
    "index": {
        "use_sublinear_tf": True,
        "min_df": 1,
        "max_df_ratio": 0.95,
        "ngram_range": [1, 1],
    },
    "preprocessing": {
        "normalize_unicode": True,
        "fold_nukta": True,
        "strip_punctuation": True,
        "lowercase_latin": True,
        "use_stopwords": True,
        "use_stemming": True,
        "enable_transliteration": True,
        "enable_romanized_normalization": True,
        "enable_phonetic": True,
        "phonetic_similarity_threshold": 0.65,
    },
    "paths": {
        "documents": "data/documents.jsonl",
        "evaluation_queries": "data/evaluation_queries.json",
        "index_dir": "data/index_cache",
        "evaluation_results": "evaluation/results",
    },
}


def _simple_yaml_parse(content: str) -> Dict[str, Any]:
    """Lightweight pure-python parser for standard nested YAML without external dependencies."""
    result: Dict[str, Any] = {}
    current_section = None

    for line in content.splitlines():
        line = line.split("#")[0].rstrip()
        if not line or line.strip() == "":
            continue

        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1].strip()
            result[current_section] = {}
        elif indent > 0 and ":" in stripped and current_section:
            k, v = stripped.split(":", 1)
            k = k.strip()
            v = v.strip()

            # Parse boolean
            if v.lower() == "true":
                parsed_val = True
            elif v.lower() == "false":
                parsed_val = False
            # Parse list [x, y]
            elif v.startswith("[") and v.endswith("]"):
                items = [item.strip() for item in v[1:-1].split(",") if item.strip()]
                parsed_val = [int(i) if i.isdigit() else i for i in items]
            # Parse float / int
            elif re.match(r"^-?\d+\.\d+$", v):
                parsed_val = float(v)
            elif re.match(r"^-?\d+$", v):
                parsed_val = int(v)
            # Parse string
            elif (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                parsed_val = v[1:-1]
            else:
                parsed_val = v

            result[current_section][k] = parsed_val

    return result


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with fallback to built-in parser and defaults."""
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG.copy()

    try:
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
        except ImportError:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = _simple_yaml_parse(f.read())

        # Deep merge with defaults
        merged = DEFAULT_CONFIG.copy()
        for section, values in user_config.items():
            if isinstance(values, dict) and section in merged:
                merged[section] = {**merged[section], **values}
            else:
                merged[section] = values
        return merged
    except Exception as e:
        print(f"[Warning] Failed to load config from {config_path}: {e}. Using defaults.")
        return DEFAULT_CONFIG.copy()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration dict."""
    return DEFAULT_CONFIG.copy()

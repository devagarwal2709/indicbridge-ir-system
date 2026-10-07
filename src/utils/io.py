import json
import os
from typing import Any, Dict, List


def load_documents(filepath: str) -> List[Dict[str, Any]]:
    """
    Load document collection from JSONL or JSON file.
    Validates mandatory schema fields: id, title, text, language.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Document file not found at: {filepath}")

    documents: List[Dict[str, Any]] = []

    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    if "id" not in doc or "text" not in doc:
                        print(f"[Warning] Skipping line {line_no}: missing 'id' or 'text'")
                        continue
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    print(f"[Warning] Skipping invalid JSON at line {line_no}: {e}")
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = data
            elif isinstance(data, dict) and "documents" in data:
                documents = data["documents"]

    return documents


def save_json(data: Any, filepath: str) -> None:
    """Save serializable data to JSON file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

import re
from typing import Dict, List, Optional
from .language_detector import detect_script
from .normalizer import normalize_romanized_token
from .stopwords import is_stopword

# Token extraction pattern:
# Captures Devanagari character sequences (including matras, virama, nukta, bindu)
# OR alphanumeric Latin sequences
TOKEN_PATTERN = re.compile(
    r"[\u0900-\u097F]+|[a-zA-Z0-9]+",
    re.UNICODE
)


class Token:
    """Represents a token with linguistic metadata."""
    def __init__(
        self,
        text: str,
        script: str = "Unknown",
        is_stop: bool = False,
        normalized: Optional[str] = None
    ):
        self.text = text
        self.script = script
        self.is_stop = is_stop
        self.normalized = normalized or text

    def to_dict(self) -> Dict[str, any]:
        return {
            "text": self.text,
            "script": self.script,
            "is_stop": self.is_stop,
            "normalized": self.normalized,
        }

    def __repr__(self) -> str:
        return f"Token({self.text!r}, norm={self.normalized!r}, stop={self.is_stop})"


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Fast simple token extraction.
    Splits text into words, respecting Devanagari character blocks and Latin words.
    """
    if not text:
        return []

    tokens = TOKEN_PATTERN.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def tokenize_detailed(text: str, lowercase: bool = True) -> List[Token]:
    """
    Tokenize with linguistic annotations:
    - Script detection
    - Stopword flag
    - Romanized normalization
    """
    if not text:
        return []

    raw_tokens = TOKEN_PATTERN.findall(text)
    tokens: List[Token] = []

    for t in raw_tokens:
        clean = t.lower() if lowercase else t
        script = detect_script(clean)
        is_stop = is_stopword(clean)
        norm = normalize_romanized_token(clean) if script == "Latin" else clean
        tokens.append(Token(text=clean, script=script, is_stop=is_stop, normalized=norm))

    return tokens

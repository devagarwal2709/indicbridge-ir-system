import re
from typing import List

# Hindi suffixes ordered by descending length (Ramanathan & Rao algorithm)
HINDI_SUFFIXES = [
    "ाओं", "ुओं", "ुएं", "ियों", "ीयों", "ियां", "ाएं", "िए", "ीं",
    "ाओ", "कर", "ना", "ने", "नी", "ता", "ती", "ते",
    "यों", "ों", "ा", "े", "ी", "ो", "ू", "ु"
]

# English inflectional suffixes ordered by length
ENGLISH_SUFFIX_RULES = [
    (r"sses$", "ss"),
    (r"ies$", "y"),
    (r"ss$", "ss"),
    (r"s$", ""),
    (r"eed$", "ee"),
    (r"ing$", ""),
    (r"ed$", ""),
    (r"ly$", ""),
    (r"tion$", "te"),
    (r"ment$", ""),
]

# Hinglish plural / verbal suffixes
HINGLISH_SUFFIXES = [
    "on", "aan", "iyaan", "iye", "kar", "enge", "aenge"
]


def stem_hindi(word: str) -> str:
    """
    Lightweight rule-based Hindi stemmer (suffix removal).
    Preserves root words with minimum stem length threshold.
    """
    if len(word) <= 3:
        return word

    for suffix in HINDI_SUFFIXES:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if len(stem) >= 2:
                return stem
    return word


def stem_english(word: str) -> str:
    """Lightweight rule-based English stemmer."""
    w = word.lower().strip()
    if len(w) <= 3:
        return w

    for pattern, replacement in ENGLISH_SUFFIX_RULES:
        if re.search(pattern, w):
            stem = re.sub(pattern, replacement, w)
            if len(stem) >= 2:
                return stem
    return w


def stem_hinglish(word: str) -> str:
    """Stem common Romanized Hindi inflections."""
    w = word.lower().strip()
    if len(w) <= 4:
        return w

    for suffix in HINGLISH_SUFFIXES:
        if w.endswith(suffix):
            stem = w[:-len(suffix)]
            if len(stem) >= 3:
                return stem
    return w


def stem_word(word: str, script: str = "auto") -> str:
    """
    Script-aware stemming.
    Detects script if set to 'auto'.
    """
    if not word or len(word) <= 2:
        return word

    # Check for Devanagari characters
    if re.search(r"[\u0900-\u097F]", word):
        return stem_hindi(word)

    # Latin script: check Hinglish vs English
    return stem_english(stem_hinglish(word))


def stem_tokens(tokens: List[str]) -> List[str]:
    """Stem a list of tokens."""
    return [stem_word(t) for t in tokens if t]

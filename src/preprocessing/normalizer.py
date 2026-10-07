import re
import unicodedata
from typing import Dict, List, Tuple

# Nukta mapping in Devanagari (Unicode folding)
NUKTA_MAP: Dict[str, str] = {
    "क़": "क",
    "ख़": "ख",
    "ग़": "ग",
    "ज़": "ज",
    "ड़": "ड",
    "ढ़": "ढ",
    "फ़": "फ",
    "य़": "य",
    "\u093C": "",  # nukta mark itself
}

# Conservative dictionary for well-known Romanized Hindi spelling variants
ROMANIZED_CANONICAL_MAP: Dict[str, str] = {
    # Variations of Bharat / India
    "bhaarat": "bharat",
    "bhaaratvarsh": "bharatvarsh",
    "bhartiya": "bharatiya",
    "bhaaratiya": "bharatiya",
    "des": "desh",

    # Weather terms
    "mosam": "mausam",
    "mausm": "mausam",
    "mousam": "mausam",
    "kall": "kal",

    # Education / Learning
    "siksha": "shiksha",
    "seeksha": "shiksha",
    "vidyalay": "vidyalaya",
    "vidyala": "vidyalaya",
    "vidhyalaya": "vidyalaya",

    # Krishna / deities
    "krishan": "krishna",
    "krisna": "krishna",
    "shree": "shri",
    "sri": "shri",
    "bhagwaan": "bhagwan",

    # Geography / places
    "himalay": "himalaya",
    "himaalaya": "himalaya",
    "dilli": "delhi",
    "dehli": "delhi",

    # Space & Science
    "chandraayan": "chandrayaan",
    "chandrayan": "chandrayaan",
    "vigyan": "vigyan",
    "vigyaan": "vigyan",

    # History & Culture
    "itihaas": "itihas",
    "etihas": "itihas",
    "tyohar": "tyohar",
    "tyouhar": "tyohar",
    "teohar": "tyohar",

    # Agriculture
    "kisaan": "kisan",
    "krishi": "krishi",
    "khetee": "kheti",

    # Common function words
    "bohot": "bahut",
    "bhot": "bahut",
    "bahoot": "bahut",
    "zaroori": "zaruri",
    "jaroori": "zaruri",
    "jaruri": "zaruri",
    "aaj": "aaj",
    "rahe": "rahe",
    "rahi": "rahi",
    "raha": "raha",
}

# Punctuation pattern that keeps spaces and Indic danda (।) as space
PUNCTUATION_RE = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """Normalize text into canonical NFC form."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def fold_nukta(text: str) -> str:
    """Fold Devanagari nukta characters into their base consonant."""
    for accented, base in NUKTA_MAP.items():
        text = text.replace(accented, base)
    return text


def strip_punctuation(text: str) -> str:
    """Remove punctuation and replace with a single space."""
    return PUNCTUATION_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse contiguous whitespace into a single space and trim."""
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_romanized_token(token: str) -> str:
    """
    Conservative Romanized Hindi token normalization.
    Preserves common roots while mapping spelling variants.
    """
    token = token.lower().strip()
    if not token:
        return ""

    # Direct canonical dictionary mapping
    if token in ROMANIZED_CANONICAL_MAP:
        return ROMANIZED_CANONICAL_MAP[token]

    # Conservative regularizations for romanized Hindi:
    # 1. Double vowels 'aa' -> 'a' (except in short words like 'aa')
    canonical = token
    if len(canonical) > 3:
        canonical = re.sub(r"aa+", "a", canonical)
        canonical = re.sub(r"ee+", "i", canonical)
        canonical = re.sub(r"oo+", "u", canonical)

    # 2. Check if normalized form matches canonical dictionary
    if canonical in ROMANIZED_CANONICAL_MAP:
        return ROMANIZED_CANONICAL_MAP[canonical]

    return canonical


def normalize_text(
    text: str,
    fold_devanagari_nukta: bool = True,
    lowercase_latin: bool = True,
    normalize_romanized: bool = True,
) -> str:
    """
    Full text normalization pipeline:
    1. Unicode NFC normalization
    2. Optional Nukta folding for Devanagari
    3. Optional lowercasing for Latin
    4. Punctuation removal
    5. Whitespace collapsing
    6. Optional Romanized token regularization
    """
    if not text:
        return ""

    normalized = normalize_unicode(text)

    if fold_devanagari_nukta:
        normalized = fold_nukta(normalized)

    if lowercase_latin:
        normalized = normalized.lower()

    normalized = strip_punctuation(normalized)
    normalized = normalize_whitespace(normalized)

    if normalize_romanized:
        tokens = normalized.split()
        normalized_tokens = [normalize_romanized_token(t) for t in tokens]
        normalized = " ".join(normalized_tokens)

    return normalized

import re
from typing import Dict, Tuple

# Unicode codepoints:
# Devanagari: U+0900 to U+097F
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[a-zA-Z]")

# Lexical markers strongly indicative of Romanized Hindi / Hinglish
ROMANIZED_HINDI_MARKERS = {
    "mera", "meri", "mere", "hamara", "hamari", "hamare",
    "ka", "ke", "ki", "ko", "se", "par", "mein", "men", "ne",
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "honge",
    "kya", "kyun", "kyon", "kaise", "kaisa", "kaisi", "kab", "kahan",
    "aur", "ya", "lekin", "magar", "bhi", "hi", "toh", "to",
    "desh", "bharat", "bhaarat", "mausam", "samay", "aaj", "kal", "parson",
    "shiksha", "siksha", "vidyalaya", "itihaas", "virasat", "tyohar",
    "kisan", "krishi", "nadi", "parvat", "mandir", "shahar", "gaaon",
    "karna", "karo", "kare", "karta", "karti", "karte", "raha", "rahi", "rahe"
}

# Common English stopwords/function words
ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "this", "that", "these", "those",
    "what", "when", "where", "which", "who", "whom", "why", "how",
    "weather", "forecast", "tomorrow", "yesterday", "today",
    "constitution", "rights", "republic", "government", "policy",
    "education", "school", "university", "science", "mission", "satellite"
}


def detect_script(text: str) -> str:
    """
    Detect the script of the text:
    - 'Devanagari' if primarily Devanagari
    - 'Latin' if primarily Latin
    - 'Mixed' if containing both scripts
    - 'Unknown' if only numbers/symbols
    """
    if not text:
        return "Unknown"

    has_devanagari = bool(DEVANAGARI_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))

    if has_devanagari and has_latin:
        return "Mixed"
    elif has_devanagari:
        return "Devanagari"
    elif has_latin:
        return "Latin"
    return "Unknown"


def detect_language(text: str) -> Tuple[str, str, float]:
    """
    Detect language and script of text.
    Returns: (language_code, script_name, confidence)
    - 'hi' (Hindi in Devanagari)
    - 'en' (English in Latin)
    - 'hi-Latn' (Romanized Hindi / Hinglish)
    - 'mixed' (Code-mixed or mixed script)
    """
    if not text or not text.strip():
        return "unknown", "Unknown", 0.0

    script = detect_script(text)

    if script == "Devanagari":
        return "hi", "Devanagari", 0.99

    if script == "Mixed":
        return "mixed", "Mixed", 0.85

    # Latin script: could be English, Romanized Hindi, or Hinglish
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return "unknown", "Latin", 0.0

    hindi_count = sum(1 for t in tokens if t in ROMANIZED_HINDI_MARKERS)
    english_count = sum(1 for t in tokens if t in ENGLISH_MARKERS)
    total = len(tokens)

    if hindi_count > 0 and english_count > 0:
        return "hi-Latn", "Latin", 0.85  # Hinglish code-mixed
    elif hindi_count > 0:
        conf = min(0.95, 0.5 + (hindi_count / max(1, total)) * 0.5)
        return "hi-Latn", "Latin", conf
    elif english_count > 0:
        conf = min(0.95, 0.5 + (english_count / max(1, total)) * 0.5)
        return "en", "Latin", conf
    else:
        # Default Latin assumption for ambiguous words
        return "en", "Latin", 0.50

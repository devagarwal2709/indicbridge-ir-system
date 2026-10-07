import re
from typing import Dict, List, Optional

# Try importing indic_transliteration, provide robust internal fallback if missing or unexpected
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate as indic_transliterate
    HAS_INDIC_TRANSLITERATION = True
except ImportError:
    HAS_INDIC_TRANSLITERATION = False

# High-precision domain transliteration lexicon for common cross-script search terms
LEXICON_ROMAN_TO_DEVANAGARI: Dict[str, str] = {
    "bharat": "भारत",
    "bhaarat": "भारत",
    "bharatiya": "भारतीय",
    "bhartiya": "भारतीय",
    "desh": "देश",
    "mausam": "मौसम",
    "kal": "कल",
    "aaj": "आज",
    "samay": "समय",
    "shiksha": "शिक्षा",
    "siksha": "शिक्षा",
    "vidyalaya": "विद्यालय",
    "krishna": "कृष्ण",
    "krishan": "कृष्ण",
    "shri": "श्री",
    "bhagwan": "भगवान",
    "himalaya": "हिमालय",
    "himalay": "हिमालय",
    "parvat": "पर्वत",
    "chandrayaan": "चंद्रयान",
    "chandrayan": "चंद्रयान",
    "isro": "इसरो",
    "delhi": "दिल्ली",
    "dilli": "दिल्ली",
    "rajdhani": "राजधानी",
    "itihas": "इतिहास",
    "itihaas": "इतिहास",
    "tyohar": "त्योहार",
    "kisan": "किसान",
    "krishi": "कृषि",
    "nadi": "नदी",
    "kheti": "खेती",
    "samvidhan": "संविधान",
    "adhikar": "अधिकार",
    "vigyan": "विज्ञान",
    "vikas": "विकास",
    "antariksh": "अंतरिक्ष",
    "karmayoga": "कर्मयोग",
    "diwali": "दिवाली",
    "deepawali": "दीपावली",
    "holi": "होली",
    "yoga": "योग",
}

LEXICON_ROMAN_TO_DEVANAGARI: Dict[str, str] = {
    # Functional markers & postpositions (Prevents matra drop)
    "ka": "का",
    "ke": "के",
    "ki": "की",
    "ko": "को",
    "se": "से",
    "par": "पर",
    "mein": "में",
    "men": "में",
    "hai": "है",
    "hain": "हैं",
    "aur": "और",
    # Domain search vocabulary
    "bharat": "भारत",
    "bhaarat": "भारत",
    "bharatiya": "भारतीय",
    "bhartiya": "भारतीय",
    "desh": "देश",
    "mausam": "मौसम",
    "kal": "कल",
    "aaj": "आज",
    "samay": "समय",
    "shiksha": "शिक्षा",
    "siksha": "शिक्षा",
    "vidyalaya": "विद्यालय",
    "krishna": "कृष्ण",
    "krishan": "कृष्ण",
    "shri": "श्री",
    "bhagwan": "भगवान",
    "himalaya": "हिमालय",
    "himalay": "हिमालय",
    "parvat": "पर्वत",
    "chandrayaan": "चंद्रयान",
    "chandrayan": "चंद्रयान",
    "isro": "इसरो",
    "delhi": "दिल्ली",
    "dilli": "दिल्ली",
    "rajdhani": "राजधानी",
    "itihas": "इतिहास",
    "itihaas": "इतिहास",
    "tyohar": "त्योहार",
    "kisan": "किसान",
    "krishi": "कृषि",
    "nadi": "नदी",
    "kheti": "खेती",
    "samvidhan": "संविधान",
    "adhikar": "अधिकार",
    "vigyan": "विज्ञान",
    "vikas": "विकास",
    "antariksh": "अंतरिक्ष",
    "karmayoga": "कर्मयोग",
    "diwali": "दिवाली",
    "deepawali": "दीपावली",
    "holi": "होली",
    "yoga": "योग",
}

LEXICON_DEVANAGARI_TO_ROMAN: Dict[str, str] = {
    v: k for k, v in LEXICON_ROMAN_TO_DEVANAGARI.items()
}
# Add canonical forms for reverse
LEXICON_DEVANAGARI_TO_ROMAN.update({
    "भारत": "bharat",
    "मौसम": "mausam",
    "देश": "desh",
    "कल": "kal",
    "शिक्षा": "shiksha",
    "विद्यालय": "vidyalaya",
    "कृष्ण": "krishna",
    "हिमालय": "himalaya",
    "दिल्ली": "delhi",
    "चंद्रयान": "chandrayaan",
    "संविधान": "samvidhan",
})

# Phonetic character mapping for rule-based fallback
CONSONANTS_MAP = {
    "kh": "ख", "gh": "घ", "ch": "च", "chh": "छ", "jh": "झ",
    "th": "थ", "dh": "ध", "ph": "फ", "bh": "भ", "sh": "श",
    "k": "क", "g": "ग", "c": "क", "j": "ज", "t": "त", "d": "द",
    "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र",
    "l": "ल", "v": "व", "w": "व", "s": "स", "h": "ह", "z": "ज़", "f": "फ़"
}

VOWELS_INITIAL = {
    "aa": "आ", "a": "अ", "ee": "ई", "i": "इ", "oo": "ऊ", "u": "उ",
    "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ"
}

VOWEL_MATRAS = {
    "aa": "ा", "a": "", "ee": "ी", "i": "ि", "oo": "ू", "u": "ु",
    "e": "े", "ai": "ै", "o": "ो", "au": "ौ"
}


def rule_based_roman_to_devanagari(word: str) -> str:
    """
    Rule-based phonetic converter for Romanized Hindi words.
    Provides robust transliteration for unseen terms.
    """
    w = word.lower().strip()
    if not w:
        return ""

    if w in LEXICON_ROMAN_TO_DEVANAGARI:
        return LEXICON_ROMAN_TO_DEVANAGARI[w]

    # Transliterate using standard phonetic rules
    result = []
    i = 0
    n = len(w)

    while i < n:
        # Check two-character consonants
        two = w[i:i+2]
        one = w[i:i+1]

        matched_cons = None
        advance = 1

        if two in CONSONANTS_MAP:
            matched_cons = CONSONANTS_MAP[two]
            advance = 2
        elif one in CONSONANTS_MAP:
            matched_cons = CONSONANTS_MAP[one]
            advance = 1

        if matched_cons:
            result.append(matched_cons)
            i += advance
            # Look ahead for vowel
            if i < n:
                next_two = w[i:i+2]
                next_one = w[i:i+1]
                if next_two in VOWEL_MATRAS:
                    result.append(VOWEL_MATRAS[next_two])
                    i += 2
                elif next_one in VOWEL_MATRAS:
                    result.append(VOWEL_MATRAS[next_one])
                    i += 1
                elif i < n and w[i] in CONSONANTS_MAP:
                    # Consonant cluster halant
                    result.append("्")
            continue

        # Initial or independent vowels
        if i == 0 or not result:
            if two in VOWELS_INITIAL:
                result.append(VOWELS_INITIAL[two])
                i += 2
                continue
            elif one in VOWELS_INITIAL:
                result.append(VOWELS_INITIAL[one])
                i += 1
                continue

        # Default fallback
        result.append(w[i])
        i += 1

    return "".join(result)


def roman_to_devanagari(text: str) -> str:
    """
    Transliterate Romanized text to Devanagari script.
    Uses lexicon lookup -> indic_transliteration -> phonetic rule fallback.
    """
    if not text:
        return ""

    words = text.split()
    converted_words = []

    for word in words:
        lower_word = word.lower().strip()
        # 1. Lexicon match
        if lower_word in LEXICON_ROMAN_TO_DEVANAGARI:
            converted_words.append(LEXICON_ROMAN_TO_DEVANAGARI[lower_word])
            continue

        # 2. Open-source library if available
        if HAS_INDIC_TRANSLITERATION:
            try:
                # ITRANS / Harvard-Kyoto to Devanagari
                trans = indic_transliterate(lower_word, sanscript.ITRANS, sanscript.DEVANAGARI)
                if trans and not re.search(r"[a-zA-Z]", trans):
                    converted_words.append(trans)
                    continue
            except Exception:
                pass

        # 3. Rule-based fallback
        converted_words.append(rule_based_roman_to_devanagari(lower_word))

    return " ".join(converted_words)


def devanagari_to_roman(text: str) -> str:
    """
    Transliterate Devanagari text to Romanized script.
    """
    if not text:
        return ""

    words = text.split()
    converted_words = []

    for word in words:
        clean = word.strip()
        if clean in LEXICON_DEVANAGARI_TO_ROMAN:
            converted_words.append(LEXICON_DEVANAGARI_TO_ROMAN[clean])
            continue

        if HAS_INDIC_TRANSLITERATION:
            try:
                trans = indic_transliterate(clean, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
                if trans:
                    converted_words.append(trans)
                    continue
            except Exception:
                pass

        converted_words.append(clean)

    return " ".join(converted_words)

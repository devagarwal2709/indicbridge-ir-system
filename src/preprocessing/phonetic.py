import difflib
import re
from typing import Dict, List, Tuple

# Indic-adapted phonetic mapping for Romanized Hindi / Hinglish.
# Unlike English Soundex (which groups by English acoustic properties),
# IndicSoundex groups Indic consonant equivalents and phonetic alternations:
# 1: Labials & Bilabials: p, b, ph, bh, f, v, w
# 2: Dentals & Alveolars: t, d, th, dh
# 3: Retroflexes & Taps: r, rh, l, ld
# 4: Palatals & Sibilants: c, ch, chh, j, jh, s, sh, shh, z
# 5: Velars & Gutturals: k, kh, g, gh, q
# 6: Nasals: m, n, ng, ny
# 7: Glides / Aspirate: y, h
INDIC_PHONETIC_GROUPS: Dict[str, str] = {
    # 1: Labials
    "p": "1", "b": "1", "ph": "1", "bh": "1", "f": "1", "v": "1", "w": "1",
    # 2: Dentals
    "t": "2", "d": "2", "th": "2", "dh": "2",
    # 3: Retroflex & Liquids
    "r": "3", "rh": "3", "l": "3", "ld": "3",
    # 4: Sibilants & Palatals
    "c": "4", "ch": "4", "chh": "4", "j": "4", "jh": "4",
    "s": "4", "sh": "4", "shh": "4", "z": "4",
    # 5: Velars
    "k": "5", "kh": "5", "g": "5", "gh": "5", "q": "5",
    # 6: Nasals
    "m": "6", "n": "6",
    # 7: Glides
    "y": "7", "h": "7",
}


def compute_indic_soundex(word: str, max_length: int = 5) -> str:
    """
    Compute Indic-adapted phonetic key for Romanized Indic words.
    
    Why this differs from English Soundex:
    - English Soundex maps 's' and 'c' together but separates 't' and 'th',
      and does not recognize Indic aspirated/unaspirated alternations ('bh' ~ 'b', 'sh' ~ 's', 'kh' ~ 'k').
    - IndicSoundex collapses aspirates and dental/palatal variants common in Romanized Indic spelling.
    """
    w = word.lower().strip()
    if not w:
        return "0000"

    # Retain the first letter (uppercase)
    first_char = w[0].upper()

    # Normalize double vowels and silent letters
    normalized = re.sub(r"aa+", "a", w)
    normalized = re.sub(r"ee+", "i", normalized)
    normalized = re.sub(r"oo+", "u", normalized)

    digits = []
    i = 1
    n = len(normalized)

    while i < n and len(digits) < max_length - 1:
        # Check two-character digraphs first
        two = normalized[i:i+2]
        one = normalized[i:i+1]

        code = None
        advance = 1

        if two in INDIC_PHONETIC_GROUPS:
            code = INDIC_PHONETIC_GROUPS[two]
            advance = 2
        elif one in INDIC_PHONETIC_GROUPS:
            code = INDIC_PHONETIC_GROUPS[one]
            advance = 1

        if code:
            # Avoid repeating same adjacent phonetic code
            if not digits or digits[-1] != code:
                digits.append(code)

        i += advance

    # Pad with zeros up to max_length
    while len(digits) < max_length - 1:
        digits.append("0")

    return first_char + "".join(digits[:max_length - 1])


def string_fuzzy_similarity(s1: str, s2: str) -> float:
    """Compute normalized character sequence similarity ratio in [0.0, 1.0]."""
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def phonetic_match_score(term1: str, term2: str) -> float:
    """
    Calculate auxiliary phonetic & fuzzy match score between two terms.
    Returns score in [0.0, 1.0].
    Combines IndicSoundex identity with Levenshtein-style fuzzy ratio.
    """
    t1 = term1.lower().strip()
    t2 = term2.lower().strip()

    if t1 == t2:
        return 1.0

    # 1. Fuzzy edit ratio
    fuzzy_sim = string_fuzzy_similarity(t1, t2)

    # 2. Indic Soundex matching
    k1 = compute_indic_soundex(t1)
    k2 = compute_indic_soundex(t2)
    soundex_match = 1.0 if (k1 == k2 and k1 != "0000") else 0.0

    # Weighted auxiliary signal: requires decent edit similarity to avoid distant collisions
    if soundex_match and fuzzy_sim >= 0.50:
        return 0.6 * soundex_match + 0.4 * fuzzy_sim
    else:
        return fuzzy_sim if fuzzy_sim >= 0.70 else 0.0


def compute_query_document_phonetic_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    threshold: float = 0.65
) -> Tuple[float, List[Tuple[str, str, float]]]:
    """
    Compute auxiliary phonetic similarity between query tokens and document tokens.
    Returns: (aggregate_score, list_of_matched_pairs)
    """
    if not query_tokens or not doc_tokens:
        return 0.0, []

    doc_token_set = set(t.lower() for t in doc_tokens)
    matches: List[Tuple[str, str, float]] = []
    total_query_score = 0.0

    for q_token in query_tokens:
        q_clean = q_token.lower()
        best_match_term = None
        best_match_score = 0.0

        for d_token in doc_token_set:
            score = phonetic_match_score(q_clean, d_token)
            if score > best_match_score:
                best_match_score = score
                best_match_term = d_token

        if best_match_score >= threshold and best_match_term:
            matches.append((q_clean, best_match_term, round(best_match_score, 4)))
            total_query_score += best_match_score

    # Normalized by number of query tokens
    aggregate_score = total_query_score / max(1, len(query_tokens))
    return min(1.0, aggregate_score), matches

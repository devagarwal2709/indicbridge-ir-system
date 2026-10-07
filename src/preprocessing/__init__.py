"""Preprocessing module for IndicBridge."""
from .language_detector import detect_language, detect_script
from .normalizer import (
    normalize_text,
    normalize_unicode,
    fold_nukta,
    strip_punctuation,
    normalize_whitespace,
    normalize_romanized_token,
)
from .transliterator import (
    roman_to_devanagari,
    devanagari_to_roman,
)
from .tokenizer import tokenize, tokenize_detailed, Token
from .stopwords import remove_stopwords, is_stopword
from .stemmer import stem_word, stem_tokens, stem_hindi, stem_english
from .phonetic import (
    compute_indic_soundex,
    phonetic_match_score,
    string_fuzzy_similarity,
    compute_query_document_phonetic_score,
)

__all__ = [
    "detect_language",
    "detect_script",
    "normalize_text",
    "normalize_unicode",
    "fold_nukta",
    "strip_punctuation",
    "normalize_whitespace",
    "normalize_romanized_token",
    "roman_to_devanagari",
    "devanagari_to_roman",
    "tokenize",
    "tokenize_detailed",
    "Token",
    "remove_stopwords",
    "is_stopword",
    "stem_word",
    "stem_tokens",
    "stem_hindi",
    "stem_english",
    "compute_indic_soundex",
    "phonetic_match_score",
    "string_fuzzy_similarity",
    "compute_query_document_phonetic_score",
]

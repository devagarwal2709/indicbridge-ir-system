try:
    import pytest
except ImportError:
    class _FakePytest:
        @staticmethod
        def approx(expected, rel=None, abs=None):
            class _Approx:
                def __init__(self, exp, tol):
                    self.exp = exp
                    self.tol = tol
                def __eq__(self, actual):
                    return __import__("math").isclose(actual, self.exp, rel_tol=self.tol, abs_tol=self.tol)
            return _Approx(expected, rel or abs or 0.01)
    pytest = _FakePytest()

from src.preprocessing.normalizer import (
    normalize_text,
    normalize_unicode,
    fold_nukta,
    normalize_romanized_token,
)
from src.preprocessing.stopwords import remove_stopwords, is_stopword
from src.preprocessing.stemmer import stem_hindi, stem_english, stem_word
from src.preprocessing.transliterator import roman_to_devanagari, devanagari_to_roman
from src.preprocessing.phonetic import compute_indic_soundex, phonetic_match_score


def test_unicode_and_nukta_normalization():
    # Nukta folding
    raw_with_nukta = "क़िताब और सफ़र"
    folded = fold_nukta(raw_with_nukta)
    assert "किताब" in folded
    assert "सफर" in folded


def test_romanized_spelling_normalization():
    assert normalize_romanized_token("bhaarat") == "bharat"
    assert normalize_romanized_token("krishan") == "krishna"
    assert normalize_romanized_token("siksha") == "shiksha"
    assert normalize_romanized_token("himalay") == "himalaya"


def test_stopword_removal():
    tokens = ["भारत", "का", "संविधान", "और", "अधिकार"]
    retained, removed = remove_stopwords(tokens)
    assert "का" in removed
    assert "और" in removed
    assert "भारत" in retained
    assert "संविधान" in retained

    hinglish_tokens = ["kal", "ka", "mausam", "kaisa", "hoga"]
    h_retained, h_removed = remove_stopwords(hinglish_tokens)
    assert "ka" in h_removed
    assert "mausam" in h_retained


def test_stemming():
    # Hindi plural/inflection stripping maps to root
    assert stem_hindi("भारतीयों") in ("भारत", "भारतीय")
    assert stem_hindi("भाषाएं") in ("भाषा", "भाष")

    # English suffix stripping
    assert stem_english("satellites") == "satellite"
    assert stem_english("explored") in ("explore", "explor")


def test_transliteration():
    # Roman to Devanagari
    dev = roman_to_devanagari("bharat")
    assert "भारत" in dev

    dev_weather = roman_to_devanagari("mausam")
    assert "मौसम" in dev_weather

    # Devanagari to Roman
    rom = devanagari_to_roman("भारत")
    assert "bharat" in rom.lower()


def test_indic_phonetic_matching():
    # Same phonetic key for Romanized Indic spelling variants
    k1 = compute_indic_soundex("bharat")
    k2 = compute_indic_soundex("bhaarat")
    assert k1 == k2

    k_krishna = compute_indic_soundex("krishna")
    k_krishan = compute_indic_soundex("krishan")
    assert k_krishna == k_krishan

    # High match score between spelling variants
    score = phonetic_match_score("krishna", "krishan")
    assert score >= 0.70

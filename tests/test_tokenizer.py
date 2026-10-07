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

from src.preprocessing.tokenizer import tokenize, tokenize_detailed
from src.preprocessing.language_detector import detect_language, detect_script


def test_tokenization_english():
    text = "Constitution of India guarantees fundamental rights."
    tokens = tokenize(text)
    assert "constitution" in tokens
    assert "india" in tokens
    assert "rights" in tokens
    assert len(tokens) == 6


def test_tokenization_hindi_devanagari():
    text = "भारत का संविधान दुनिया का सबसे बड़ा संविधान है।"
    tokens = tokenize(text)
    assert "भारत" in tokens
    assert "संविधान" in tokens
    assert "दुनिया" in tokens


def test_tokenization_code_mixed_hinglish():
    text = "Kal ka mausam kaisa hoga weather in Delhi?"
    tokens = tokenize(text)
    assert "kal" in tokens
    assert "mausam" in tokens
    assert "weather" in tokens
    assert "delhi" in tokens


def test_language_detection():
    # Hindi Devanagari
    lang_hi, script_hi, _ = detect_language("मेरा भारत देश")
    assert lang_hi == "hi"
    assert script_hi == "Devanagari"

    # English Latin
    lang_en, script_en, _ = detect_language("Weather forecast for tomorrow")
    assert lang_en == "en"
    assert script_en == "Latin"

    # Hinglish
    lang_hg, script_hg, _ = detect_language("mera bharat mahan desh hai")
    assert lang_hg == "hi-Latn"
    assert script_hg == "Latin"

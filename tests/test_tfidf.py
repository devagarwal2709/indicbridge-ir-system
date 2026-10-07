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

import math
from src.index.inverted_index import InvertedIndex
from src.index.tfidf_index import TFIDFIndex


def test_tfidf_math_formulation():
    index = InvertedIndex()
    index.add_document("doc_1", ["bharat", "desh", "bharat"], {"title": "Doc 1"})
    index.add_document("doc_2", ["bharat", "samvidhan"], {"title": "Doc 2"})
    index.add_document("doc_3", ["mausam", "delhi"], {"title": "Doc 3"})

    tfidf = TFIDFIndex(index, use_sublinear_tf=True)

    # N = 3
    # df('bharat') = 2
    # idf('bharat') = ln((3 + 1) / (2 + 1)) + 1 = ln(4/3) + 1
    expected_idf_bharat = math.log(4.0 / 3.0) + 1.0
    computed_idf = tfidf.get_idf("bharat")
    assert pytest.approx(computed_idf, 0.001) == expected_idf_bharat

    # Check vector normalization: L2 norm of doc vector weights should equal 1.0
    vec1 = tfidf.doc_vectors["doc_1"]
    sum_sq = sum(w * w for w in vec1.values())
    assert pytest.approx(sum_sq, 0.001) == 1.0


def test_cosine_similarity():
    index = InvertedIndex()
    index.add_document("doc_1", ["bharat", "desh"], {"title": "Doc 1"})
    index.add_document("doc_2", ["weather", "tomorrow"], {"title": "Doc 2"})

    tfidf = TFIDFIndex(index, use_sublinear_tf=False)
    q_vec, _ = tfidf.compute_query_vector(["bharat", "desh"])

    score_1, contribs_1 = tfidf.score_document(q_vec, "doc_1")
    score_2, contribs_2 = tfidf.score_document(q_vec, "doc_2")

    assert score_1 > 0.8  # Strong similarity
    assert score_2 == 0.0  # Orthogonal vector
    assert "bharat" in contribs_1
    assert "desh" in contribs_1

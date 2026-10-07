try:
    import pytest
except ImportError:
    pass

from src.index.inverted_index import InvertedIndex, PostingEntry


def test_inverted_index_creation():
    index = InvertedIndex()
    index.add_document("doc_1", ["bharat", "desh", "mahan"], {"title": "Doc 1"})
    index.add_document("doc_2", ["bharat", "samvidhan"], {"title": "Doc 2"})
    index.add_document("doc_3", ["mausam", "delhi"], {"title": "Doc 3"})

    assert index.num_docs == 3
    assert index.get_document_frequency("bharat") == 2
    assert index.get_document_frequency("desh") == 1
    assert index.get_document_frequency("unseen_term") == 0

    postings = index.get_postings("bharat")
    assert len(postings) == 2
    assert {p.doc_id for p in postings} == {"doc_1", "doc_2"}


def test_candidate_retrieval():
    index = InvertedIndex()
    index.add_document("doc_1", ["weather", "forecast"], {"title": "Weather"})
    index.add_document("doc_2", ["space", "mission"], {"title": "Space"})
    index.add_document("doc_3", ["weather", "rain"], {"title": "Rain"})

    candidates = index.get_candidate_documents(["weather"])
    assert candidates == {"doc_1", "doc_3"}

    multi_candidates = index.get_candidate_documents(["weather", "space"])
    assert multi_candidates == {"doc_1", "doc_2", "doc_3"}

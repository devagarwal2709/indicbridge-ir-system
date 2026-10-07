from typing import Any, Dict, List, Optional, Set, Tuple
from ..index.inverted_index import InvertedIndex
from ..index.tfidf_index import TFIDFIndex


class TFIDFRetriever:
    """Standard TF-IDF Cosine similarity retriever over inverted index candidates."""

    def __init__(self, index: InvertedIndex, tfidf: Optional[TFIDFIndex] = None):
        self.index = index
        self.tfidf = tfidf or TFIDFIndex(index)

    def retrieve(
        self,
        query_terms: List[str],
        candidate_doc_ids: Optional[Set[str]] = None,
        min_similarity: float = 0.001
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate documents and calculate cosine similarity scores.
        """
        if not query_terms:
            return []

        query_vector, _ = self.tfidf.compute_query_vector(query_terms)

        if candidate_doc_ids is None:
            candidates = self.index.get_candidate_documents(query_terms)
        else:
            candidates = candidate_doc_ids

        scored_docs: List[Dict[str, Any]] = []

        for doc_id in candidates:
            score, contribs = self.tfidf.score_document(query_vector, doc_id)
            if score >= min_similarity:
                scored_docs.append({
                    "doc_id": doc_id,
                    "cosine_score": score,
                    "term_contributions": contribs,
                })

        scored_docs.sort(key=lambda x: x["cosine_score"], reverse=True)
        return scored_docs

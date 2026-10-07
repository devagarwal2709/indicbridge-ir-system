import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple
from .inverted_index import InvertedIndex


class TFIDFIndex:
    """
    Vector Space Model with TF-IDF Weighting and Cosine Similarity.
    
    Mathematical Formulation:
    - IDF: idf(t) = ln((N + 1) / (df(t) + 1)) + 1
    - TF (sublinear): tf(t, d) = 1 + ln(count(t, d)) if count > 0 else 0
    - Weight: w(t, d) = tf(t, d) * idf(t)
    - Vector L2 Norm: ||d||_2 = sqrt( sum( w(t, d)^2 ) )
    - Normalized Weight: w_norm(t, d) = w(t, d) / ||d||_2
    - Cosine Similarity:
        cosine(q, d) = sum_{t in q cap d} (w_norm(t, q) * w_norm(t, d))
    """

    def __init__(self, inverted_index: InvertedIndex, use_sublinear_tf: bool = True):
        self.inverted_index = inverted_index
        self.use_sublinear_tf = use_sublinear_tf

        # Precomputed IDF table: term -> idf value
        self.idf_cache: Dict[str, float] = {}

        # Document vectors: doc_id -> {term: normalized_tfidf_weight}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}

        # Document vector norms before normalization: doc_id -> L2_norm
        self.doc_norms: Dict[str, float] = {}

        self._build_model()

    def _compute_idf(self, df: int, num_docs: int) -> float:
        """Standard smoothed textbook IDF formulation."""
        return math.log((num_docs + 1) / (df + 1)) + 1.0

    def get_idf(self, term: str) -> float:
        """Retrieve precomputed IDF for a term, or compute for unseen term."""
        term_clean = term.lower().strip()
        if term_clean in self.idf_cache:
            return self.idf_cache[term_clean]

        df = self.inverted_index.get_document_frequency(term_clean)
        val = self._compute_idf(df, self.inverted_index.num_docs)
        self.idf_cache[term_clean] = val
        return val

    def _compute_tf(self, count: int) -> float:
        """Compute Term Frequency weight (sublinear or raw)."""
        if count <= 0:
            return 0.0
        if self.use_sublinear_tf:
            return 1.0 + math.log(count)
        return float(count)

    def _build_model(self) -> None:
        """Precompute IDF values and pre-normalize document TF-IDF vectors."""
        num_docs = self.inverted_index.num_docs
        if num_docs == 0:
            return

        # 1. Compute IDFs for all terms in vocabulary
        for term, postings in self.inverted_index.index.items():
            self.idf_cache[term] = self._compute_idf(len(postings), num_docs)

        # 2. Compute TF-IDF vectors per document
        temp_doc_weights: Dict[str, Dict[str, float]] = {
            doc_id: {} for doc_id in self.inverted_index.doc_lengths.keys()
        }

        for term, postings in self.inverted_index.index.items():
            idf_val = self.idf_cache[term]
            for p in postings:
                tf_val = self._compute_tf(p.tf)
                weight = tf_val * idf_val
                temp_doc_weights[p.doc_id][term] = weight

        # 3. L2 normalize document vectors
        for doc_id, weights in temp_doc_weights.items():
            sum_sq = sum(w * w for w in weights.values())
            norm = math.sqrt(sum_sq) if sum_sq > 0 else 1.0
            self.doc_norms[doc_id] = norm

            # Normalize weights
            self.doc_vectors[doc_id] = {
                term: w / norm for term, w in weights.items()
            }

    def compute_query_vector(self, query_tokens: List[str]) -> Tuple[Dict[str, float], float]:
        """
        Build L2-normalized TF-IDF vector for a query.
        Returns: (normalized_query_weights, original_l2_norm)
        """
        if not query_tokens:
            return {}, 0.0

        counts = Counter(t.lower() for t in query_tokens)
        raw_weights: Dict[str, float] = {}

        for term, cnt in counts.items():
            tf_val = self._compute_tf(cnt)
            idf_val = self.get_idf(term)
            raw_weights[term] = tf_val * idf_val

        sum_sq = sum(w * w for w in raw_weights.values())
        norm = math.sqrt(sum_sq) if sum_sq > 0 else 1.0

        normalized_weights = {term: w / norm for term, w in raw_weights.items()}
        return normalized_weights, norm

    def score_document(
        self,
        query_vector: Dict[str, float],
        doc_id: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute cosine similarity between query and a document.
        Returns: (cosine_similarity, term_contributions)
        """
        if doc_id not in self.doc_vectors or not query_vector:
            return 0.0, {}

        doc_weights = self.doc_vectors[doc_id]
        score = 0.0
        contributions: Dict[str, float] = {}

        for term, q_w in query_vector.items():
            if term in doc_weights:
                term_score = q_w * doc_weights[term]
                score += term_score
                contributions[term] = term_score

        return score, contributions

import time
from typing import Any, Dict, List, Optional
from ..index.inverted_index import InvertedIndex
from ..index.tfidf_index import TFIDFIndex
from ..preprocessing.tokenizer import tokenize


class BaselineRetriever:
    """
    BASELINE RETRIEVAL SYSTEM:
    Pure exact-token TF-IDF search without cross-script transliteration,
    without Indic phonetic matching, and without Romanized spelling normalization.
    
    Serves as the rigorous benchmark comparison required by Section 10.
    """

    def __init__(self, index: InvertedIndex, tfidf: Optional[TFIDFIndex] = None):
        self.index = index
        self.tfidf = tfidf or TFIDFIndex(index)

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.001
    ) -> Dict[str, Any]:
        """
        Execute baseline retrieval:
        1. Simple tokenization (exact tokens)
        2. Candidate retrieval via inverted index
        3. Cosine similarity
        4. Top-K ranking
        """
        start_time = time.time()

        if not query or not query.strip():
            return {
                "query": query,
                "retriever": "Baseline (Exact TF-IDF)",
                "total_candidates": 0,
                "elapsed_ms": 0.0,
                "results": [],
            }

        # Raw tokenization only (no transliteration, no phonetic, no spelling normalization)
        raw_tokens = tokenize(query, lowercase=True)
        query_vector, _ = self.tfidf.compute_query_vector(raw_tokens)

        # Candidate documents containing at least one exact query term
        candidates = self.index.get_candidate_documents(raw_tokens)

        results: List[Dict[str, Any]] = []

        for doc_id in candidates:
            cosine_score, term_contribs = self.tfidf.score_document(query_vector, doc_id)
            if cosine_score < min_similarity:
                continue

            metadata = self.index.doc_metadata.get(doc_id, {})
            matched_terms = list(term_contribs.keys())

            results.append({
                "doc_id": doc_id,
                "title": metadata.get("title", ""),
                "language": metadata.get("language", ""),
                "text": metadata.get("text", ""),
                "url": metadata.get("url", ""),
                "final_score": round(cosine_score, 4),
                "cosine_score": round(cosine_score, 4),
                "transliteration_score": 0.0,
                "phonetic_score": 0.0,
                "exact_match_score": 1.0 if any(t in raw_tokens for t in matched_terms) else 0.0,
                "matched_terms": matched_terms,
                "term_contributions": {t: round(s, 4) for t, s in term_contribs.items()},
                "explanation": (
                    f"Baseline exact match on terms: [{', '.join(matched_terms)}]. "
                    f"Cosine similarity = {cosine_score:.4f}. "
                    f"No cross-script or phonetic expansion applied."
                ),
            })

        # Rank by cosine score descending
        results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = results[:top_k]

        # Add ranks
        for rank, res in enumerate(top_results, start=1):
            res["rank"] = rank

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "query": query,
            "tokens": raw_tokens,
            "retriever": "Baseline (Exact TF-IDF)",
            "total_candidates": len(candidates),
            "elapsed_ms": round(elapsed_ms, 2),
            "results": top_results,
        }

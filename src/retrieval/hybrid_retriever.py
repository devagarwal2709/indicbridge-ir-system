import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..index.inverted_index import InvertedIndex
from ..index.tfidf_index import TFIDFIndex
from ..preprocessing.language_detector import detect_language, detect_script
from ..preprocessing.normalizer import (
    normalize_text,
    normalize_unicode,
    normalize_romanized_token,
    fold_nukta,
)
from ..preprocessing.transliterator import (
    roman_to_devanagari,
    devanagari_to_roman,
)
from ..preprocessing.tokenizer import tokenize, tokenize_detailed
from ..preprocessing.stopwords import remove_stopwords
from ..preprocessing.stemmer import stem_tokens
from ..preprocessing.phonetic import (
    compute_query_document_phonetic_score,
    phonetic_match_score,
)
from .scorer import ExplainableScorer


class IndicBridgeRetriever:
    """
    INDICBRIDGE: Explainable Multilingual Information Retrieval System.
    Cross-script, cross-variant retrieval with transparent score fusion.
    """

    def __init__(
        self,
        index: InvertedIndex,
        tfidf: Optional[TFIDFIndex] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.tfidf = tfidf or TFIDFIndex(index)
        self.config = config or {}

        # Retreival parameters
        ret_conf = self.config.get("retrieval", {})
        self.alpha = float(ret_conf.get("alpha", 0.70))
        self.beta = float(ret_conf.get("beta", 0.15))
        self.gamma = float(ret_conf.get("gamma", 0.10))
        self.delta = float(ret_conf.get("delta", 0.05))
        self.min_similarity = float(ret_conf.get("min_similarity", 0.001))

        self.scorer = ExplainableScorer(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
        )

    def set_weights(self, alpha: float, beta: float, gamma: float, delta: float) -> None:
        """Dynamically update fusion weights."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.scorer = ExplainableScorer(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

    def search(
        self,
        query: str,
        top_k: int = 10,
        enable_transliteration: bool = True,
        enable_phonetic: bool = True,
        enable_stemming: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full 11-stage explainable multilingual IR pipeline.
        Measures real runtime in milliseconds.
        """
        total_start = time.time()

        if not query or not query.strip():
            return {
                "original_query": query,
                "retriever": "IndicBridge",
                "results": [],
                "timing_ms": {"query_processing": 0.0, "candidate_retrieval": 0.0, "ranking": 0.0, "total": 0.0},
                "pipeline_trace": {},
            }

        # -------------------------------------------------------------
        # STAGE 1: Query Preprocessing (Preserve original for display)
        # -------------------------------------------------------------
        qp_start = time.time()
        raw_query = query.strip()

        # -------------------------------------------------------------
        # STAGE 2: Language & Script Identification
        # -------------------------------------------------------------
        detected_lang, detected_script, lang_confidence = detect_language(raw_query)

        # -------------------------------------------------------------
        # STAGE 3: Unicode & Text Normalization
        # -------------------------------------------------------------
        norm_query = normalize_text(
            raw_query,
            fold_devanagari_nukta=True,
            lowercase_latin=True,
            normalize_romanized=False,
        )

        # -------------------------------------------------------------
        # STAGE 4: Transliteration Normalization (Cross-Script Expansion)
        # -------------------------------------------------------------
        translit_query = ""
        translit_tokens: List[str] = []
        if enable_transliteration:
            if detected_script in ("Latin", "Mixed"):
                # Transliterate Romanized Hindi into Devanagari
                translit_query = roman_to_devanagari(norm_query)
            elif detected_script == "Devanagari":
                # Transliterate Devanagari to Romanized
                translit_query = devanagari_to_roman(norm_query)

            if translit_query:
                translit_tokens = tokenize(translit_query)

        # -------------------------------------------------------------
        # STAGE 5: Romanized Spelling Normalization
        # -------------------------------------------------------------
        canonical_query = normalize_text(
            norm_query,
            fold_devanagari_nukta=True,
            lowercase_latin=True,
            normalize_romanized=True,
        )

        # -------------------------------------------------------------
        # STAGE 6: Tokenization & Stop-word Handling
        # -------------------------------------------------------------
        raw_tokens = tokenize(norm_query)
        canonical_tokens = tokenize(canonical_query)

        # Merge unique expanded terms while preserving primary tokens
        all_query_terms_set = set(raw_tokens + canonical_tokens)
        if enable_transliteration and translit_tokens:
            all_query_terms_set.update(translit_tokens)

        # Remove stop words from primary tokens
        active_tokens, removed_stopwords = remove_stopwords(list(all_query_terms_set))

        # -------------------------------------------------------------
        # STAGE 7: Stemming / Lemmatization
        # -------------------------------------------------------------
        stemmed_tokens = stem_tokens(active_tokens) if enable_stemming else active_tokens

        # Unified query terms for TF-IDF candidate matching
        retrieval_query_terms = list(set(active_tokens + stemmed_tokens))
        qp_ms = (time.time() - qp_start) * 1000.0

        # -------------------------------------------------------------
        # STAGE 8: Sparse TF-IDF Retrieval over Inverted Index
        # -------------------------------------------------------------
        cand_start = time.time()
        candidate_ids = self.index.get_candidate_documents(retrieval_query_terms)

        # Build query vector
        query_vector, q_norm = self.tfidf.compute_query_vector(retrieval_query_terms)
        cand_ms = (time.time() - cand_start) * 1000.0

        # -------------------------------------------------------------
        # STAGE 9 & 10: Auxiliary Matching & Score Fusion
        # -------------------------------------------------------------
        rank_start = time.time()
        scored_results: List[Dict[str, Any]] = []

        # Find high-IDF discriminative terms in query
        term_idfs = [(t, self.tfidf.get_idf(t)) for t in retrieval_query_terms]
        term_idfs.sort(key=lambda x: x[1], reverse=True)
        high_idf_terms = [t for t, idf in term_idfs[:3] if idf > 1.2]

        for doc_id in candidate_ids:
            cosine_score, term_contribs = self.tfidf.score_document(query_vector, doc_id)
            doc_meta = self.index.doc_metadata.get(doc_id, {})
            doc_tokens = self.index.doc_tokens.get(doc_id, [])
            doc_text = doc_meta.get("text", "")

            # A. Transliteration match signal
            translit_score = 0.0
            translit_matches = []
            if enable_transliteration and translit_tokens:
                doc_tokens_set = set(t.lower() for t in doc_tokens)
                for q_t in translit_tokens:
                    if q_t.lower() in doc_tokens_set:
                        translit_matches.append((norm_query, q_t))
                        translit_score += 1.0
                if translit_tokens:
                    translit_score = min(1.0, translit_score / len(translit_tokens))

            # B. Phonetic & fuzzy match signal
            phonetic_score = 0.0
            phonetic_matches = []
            if enable_phonetic and detected_script in ("Latin", "Mixed"):
                phonetic_score, phonetic_matches = compute_query_document_phonetic_score(
                    raw_tokens,
                    doc_tokens,
                    threshold=0.65
                )

            # C. Exact token match signal
            exact_matches = [t for t in raw_tokens if t in doc_tokens]
            exact_match_score = len(exact_matches) / max(1, len(raw_tokens))

            # D. Fused Score
            final_score = self.scorer.compute_fusion_score(
                cosine_score=cosine_score,
                translit_score=translit_score,
                phonetic_score=phonetic_score,
                exact_match_score=exact_match_score,
            )

            if final_score < self.min_similarity:
                continue

            # Generate explanation details
            matched_terms = list(term_contribs.keys())
            doc_high_idf = [t for t in high_idf_terms if t in matched_terms]

            explanation = self.scorer.generate_explanation(
                doc_id=doc_id,
                title=doc_meta.get("title", ""),
                language=doc_meta.get("language", ""),
                cosine_score=cosine_score,
                translit_score=translit_score,
                phonetic_score=phonetic_score,
                exact_match_score=exact_match_score,
                final_score=final_score,
                matched_terms=matched_terms,
                high_idf_terms=doc_high_idf,
                translit_matches=translit_matches,
                phonetic_matches=phonetic_matches,
            )

            scored_results.append({
                "doc_id": doc_id,
                "title": doc_meta.get("title", ""),
                "language": doc_meta.get("language", ""),
                "text": doc_text,
                "url": doc_meta.get("url", ""),
                "final_score": final_score,
                "cosine_score": round(cosine_score, 4),
                "transliteration_score": round(translit_score, 4),
                "phonetic_score": round(phonetic_score, 4),
                "exact_match_score": round(exact_match_score, 4),
                "matched_terms": matched_terms,
                "high_idf_terms": doc_high_idf,
                "term_contributions": {t: round(s, 4) for t, s in term_contribs.items()},
                "explanation_summary": explanation["summary"],
                "explanation_details": explanation,
            })

        # -------------------------------------------------------------
        # STAGE 11: Top-K Ranking
        # -------------------------------------------------------------
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = scored_results[:top_k]

        for rank, res in enumerate(top_results, start=1):
            res["rank"] = rank

        rank_ms = (time.time() - rank_start) * 1000.0
        total_ms = (time.time() - total_start) * 1000.0

        return {
            "original_query": raw_query,
            "normalized_query": norm_query,
            "transliterated_query": translit_query,
            "detected_language": detected_lang,
            "detected_script": detected_script,
            "confidence": round(lang_confidence, 2),
            "tokens": raw_tokens,
            "removed_stopwords": removed_stopwords,
            "stemmed_tokens": stemmed_tokens,
            "retrieval_query_terms": retrieval_query_terms,
            "total_candidates": len(candidate_ids),
            "total_retrieved": len(scored_results),
            "results": top_results,
            "timing_ms": {
                "query_processing": round(qp_ms, 2),
                "candidate_retrieval": round(cand_ms, 2),
                "ranking": round(rank_ms, 2),
                "total": round(total_ms, 2),
            },
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta,
            },
        }

from typing import Any, Dict, List, Optional, Tuple


class ExplainableScorer:
    """
    Transparent Multi-Signal Score Fusion & Explanation Engine.
    
    Formula:
      final_score = alpha * cosine_score
                  + beta  * translit_score
                  + gamma * phonetic_score
                  + delta * exact_match_score
    
    Ensures all component signals are scaled to [0, 1] before linear combination.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        beta: float = 0.15,
        gamma: float = 0.10,
        delta: float = 0.05
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def compute_fusion_score(
        self,
        cosine_score: float,
        translit_score: float,
        phonetic_score: float,
        exact_match_score: float
    ) -> float:
        """Compute transparent fused relevance score."""
        # Ensure clamped range [0.0, 1.0]
        c = max(0.0, min(1.0, cosine_score))
        t = max(0.0, min(1.0, translit_score))
        p = max(0.0, min(1.0, phonetic_score))
        e = max(0.0, min(1.0, exact_match_score))

        final = (
            self.alpha * c +
            self.beta * t +
            self.gamma * p +
            self.delta * e
        )
        return round(final, 4)

    def generate_explanation(
        self,
        doc_id: str,
        title: str,
        language: str,
        cosine_score: float,
        translit_score: float,
        phonetic_score: float,
        exact_match_score: float,
        final_score: float,
        matched_terms: List[str],
        high_idf_terms: List[str],
        translit_matches: List[Tuple[str, str]],
        phonetic_matches: List[Tuple[str, str, float]]
    ) -> Dict[str, Any]:
        """
        Generate human-interpretable explanation card for search result.
        Avoids obscure internal vector dumps and provides scannable insights.
        """
        reasons = []

        if cosine_score > 0.01:
            if high_idf_terms:
                reasons.append(
                    f"Strong TF-IDF vector match on key distinctive terms: '{', '.join(high_idf_terms)}' "
                    f"(cosine: {cosine_score:.3f})"
                )
            else:
                reasons.append(f"Lexical TF-IDF match (cosine: {cosine_score:.3f})")

        if translit_score > 0.1:
            translit_pairs_str = ", ".join(f"{q}->{d}" for q, d in translit_matches[:3])
            reasons.append(
                f"Cross-script transliteration match ({translit_pairs_str}) boosted score by "
                f"+{(self.beta * translit_score):.3f}"
            )

        if phonetic_score > 0.1:
            phonetic_pairs_str = ", ".join(f"{q}~{d}" for q, d, _ in phonetic_matches[:3])
            reasons.append(
                f"Adapted Indic phonetic/fuzzy match on spelling variants ({phonetic_pairs_str}) "
                f"boosted score by +{(self.gamma * phonetic_score):.3f}"
            )

        if exact_match_score > 0.5:
            reasons.append("Contains exact match of original query terms")

        why_text = (
            " ".join(reasons)
            if reasons
            else "Retrieved via auxiliary multilingual similarity match."
        )

        return {
            "summary": why_text,
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta,
            },
            "components": {
                "cosine_similarity": round(cosine_score, 4),
                "transliteration_match": round(translit_score, 4),
                "phonetic_similarity": round(phonetic_score, 4),
                "exact_match_signal": round(exact_match_score, 4),
                "final_score": round(final_score, 4),
            },
            "matched_terms": matched_terms,
            "high_idf_terms": high_idf_terms,
            "translit_matches": [f"{q} ➔ {d}" for q, d in translit_matches],
            "phonetic_matches": [f"{q} ≈ {d} ({s})" for q, d, s in phonetic_matches],
        }

import json
from typing import Any, Dict, List, Optional, Set
from .metrics import (
    precision_at_k,
    recall_at_k,
    f1_score_at_k,
    reciprocal_rank,
    average_precision,
)
from ..retrieval.baseline import BaselineRetriever
from ..retrieval.hybrid_retriever import IndicBridgeRetriever


class BenchmarkEvaluator:
    """
    Academic IR Evaluation Suite for IndicBridge and Baseline Systems.
    Evaluates Precision@5, Precision@10, Recall@5, Recall@10, F1@5, MRR, and MAP.
    Computes both macro-averages and per-category breakdowns.
    """

    def __init__(
        self,
        baseline_retriever: BaselineRetriever,
        indicbridge_retriever: IndicBridgeRetriever,
        queries_file: str = "data/evaluation_queries.json",
    ):
        self.baseline = baseline_retriever
        self.indicbridge = indicbridge_retriever
        self.queries_file = queries_file
        self.queries: List[Dict[str, Any]] = self._load_queries()

    def _load_queries(self) -> List[Dict[str, Any]]:
        with open(self.queries_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_system(
        self,
        system_type: str = "indicbridge",
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Run evaluation on all queries for either 'baseline' or 'indicbridge'.
        """
        query_evals: List[Dict[str, Any]] = []

        for item in self.queries:
            query_text = item["query"]
            relevant_docs: Set[str] = set(item["relevant_documents"])
            category = item.get("category", "General")

            if system_type == "baseline":
                search_res = self.baseline.search(query_text, top_k=top_k)
            else:
                search_res = self.indicbridge.search(query_text, top_k=top_k)

            retrieved_ids = [r["doc_id"] for r in search_res["results"]]

            p5 = precision_at_k(retrieved_ids, relevant_docs, k=5)
            p10 = precision_at_k(retrieved_ids, relevant_docs, k=10)
            r5 = recall_at_k(retrieved_ids, relevant_docs, k=5)
            r10 = recall_at_k(retrieved_ids, relevant_docs, k=10)
            f1_5 = f1_score_at_k(retrieved_ids, relevant_docs, k=5)
            mrr = reciprocal_rank(retrieved_ids, relevant_docs)
            ap = average_precision(retrieved_ids, relevant_docs)

            query_evals.append({
                "query_id": item.get("id", ""),
                "query": query_text,
                "category": category,
                "relevant_count": len(relevant_docs),
                "retrieved_count": len(retrieved_ids),
                "retrieved_ids": retrieved_ids,
                "p@5": p5,
                "p@10": p10,
                "r@5": r5,
                "r@10": r10,
                "f1@5": f1_5,
                "mrr": mrr,
                "ap": ap,
            })

        n = max(1, len(query_evals))
        macro_metrics = {
            "p@5": round(sum(q["p@5"] for q in query_evals) / n, 4),
            "p@10": round(sum(q["p@10"] for q in query_evals) / n, 4),
            "r@5": round(sum(q["r@5"] for q in query_evals) / n, 4),
            "r@10": round(sum(q["r@10"] for q in query_evals) / n, 4),
            "f1@5": round(sum(q["f1@5"] for q in query_evals) / n, 4),
            "mrr": round(sum(q["mrr"] for q in query_evals) / n, 4),
            "map": round(sum(q["ap"] for q in query_evals) / n, 4),
        }

        # Category-wise breakdown
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for q in query_evals:
            cat = q["category"]
            categories.setdefault(cat, []).append(q)

        category_breakdown = {}
        for cat, items in categories.items():
            k_len = max(1, len(items))
            category_breakdown[cat] = {
                "num_queries": len(items),
                "p@5": round(sum(x["p@5"] for x in items) / k_len, 4),
                "r@5": round(sum(x["r@5"] for x in items) / k_len, 4),
                "mrr": round(sum(x["mrr"] for x in items) / k_len, 4),
                "map": round(sum(x["ap"] for x in items) / k_len, 4),
            }

        return {
            "system": system_type,
            "num_queries": len(self.queries),
            "macro_metrics": macro_metrics,
            "category_breakdown": category_breakdown,
            "queries": query_evals,
        }

    def compare(self, top_k: int = 10) -> Dict[str, Any]:
        """Run complete head-to-head comparison between Baseline and IndicBridge."""
        baseline_eval = self.evaluate_system("baseline", top_k=top_k)
        indicbridge_eval = self.evaluate_system("indicbridge", top_k=top_k)

        delta = {}
        for metric, val in indicbridge_eval["macro_metrics"].items():
            base_val = baseline_eval["macro_metrics"][metric]
            diff = val - base_val
            pct = (diff / base_val * 100.0) if base_val > 0 else 0.0
            delta[metric] = {
                "baseline": base_val,
                "indicbridge": val,
                "absolute_gain": round(diff, 4),
                "relative_gain_pct": round(pct, 2),
            }

        return {
            "baseline": baseline_eval,
            "indicbridge": indicbridge_eval,
            "comparison": delta,
        }

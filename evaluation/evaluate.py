#!/usr/bin/env python3
"""
IndicBridge Evaluation Script
Executes full benchmark evaluation across Baseline and IndicBridge systems.
Calculates real Precision@K, Recall@K, F1, MRR, and MAP.
Generates JSON and CSV summary reports.
"""

import csv
import json
import os
import sys

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.index.inverted_index import InvertedIndex
from src.index.tfidf_index import TFIDFIndex
from src.retrieval.baseline import BaselineRetriever
from src.retrieval.hybrid_retriever import IndicBridgeRetriever
from src.evaluation.evaluator import BenchmarkEvaluator


def run_evaluation():
    print("=" * 75)
    print("INDICBRIDGE: Cross-Script Multilingual IR Benchmark Evaluation")
    print("=" * 75)

    config = load_config()
    cache_dir = config["paths"]["index_dir"]

    # Load inverted index
    index = InvertedIndex()
    index_file = os.path.join(cache_dir, "inverted_index.json")
    if not os.path.exists(index_file):
        from scripts.build_index import build_and_save_index
        index, tfidf = build_and_save_index()
    else:
        index.load_index(cache_dir)
        tfidf = TFIDFIndex(index, use_sublinear_tf=config["index"]["use_sublinear_tf"])

    baseline = BaselineRetriever(index, tfidf)
    indicbridge = IndicBridgeRetriever(index, tfidf, config)

    evaluator = BenchmarkEvaluator(
        baseline_retriever=baseline,
        indicbridge_retriever=indicbridge,
        queries_file=config["paths"]["evaluation_queries"],
    )

    print(f"[*] Running benchmark on {len(evaluator.queries)} evaluation queries...")
    results = evaluator.compare(top_k=10)

    base_m = results["baseline"]["macro_metrics"]
    ind_m = results["indicbridge"]["macro_metrics"]
    comp = results["comparison"]

    print("\n" + "=" * 75)
    print("OVERALL MACRO-AVERAGE METRICS COMPARISON")
    print("=" * 75)
    print(f"{'Metric':<12} | {'Baseline (Exact TF-IDF)':<24} | {'IndicBridge (Multilingual)':<26} | {'Gain':<10}")
    print("-" * 75)

    for metric in ["p@5", "p@10", "r@5", "r@10", "f1@5", "mrr", "map"]:
        b_val = base_m[metric]
        i_val = ind_m[metric]
        gain = comp[metric]["relative_gain_pct"]
        sign = "+" if gain >= 0 else ""
        print(f"{metric.upper():<12} | {b_val:<24.4f} | {i_val:<26.4f} | {sign}{gain:.1f}%")

    print("-" * 75)

    # Category breakdown table
    print("\n" + "=" * 75)
    print("PER-CATEGORY BREAKDOWN (IndicBridge Performance)")
    print("=" * 75)
    print(f"{'Category':<22} | {'Queries':<8} | {'P@5':<8} | {'R@5':<8} | {'MRR':<8} | {'MAP':<8}")
    print("-" * 75)

    cat_rows = []
    for cat, data in results["indicbridge"]["category_breakdown"].items():
        print(f"{cat:<22} | {data['num_queries']:<8} | {data['p@5']:<8.4f} | {data['r@5']:<8.4f} | {data['mrr']:<8.4f} | {data['map']:<8.4f}")
        cat_rows.append({
            "category": cat,
            "num_queries": data["num_queries"],
            "p@5": data["p@5"],
            "r@5": data["r@5"],
            "mrr": data["mrr"],
            "map": data["map"],
        })
    print("-" * 75)

    # Save outputs to disk
    out_dir = config["paths"]["evaluation_results"]
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Detailed JSON evaluation saved to: {json_path}")

    # Save summary CSV using standard csv writer
    csv_path = os.path.join(out_dir, "evaluation_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", "IndicBridge", "Relative_Gain_Pct"])
        for m in ["p@5", "p@10", "r@5", "r@10", "f1@5", "mrr", "map"]:
            writer.writerow([m.upper(), base_m[m], ind_m[m], comp[m]["relative_gain_pct"]])
    print(f"[+] Summary CSV saved to: {csv_path}")

    cat_csv_path = os.path.join(out_dir, "category_summary.csv")
    with open(cat_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "num_queries", "p@5", "r@5", "mrr", "map"])
        for r in cat_rows:
            writer.writerow([r["category"], r["num_queries"], r["p@5"], r["r@5"], r["mrr"], r["map"]])
    print(f"[+] Category CSV saved to: {cat_csv_path}")
    print("=" * 75)

    return results


if __name__ == "__main__":
    run_evaluation()

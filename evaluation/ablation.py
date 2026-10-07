#!/usr/bin/env python3
"""
IndicBridge Ablation Study
Empirically quantifies the retrieval contribution of each architectural component:
1. TF-IDF Only (Baseline)
2. TF-IDF + Normalization
3. TF-IDF + Normalization + Transliteration
4. TF-IDF + Normalization + Phonetic/Fuzzy Matching
5. Full IndicBridge (All components integrated)

Generates:
- evaluation/results/ablation_results.csv
- evaluation/results/ablation_chart.png (if matplotlib available)
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
from src.evaluation.metrics import precision_at_k, recall_at_k, f1_score_at_k, reciprocal_rank, average_precision


def run_ablation():
    print("=" * 75)
    print("INDICBRIDGE: Architectural Component Ablation Study")
    print("=" * 75)

    config = load_config()
    cache_dir = config["paths"]["index_dir"]

    # Load index
    index = InvertedIndex()
    index_file = os.path.join(cache_dir, "inverted_index.json")
    if not os.path.exists(index_file):
        from scripts.build_index import build_and_save_index
        index, tfidf = build_and_save_index()
    else:
        index.load_index(cache_dir)
        tfidf = TFIDFIndex(index, use_sublinear_tf=config["index"]["use_sublinear_tf"])

    # Load queries
    with open(config["paths"]["evaluation_queries"], "r", encoding="utf-8") as f:
        queries = json.load(f)

    # 5 Ablation Configurations
    configurations = [
        {
            "name": "1. TF-IDF Only (Baseline)",
            "type": "baseline",
            "translit": False,
            "phonetic": False,
            "stemming": False,
            "description": "Exact token TF-IDF matching without linguistic preprocessing",
        },
        {
            "name": "2. TF-IDF + Normalization",
            "type": "indicbridge",
            "translit": False,
            "phonetic": False,
            "stemming": True,
            "description": "Unicode + nukta folding + romanized normalization + stemming",
        },
        {
            "name": "3. TF-IDF + Transliteration",
            "type": "indicbridge",
            "translit": True,
            "phonetic": False,
            "stemming": True,
            "description": "Adds bidirectional Devanagari <-> Latin transliteration expansion",
        },
        {
            "name": "4. TF-IDF + Phonetic Matching",
            "type": "indicbridge",
            "translit": False,
            "phonetic": True,
            "stemming": True,
            "description": "Adds adapted IndicSoundex and fuzzy string distance scoring",
        },
        {
            "name": "5. Full IndicBridge",
            "type": "indicbridge",
            "translit": True,
            "phonetic": True,
            "stemming": True,
            "description": "All pipeline components with multi-signal score fusion",
        },
    ]

    ablation_rows = []

    for cfg in configurations:
        p5_list, p10_list, r5_list, r10_list, f1_list, mrr_list, map_list = [], [], [], [], [], [], []

        if cfg["type"] == "baseline":
            retriever = BaselineRetriever(index, tfidf)
            for item in queries:
                q = item["query"]
                rel = set(item["relevant_documents"])
                res = retriever.search(q, top_k=10)
                ret_ids = [r["doc_id"] for r in res["results"]]

                p5_list.append(precision_at_k(ret_ids, rel, 5))
                p10_list.append(precision_at_k(ret_ids, rel, 10))
                r5_list.append(recall_at_k(ret_ids, rel, 5))
                r10_list.append(recall_at_k(ret_ids, rel, 10))
                f1_list.append(f1_score_at_k(ret_ids, rel, 5))
                mrr_list.append(reciprocal_rank(ret_ids, rel))
                map_list.append(average_precision(ret_ids, rel))
        else:
            retriever = IndicBridgeRetriever(index, tfidf, config)
            for item in queries:
                q = item["query"]
                rel = set(item["relevant_documents"])
                res = retriever.search(
                    q,
                    top_k=10,
                    enable_transliteration=cfg["translit"],
                    enable_phonetic=cfg["phonetic"],
                    enable_stemming=cfg["stemming"],
                )
                ret_ids = [r["doc_id"] for r in res["results"]]

                p5_list.append(precision_at_k(ret_ids, rel, 5))
                p10_list.append(precision_at_k(ret_ids, rel, 10))
                r5_list.append(recall_at_k(ret_ids, rel, 5))
                r10_list.append(recall_at_k(ret_ids, rel, 10))
                f1_list.append(f1_score_at_k(ret_ids, rel, 5))
                mrr_list.append(reciprocal_rank(ret_ids, rel))
                map_list.append(average_precision(ret_ids, rel))

        n = len(queries)
        row = {
            "Configuration": cfg["name"],
            "Description": cfg["description"],
            "P@5": round(sum(p5_list) / n, 4),
            "P@10": round(sum(p10_list) / n, 4),
            "R@5": round(sum(r5_list) / n, 4),
            "R@10": round(sum(r10_list) / n, 4),
            "F1@5": round(sum(f1_list) / n, 4),
            "MRR": round(sum(mrr_list) / n, 4),
            "MAP": round(sum(map_list) / n, 4),
        }
        ablation_rows.append(row)

    print("\nABLATION STUDY RESULTS:")
    print("-" * 75)
    print(f"{'Configuration':<30} | {'P@5':<8} | {'R@5':<8} | {'F1@5':<8} | {'MRR':<8} | {'MAP':<8}")
    print("-" * 75)
    for r in ablation_rows:
        print(f"{r['Configuration']:<30} | {r['P@5']:<8.4f} | {r['R@5']:<8.4f} | {r['F1@5']:<8.4f} | {r['MRR']:<8.4f} | {r['MAP']:<8.4f}")
    print("-" * 75)

    out_dir = config["paths"]["evaluation_results"]
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ablation_results.csv")
    fieldnames = ["Configuration", "Description", "P@5", "P@10", "R@5", "R@10", "F1@5", "MRR", "MAP"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"\n[+] Ablation results saved to: {csv_path}")

    # Generate Matplotlib chart if available
    chart_path = os.path.join(out_dir, "ablation_chart.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.5))
        labels = [c["name"].split(". ")[1] for c in configurations]
        x = range(len(labels))
        width = 0.2

        p5s = [r["P@5"] for r in ablation_rows]
        r5s = [r["R@5"] for r in ablation_rows]
        mrrs = [r["MRR"] for r in ablation_rows]
        maps = [r["MAP"] for r in ablation_rows]

        ax.bar([i - 1.5 * width for i in x], p5s, width=width, label="Precision@5", color="#2563eb")
        ax.bar([i - 0.5 * width for i in x], r5s, width=width, label="Recall@5", color="#10b981")
        ax.bar([i + 0.5 * width for i in x], mrrs, width=width, label="MRR", color="#f59e0b")
        ax.bar([i + 1.5 * width for i in x], maps, width=width, label="MAP", color="#8b5cf6")

        ax.set_title("IndicBridge: Ablation Study Across Pipeline Components", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9.5)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(chart_path, dpi=200)
        plt.close()
        print(f"[+] Ablation chart successfully saved to: {chart_path}")
    except Exception as e:
        print(f"[Notice] Matplotlib chart skipped ({e}). CSV data is complete.")

    print("=" * 75)
    return ablation_rows


if __name__ == "__main__":
    run_ablation()

#!/usr/bin/env python3
"""
CLI Search Bridge for IndicBridge
Executes queries from command line or subprocess and returns JSON.
Usage:
  python3 scripts/cli_search.py --query "kal ka mausam" --mode indicbridge --top_k 10
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="IndicBridge CLI Search Interface")
    parser.add_argument("--query", type=str, required=True, help="Search query string")
    parser.add_argument("--mode", type=str, default="indicbridge", choices=["indicbridge", "baseline", "compare"], help="Retriever mode")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K results to return")
    parser.add_argument("--alpha", type=float, default=None, help="TF-IDF Cosine weight")
    parser.add_argument("--beta", type=float, default=None, help="Transliteration weight")
    parser.add_argument("--gamma", type=float, default=None, help="Phonetic weight")
    parser.add_argument("--delta", type=float, default=None, help="Exact match weight")

    args = parser.parse_args()

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

    if args.mode == "baseline":
        retriever = BaselineRetriever(index, tfidf)
        output = retriever.search(args.query, top_k=args.top_k)
    elif args.mode == "indicbridge":
        retriever = IndicBridgeRetriever(index, tfidf, config)
        if args.alpha is not None or args.beta is not None or args.gamma is not None or args.delta is not None:
            a = args.alpha if args.alpha is not None else retriever.alpha
            b = args.beta if args.beta is not None else retriever.beta
            g = args.gamma if args.gamma is not None else retriever.gamma
            d = args.delta if args.delta is not None else retriever.delta
            retriever.set_weights(a, b, g, d)
        output = retriever.search(args.query, top_k=args.top_k)
    elif args.mode == "compare":
        base_ret = BaselineRetriever(index, tfidf)
        ind_ret = IndicBridgeRetriever(index, tfidf, config)
        if args.alpha is not None or args.beta is not None or args.gamma is not None or args.delta is not None:
            a = args.alpha if args.alpha is not None else ind_ret.alpha
            b = args.beta if args.beta is not None else ind_ret.beta
            g = args.gamma if args.gamma is not None else ind_ret.gamma
            d = args.delta if args.delta is not None else ind_ret.delta
            ind_ret.set_weights(a, b, g, d)
        base_res = base_ret.search(args.query, top_k=args.top_k)
        ind_res = ind_ret.search(args.query, top_k=args.top_k)
        output = {
            "query": args.query,
            "baseline": base_res,
            "indicbridge": ind_res,
        }

    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

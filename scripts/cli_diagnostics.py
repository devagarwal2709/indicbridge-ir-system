#!/usr/bin/env python3
"""
CLI Diagnostics Bridge for Inverted Index Postings & IDF
Usage:
  python3 scripts/cli_diagnostics.py --term "bharat"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.index.inverted_index import InvertedIndex
from src.index.tfidf_index import TFIDFIndex


def main():
    parser = argparse.ArgumentParser(description="IndicBridge Inverted Index Diagnostics")
    parser.add_argument("--term", type=str, required=True, help="Term to inspect")
    args = parser.parse_args()

    config = load_config()
    cache_dir = config["paths"]["index_dir"]

    index = InvertedIndex()
    index.load_index(cache_dir)
    tfidf = TFIDFIndex(index, use_sublinear_tf=config["index"]["use_sublinear_tf"])

    term = args.term.strip().lower()
    diag = index.get_diagnostics(term)
    diag["idf"] = round(tfidf.get_idf(term), 4)

    print(json.dumps(diag, ensure_ascii=False))


if __name__ == "__main__":
    main()

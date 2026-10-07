#!/usr/bin/env python3
"""
INDICBRIDGE UNIFIED CLI ENTRYPOINT
Command-line search, diagnostics, indexing, and evaluation tool.

Usage:
  python indicbridge.py search "kal ka mausam"
  python indicbridge.py search "कल का मौसम" --top-k 5 --explain
  python indicbridge.py compare "mera bharat"
  python indicbridge.py diag bharat
  python indicbridge.py eval
  python indicbridge.py ablation
  python indicbridge.py index
"""

import argparse
import json
import os
import sys

# Force UTF-8 encoding on Windows standard streams
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.utils.config import load_config
from src.index.inverted_index import InvertedIndex
from src.index.tfidf_index import TFIDFIndex
from src.retrieval.baseline import BaselineRetriever
from src.retrieval.hybrid_retriever import IndicBridgeRetriever

# Terminal ANSI colors
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RESET = "\033[0m"


def load_engines():
    config = load_config()
    cache_dir = config["paths"]["index_dir"]
    index_file = os.path.join(cache_dir, "inverted_index.json")

    if not os.path.exists(index_file):
        print(f"{C_YELLOW}[!] Index cache not found. Building index first...{C_RESET}")
        from scripts.build_index import build_and_save_index
        index, tfidf = build_and_save_index()
    else:
        index = InvertedIndex()
        index.load_index(cache_dir)
        tfidf = TFIDFIndex(index, use_sublinear_tf=config["index"]["use_sublinear_tf"])

    baseline = BaselineRetriever(index, tfidf)
    indicbridge = IndicBridgeRetriever(index, tfidf, config)
    return config, index, tfidf, baseline, indicbridge


def cmd_search(args):
    config, index, tfidf, baseline, indicbridge = load_engines()

    if args.alpha is not None or args.beta is not None or args.gamma is not None or args.delta is not None:
        a = args.alpha if args.alpha is not None else indicbridge.alpha
        b = args.beta if args.beta is not None else indicbridge.beta
        g = args.gamma if args.gamma is not None else indicbridge.gamma
        d = args.delta if args.delta is not None else indicbridge.delta
        indicbridge.set_weights(a, b, g, d)

    if args.mode == "baseline":
        res = baseline.search(args.query, top_k=args.top_k)
    else:
        res = indicbridge.search(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # Formatted terminal output
    print(f"\n{C_BOLD}{C_BLUE}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"{C_BOLD}🔍 INDICBRIDGE SEARCH RESULTS{C_RESET}")
    print(f"{C_BLUE}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"  {C_BOLD}Query:{C_RESET}               '{args.query}'")
    print(f"  {C_BOLD}Retriever Engine:{C_RESET}    {args.mode.upper()}")
    if args.mode == "indicbridge":
        print(f"  {C_BOLD}Detected Script:{C_RESET}     {res.get('detected_script')} ({res.get('detected_language')})")
        if res.get('transliterated_query'):
            print(f"  {C_BOLD}Transliteration:{C_RESET}     '{res.get('transliterated_query')}'")
        timing = res.get('timing_ms', {})
        print(f"  {C_BOLD}Latency:{C_RESET}             {timing.get('total', 0)} ms (Candidate search: {timing.get('candidate_retrieval', 0)} ms)")

    results = res.get("results", [])
    print(f"  {C_BOLD}Retrieved Count:{C_RESET}     {len(results)} documents")
    print(f"{C_BLUE}──────────────────────────────────────────────────────────────────────{C_RESET}\n")

    if not results:
        print(f"{C_YELLOW}  No matching documents found.{C_RESET}\n")
        return

    for item in results:
        lang_color = C_GREEN if item['language'] == 'hi' else (C_CYAN if item['language'] == 'en' else C_YELLOW)
        lang_label = "HINDI" if item['language'] == 'hi' else ("ENGLISH" if item['language'] == 'en' else "HINGLISH")

        print(f"  {C_BOLD}#{item['rank']}{C_RESET} [{lang_color}{lang_label}{C_RESET}] {C_BOLD}{item['title']}{C_RESET} {C_DIM}(ID: {item['doc_id']}){C_RESET}")
        print(f"     {C_BOLD}Final Score:{C_RESET} {C_GREEN}{item['final_score']:.4f}{C_RESET}  {C_DIM}| Cosine: {item.get('cosine_score', 0):.4f} | Translit: {item.get('transliteration_score', 0):.4f} | Phonetic: {item.get('phonetic_score', 0):.4f}{C_RESET}")
        print(f"     {C_DIM}Matched terms: [{', '.join(item.get('matched_terms', []))}]{C_RESET}")
        print(f"     \"{item['text'][:120]}...\"")

        if args.explain and item.get("explanation_summary"):
            print(f"     {C_CYAN}💡 Rationale:{C_RESET} {item['explanation_summary']}")
        print()


def cmd_compare(args):
    config, index, tfidf, baseline, indicbridge = load_engines()

    base_res = baseline.search(args.query, top_k=args.top_k)
    ind_res = indicbridge.search(args.query, top_k=args.top_k)

    if args.json:
        print(json.dumps({"query": args.query, "baseline": base_res, "indicbridge": ind_res}, ensure_ascii=False, indent=2))
        return

    print(f"\n{C_BOLD}{C_BLUE}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"{C_BOLD}⚖️  HEAD-TO-HEAD COMPARISON: Baseline vs. IndicBridge{C_RESET}")
    print(f"{C_BLUE}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"Query: '{args.query}'\n")

    base_docs = {r['doc_id']: r for r in base_res['results']}
    ind_docs = {r['doc_id']: r for r in ind_res['results']}

    additional = [doc_id for doc_id in ind_docs if doc_id not in base_docs]

    print(f"{C_BOLD}{'Rank':<5} | {'Baseline (Exact TF-IDF)':<30} | {'IndicBridge (Multilingual)':<30}{C_RESET}")
    print("─" * 70)

    max_len = max(len(base_res['results']), len(ind_res['results']), 1)
    for i in range(max_len):
        b_str = "-"
        i_str = "-"
        if i < len(base_res['results']):
            b_item = base_res['results'][i]
            b_str = f"{b_item['doc_id']} ({b_item['final_score']:.3f}) {b_item['title'][:16]}"
        if i < len(ind_res['results']):
            i_item = ind_res['results'][i]
            is_new = f"{C_GREEN}* {C_RESET}" if i_item['doc_id'] in additional else "  "
            i_str = f"{is_new}{i_item['doc_id']} ({i_item['final_score']:.3f}) {i_item['title'][:16]}"

        print(f"#{i+1:<4} | {b_str:<30} | {i_str:<30}")

    print("─" * 70)
    print(f"{C_BOLD}Retrieval Delta:{C_RESET} IndicBridge retrieved {C_GREEN}{len(additional)} additional{C_RESET} relevant document(s) missed by Baseline: {additional}\n")


def cmd_diag(args):
    config, index, tfidf, baseline, indicbridge = load_engines()
    term = args.term.strip().lower()
    diag = index.get_diagnostics(term)
    idf_val = tfidf.get_idf(term)

    if args.json:
        diag["idf"] = round(idf_val, 4)
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        return

    print(f"\n{C_BOLD}{C_BLUE}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"{C_BOLD}🔬 INVERTED INDEX POSTINGS DIAGNOSTICS: '{term}'{C_RESET}")
    print(f"{C_BLUE}══════════════════════════════════════════════════════════════════════{C_RESET}")
    print(f"  {C_BOLD}Document Frequency (DF):{C_RESET} {diag['document_frequency']} / {diag['collection_size']} documents")
    print(f"  {C_BOLD}Smoothed IDF Value:{C_RESET}      {idf_val:.4f} [ln(({diag['collection_size']}+1)/({diag['document_frequency']}+1)) + 1]")
    print(f"{C_BLUE}──────────────────────────────────────────────────────────────────────{C_RESET}")

    if not diag["postings"]:
        print(f"{C_YELLOW}  Term '{term}' is not present in collection vocabulary.{C_RESET}\n")
        return

    print(f"\n  {C_BOLD}{'Doc ID':<10} | {'Lang':<8} | {'TF':<5} | {'Title':<25} | {'Positions'}{C_RESET}")
    print("  " + "─" * 66)
    for p in diag["postings"]:
        pos_preview = f"{p['positions'][:5]}" + ("..." if len(p['positions']) > 5 else "")
        print(f"  {p['doc_id']:<10} | {p['language']:<8} | {p['tf']:<5} | {p['title'][:25]:<25} | {pos_preview}")
    print()


def cmd_eval(args):
    from evaluation.evaluate import run_evaluation
    run_evaluation()


def cmd_ablation(args):
    from evaluation.ablation import run_ablation
    run_ablation()


def cmd_index(args):
    from scripts.build_index import build_and_save_index
    build_and_save_index()


def main():
    parser = argparse.ArgumentParser(
        description="INDICBRIDGE: Explainable Multilingual Information Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Subcommand: search
    p_search = subparsers.add_parser("search", help="Execute search query")
    p_search.add_argument("query", type=str, help="Search query string")
    p_search.add_argument("--mode", type=str, default="indicbridge", choices=["indicbridge", "baseline"], help="Retrieval mode")
    p_search.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    p_search.add_argument("--explain", action="store_true", help="Print explanation details for results")
    p_search.add_argument("--alpha", type=float, default=None, help="TF-IDF Cosine weight")
    p_search.add_argument("--beta", type=float, default=None, help="Transliteration weight")
    p_search.add_argument("--gamma", type=float, default=None, help="Phonetic weight")
    p_search.add_argument("--delta", type=float, default=None, help="Exact match weight")
    p_search.add_argument("--json", action="store_true", help="Output raw JSON")

    # Subcommand: compare
    p_compare = subparsers.add_parser("compare", help="Side-by-side comparison between Baseline and IndicBridge")
    p_compare.add_argument("query", type=str, help="Search query string")
    p_compare.add_argument("--top-k", type=int, default=10, help="Top-K results")
    p_compare.add_argument("--json", action="store_true", help="Output raw JSON")

    # Subcommand: diag
    p_diag = subparsers.add_parser("diag", help="Inspect inverted index postings and IDF for a term")
    p_diag.add_argument("term", type=str, help="Vocabulary term to inspect")
    p_diag.add_argument("--json", action="store_true", help="Output raw JSON")

    # Subcommand: eval
    subparsers.add_parser("eval", help="Run benchmark evaluation over all 24 ground-truth queries")

    # Subcommand: ablation
    subparsers.add_parser("ablation", help="Run 5-stage component ablation experiment")

    # Subcommand: index
    subparsers.add_parser("index", help="Rebuild inverted index from data/documents.jsonl")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    if args.subcommand == "search":
        cmd_search(args)
    elif args.subcommand == "compare":
        cmd_compare(args)
    elif args.subcommand == "diag":
        cmd_diag(args)
    elif args.subcommand == "eval":
        cmd_eval(args)
    elif args.subcommand == "ablation":
        cmd_ablation(args)
    elif args.subcommand == "index":
        cmd_index(args)


if __name__ == "__main__":
    main()

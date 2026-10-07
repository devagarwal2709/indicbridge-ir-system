#!/usr/bin/env python3
"""
IndicBridge Index Builder
Builds the classical inverted index and precomputes TF-IDF vector representations.
"""

import os
import sys
import time
from typing import Tuple

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.io import load_documents
from src.preprocessing.tokenizer import tokenize
from src.preprocessing.normalizer import normalize_text
from src.preprocessing.transliterator import devanagari_to_roman, roman_to_devanagari
from src.preprocessing.language_detector import detect_script
from src.index.inverted_index import InvertedIndex
from src.index.tfidf_index import TFIDFIndex


def build_and_save_index(config_path: str = "config/config.yaml") -> Tuple[InvertedIndex, TFIDFIndex]:
    print("=" * 60)
    print("INDICBRIDGE: Building Inverted Index & TF-IDF Vectors")
    print("=" * 60)

    start_time = time.time()
    config = load_config(config_path)

    doc_path = config["paths"]["documents"]
    cache_dir = config["paths"]["index_dir"]

    print(f"[*] Loading documents from: {doc_path}")
    documents = load_documents(doc_path)
    print(f"[+] Loaded {len(documents)} documents.")

    index = InvertedIndex()

    for doc in documents:
        doc_id = doc["id"]
        title = doc.get("title", "")
        text = doc.get("text", "")
        lang = doc.get("language", "hi")
        full_content = f"{title} {text}"

        # 1. Base tokens
        norm_text = normalize_text(
            full_content,
            fold_devanagari_nukta=True,
            lowercase_latin=True,
            normalize_romanized=False,
        )
        base_tokens = tokenize(norm_text)

        # 2. Canonical normalized tokens
        canon_text = normalize_text(
            full_content,
            fold_devanagari_nukta=True,
            lowercase_latin=True,
            normalize_romanized=True,
        )
        canon_tokens = tokenize(canon_text)

        # 3. Cross-script dual indexing:
        # For Hindi documents, also index Romanized equivalents
        # For English/Romanized documents, also index Devanagari transliteration
        cross_tokens = []
        script = detect_script(full_content)
        if script == "Devanagari":
            roman_rep = devanagari_to_roman(norm_text)
            cross_tokens = tokenize(roman_rep)
        elif script in ("Latin", "Mixed"):
            dev_rep = roman_to_devanagari(norm_text)
            cross_tokens = tokenize(dev_rep)

        # Combine all tokens for comprehensive inverted indexing
        all_doc_tokens = base_tokens + [t for t in canon_tokens if t not in base_tokens]
        if cross_tokens:
            all_doc_tokens += cross_tokens

        metadata = {
            "title": title,
            "language": lang,
            "text": text,
            "url": doc.get("url", ""),
            "source": doc.get("source", ""),
        }

        index.add_document(doc_id=doc_id, tokens=all_doc_tokens, metadata=metadata)

    vocab_size = len(index.get_vocabulary())
    print(f"[+] Indexed {index.num_docs} documents.")
    print(f"[+] Total Vocabulary Size: {vocab_size} unique terms.")

    print(f"[*] Building TF-IDF Vector Space Model...")
    tfidf = TFIDFIndex(index, use_sublinear_tf=config["index"]["use_sublinear_tf"])
    print(f"[+] TF-IDF model initialized with {len(tfidf.idf_cache)} cached IDF terms.")

    print(f"[*] Saving inverted index to: {cache_dir}")
    index.save_index(cache_dir)
    print(f"[+] Index successfully saved to {cache_dir}/inverted_index.json")

    elapsed = time.time() - start_time
    print(f"[✓] Indexing completed in {elapsed:.3f} seconds.")
    print("=" * 60)

    return index, tfidf


if __name__ == "__main__":
    build_and_save_index()

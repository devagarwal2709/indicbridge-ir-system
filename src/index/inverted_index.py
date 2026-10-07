import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class PostingEntry:
    """Represents an entry in a postings list."""
    def __init__(self, doc_id: str, tf: int, positions: Optional[List[int]] = None):
        self.doc_id = doc_id
        self.tf = tf
        self.positions = positions or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "tf": self.tf,
            "positions": self.positions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostingEntry":
        return cls(
            doc_id=data["doc_id"],
            tf=data["tf"],
            positions=data.get("positions", []),
        )

    def __repr__(self) -> str:
        return f"PostingEntry({self.doc_id}, tf={self.tf})"


class InvertedIndex:
    """
    Classical Inverted Index.
    Maps term -> List[PostingEntry]
    Tracks document frequencies, document lengths, and collection vocabulary.
    """
    def __init__(self):
        # term -> List[PostingEntry]
        self.index: Dict[str, List[PostingEntry]] = defaultdict(list)
        # doc_id -> total token count in document
        self.doc_lengths: Dict[str, int] = {}
        # doc_id -> metadata dictionary (title, language, url, etc.)
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
        # Total number of documents indexed
        self.num_docs: int = 0
        # doc_id -> list of all tokens (for phonetic and exact analysis)
        self.doc_tokens: Dict[str, List[str]] = {}

    def add_document(
        self,
        doc_id: str,
        tokens: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document's token stream to the inverted index."""
        if not tokens:
            return

        self.doc_lengths[doc_id] = len(tokens)
        self.doc_metadata[doc_id] = metadata or {}
        self.doc_tokens[doc_id] = tokens
        self.num_docs += 1

        # Count term frequencies and track positions
        term_positions: Dict[str, List[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            term_positions[term].append(pos)

        # Update postings list
        for term, positions in term_positions.items():
            self.index[term].append(
                PostingEntry(
                    doc_id=doc_id,
                    tf=len(positions),
                    positions=positions,
                )
            )

    def get_postings(self, term: str) -> List[PostingEntry]:
        """Retrieve postings list for a term."""
        return self.index.get(term.lower().strip(), [])

    def get_document_frequency(self, term: str) -> int:
        """Get document frequency (DF) for a term."""
        return len(self.get_postings(term))

    def get_vocabulary(self) -> Set[str]:
        """Return the set of all unique indexed vocabulary terms."""
        return set(self.index.keys())

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """Get term frequency (TF) of a term in a specific document."""
        for entry in self.get_postings(term):
            if entry.doc_id == doc_id:
                return entry.tf
        return 0

    def get_candidate_documents(self, query_terms: List[str]) -> Set[str]:
        """
        Efficient candidate retrieval:
        Returns all document IDs containing at least one query term.
        Avoids computing similarity over non-relevant documents.
        """
        candidates: Set[str] = set()
        for term in query_terms:
            for entry in self.get_postings(term):
                candidates.add(entry.doc_id)
        return candidates

    def get_diagnostics(self, term: str) -> Dict[str, Any]:
        """
        Detailed IR diagnostics for a term (Section 19):
        Displays Term, Document Frequency, and Postings breakdown.
        """
        clean_term = term.lower().strip()
        postings = self.get_postings(clean_term)
        return {
            "term": clean_term,
            "document_frequency": len(postings),
            "collection_size": self.num_docs,
            "postings": [
                {
                    "doc_id": p.doc_id,
                    "title": self.doc_metadata.get(p.doc_id, {}).get("title", ""),
                    "language": self.doc_metadata.get(p.doc_id, {}).get("language", ""),
                    "tf": p.tf,
                    "positions": p.positions[:10],  # first 10 positions
                }
                for p in postings
            ],
        }

    def save_index(self, directory: str) -> None:
        """Serialize index to disk."""
        os.makedirs(directory, exist_ok=True)
        serialized_index = {
            term: [entry.to_dict() for entry in entries]
            for term, entries in self.index.items()
        }
        data = {
            "num_docs": self.num_docs,
            "doc_lengths": self.doc_lengths,
            "doc_metadata": self.doc_metadata,
            "doc_tokens": self.doc_tokens,
            "index": serialized_index,
        }
        with open(os.path.join(directory, "inverted_index.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load_index(self, directory: str) -> None:
        """Deserialize index from disk."""
        index_file = os.path.join(directory, "inverted_index.json")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Inverted index file not found at: {index_file}")

        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.num_docs = data["num_docs"]
        self.doc_lengths = data["doc_lengths"]
        self.doc_metadata = data["doc_metadata"]
        self.doc_tokens = data.get("doc_tokens", {})
        self.index = defaultdict(list)

        for term, entries_data in data["index"].items():
            self.index[term] = [PostingEntry.from_dict(d) for d in entries_data]

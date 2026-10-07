from typing import List, Set


def precision_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """
    Compute Precision@K:
    Number of relevant documents in top K / K
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0

    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return round(relevant_retrieved / len(top_k), 4)


def recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """
    Compute Recall@K:
    Number of relevant documents in top K / Total relevant documents
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return round(relevant_retrieved / len(relevant_doc_ids), 4)


def f1_score_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Compute F1 score at rank K."""
    p = precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
    r = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
    if (p + r) == 0.0:
        return 0.0
    return round(2.0 * (p * r) / (p + r), 4)


def reciprocal_rank(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """
    Compute Reciprocal Rank:
    1 / rank of first relevant document retrieved, or 0 if none retrieved.
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return round(1.0 / rank, 4)
    return 0.0


def average_precision(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """
    Compute Average Precision (AP) for a query:
    Mean of precision at each position where a relevant document is retrieved.
    """
    if not relevant_doc_ids:
        return 0.0

    num_relevant_seen = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            num_relevant_seen += 1
            precision_sum += num_relevant_seen / rank

    if num_relevant_seen == 0:
        return 0.0

    return round(precision_sum / len(relevant_doc_ids), 4)

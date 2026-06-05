"""Pure ranking metrics for recommender evaluation."""

from __future__ import annotations

import math


def precision_at_k(relevances: list[int], k: int = 5, threshold: int = 1) -> float:
    """Return the fraction of top-k items with relevance >= threshold."""
    top_relevances = _top_k(relevances, k)
    if not top_relevances:
        return 0.0

    relevant_count = sum(1 for relevance in top_relevances if relevance >= threshold)
    return relevant_count / len(top_relevances)


def dcg_at_k(relevances: list[int], k: int = 5) -> float:
    """Return discounted cumulative gain for the first k relevance grades."""
    top_relevances = _top_k(relevances, k)
    return sum(
        ((2**relevance) - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(top_relevances, start=1)
    )


def ndcg_at_k(relevances: list[int], k: int = 5) -> float:
    """Return normalized discounted cumulative gain for the first k results."""
    top_relevances = _top_k(relevances, k)
    if not top_relevances:
        return 0.0

    ideal_relevances = sorted(top_relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k=len(ideal_relevances))
    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(top_relevances, k=len(top_relevances)) / ideal_dcg


def reciprocal_rank(relevances: list[int], threshold: int = 1) -> float:
    """Return reciprocal rank of the first result with relevance >= threshold."""
    for index, relevance in enumerate(relevances, start=1):
        if relevance >= threshold:
            return 1 / index

    return 0.0


def mean_reciprocal_rank(all_relevances: list[list[int]], threshold: int = 1) -> float:
    """Return mean reciprocal rank across multiple ranked relevance lists."""
    if not all_relevances:
        return 0.0

    return sum(
        reciprocal_rank(relevances, threshold=threshold)
        for relevances in all_relevances
    ) / len(all_relevances)


def _top_k(relevances: list[int], k: int) -> list[int]:
    if k <= 0:
        return []

    return [int(relevance) for relevance in relevances[:k]]

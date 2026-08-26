import re
from typing import List, Dict, Any, Callable

from src.query_rewriter import rewrite_query
from src.retrieval_confidence import assess_retrieval_confidence


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
    "and", "or", "how", "does", "do", "these", "this", "that", "with",
    "from", "by", "as", "it", "be", "was", "were", "about", "according",
    "explain", "describe", "discuss", "main", "role", "purpose"
}


def confidence_to_dict(confidence):
    return {
        "label": confidence.label,
        "should_answer": confidence.should_answer,
        "should_retry": confidence.should_retry,
        "reason": confidence.reason,
        "top_rerank_score": confidence.top_rerank_score,
        "avg_rerank_score": confidence.avg_rerank_score,
        "keyword_coverage": confidence.keyword_coverage,
    }


def build_keyword_query(query: str) -> str:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", query.lower())
    keywords = []

    for word in words:
        if word not in STOPWORDS and word not in keywords:
            keywords.append(word)

    if not keywords:
        return query

    return " ".join(keywords)


def retrieve_with_retries(
    original_query: str,
    retriever,
    reranker,
    rerank_function: Callable,
    max_attempts: int = 3,
    retrieve_top_k: int = 12,
    rerank_top_k: int = 4,
):
    """
    Retrieval guardrail.

    Attempts:
    1. Original query
    2. Rewritten query
    3. Keyword-only query

    If all attempts fail confidence criteria, return should_answer=False.
    """

    attempted_queries = [
        original_query,
        rewrite_query(original_query),
        build_keyword_query(original_query),
    ]

    attempts = []
    best_results = []
    best_confidence = None
    best_query = original_query

    for index, attempt_query in enumerate(attempted_queries[:max_attempts], start=1):
        candidate_results = retriever.retrieve_evidence(
            query=attempt_query,
            top_k=retrieve_top_k,
        )

        reranked_results = rerank_function(
            query=attempt_query,
            candidate_chunks=candidate_results,
            reranker=reranker,
            top_k=rerank_top_k,
        )

        confidence = assess_retrieval_confidence(
            query=original_query,
            reranked_chunks=reranked_results,
        )

        attempts.append(
            {
                "attempt": index,
                "query": attempt_query,
                "confidence": confidence_to_dict(confidence),
            }
        )

        if best_confidence is None:
            best_results = reranked_results
            best_confidence = confidence
            best_query = attempt_query
        else:
            if confidence.top_rerank_score > best_confidence.top_rerank_score:
                best_results = reranked_results
                best_confidence = confidence
                best_query = attempt_query

        if confidence.should_answer:
            return {
                "should_answer": True,
                "results": reranked_results,
                "confidence": confidence,
                "attempts": attempts,
                "used_query": attempt_query,
            }

    return {
        "should_answer": False,
        "results": best_results,
        "confidence": best_confidence,
        "attempts": attempts,
        "used_query": best_query,
    }
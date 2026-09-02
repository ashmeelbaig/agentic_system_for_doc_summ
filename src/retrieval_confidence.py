import re
from dataclasses import dataclass
from typing import List, Dict, Any


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
    "and", "or", "how", "does", "do", "these", "this", "that", "with",
    "from", "by", "as", "it", "be", "was", "were", "about", "according"
}


@dataclass
class RetrievalConfidence:
    label: str
    should_answer: bool
    should_retry: bool
    reason: str
    top_rerank_score: float
    avg_rerank_score: float
    keyword_coverage: float


def _keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def _keyword_coverage(query: str, evidence_text: str) -> float:
    query_terms = set(_keywords(query))
    if not query_terms:
        return 0.0

    evidence_terms = set(_keywords(evidence_text))
    matched = query_terms.intersection(evidence_terms)

    return len(matched) / len(query_terms)


def assess_retrieval_confidence(
    query: str,
    reranked_chunks: List[Dict[str, Any]],
    min_top_score: float = 1.5,
    min_avg_score: float = 1.0,
    min_keyword_coverage: float = 0.25,
) -> RetrievalConfidence:
    """
    Efficient retrieval confidence check.

    It does not call an LLM.
    It uses:
    - top rerank score
    - average rerank score
    - keyword coverage between question and retrieved evidence
    """

    if not reranked_chunks:
        return RetrievalConfidence(
            label="low",
            should_answer=False,
            should_retry=True,
            reason="No retrieved evidence was available.",
            top_rerank_score=0.0,
            avg_rerank_score=0.0,
            keyword_coverage=0.0,
        )

    rerank_scores = [
        float(chunk.get("rerank_score", 0.0))
        for chunk in reranked_chunks
    ]

    top_rerank_score = max(rerank_scores)
    avg_rerank_score = sum(rerank_scores) / len(rerank_scores)

    combined_evidence = " ".join(
        str(chunk.get("text", ""))
        for chunk in reranked_chunks
    )

    keyword_coverage = _keyword_coverage(query, combined_evidence)

    weak_score = top_rerank_score < min_top_score and avg_rerank_score < min_avg_score
    weak_keywords = keyword_coverage < min_keyword_coverage

    if weak_score and weak_keywords:
        return RetrievalConfidence(
            label="low",
            should_answer=False,
            should_retry=True,
            reason="Retrieved evidence has low rerank confidence and weak keyword coverage.",
            top_rerank_score=top_rerank_score,
            avg_rerank_score=avg_rerank_score,
            keyword_coverage=keyword_coverage,
        )

    if weak_score or weak_keywords:
        return RetrievalConfidence(
            label="medium",
            should_answer=True,
            should_retry=False,
            reason="Retrieved evidence is usable but not very strong.",
            top_rerank_score=top_rerank_score,
            avg_rerank_score=avg_rerank_score,
            keyword_coverage=keyword_coverage,
        )

    return RetrievalConfidence(
        label="high",
        should_answer=True,
        should_retry=False,
        reason="Retrieved evidence appears strong enough for answer generation.",
        top_rerank_score=top_rerank_score,
        avg_rerank_score=avg_rerank_score,
        keyword_coverage=keyword_coverage,
    )
from src.retrieval_retry import retrieve_with_retries


class FakeRetriever:
    def __init__(self, chunks_per_attempt):
        self.chunks_per_attempt = chunks_per_attempt
        self.call_count = 0

    def retrieve_evidence(self, query, top_k):
        result = self.chunks_per_attempt[self.call_count]
        self.call_count += 1
        return result


def fake_rerank_function(query, candidate_chunks, reranker, top_k):
    return candidate_chunks[:top_k]


WEAK = [{"text": "Unrelated material about penguins.", "rerank_score": 0.1}]
STRONG = [
    {"text": "Photosynthesis uses chlorophyll to capture sunlight.", "rerank_score": 2.0},
    {"text": "Sunlight powers photosynthesis in plants.", "rerank_score": 1.8},
]
QUERY = "How does photosynthesis use sunlight?"


def run_retry(chunks_per_attempt):
    retriever = FakeRetriever(chunks_per_attempt)
    result = retrieve_with_retries(
        original_query=QUERY,
        retriever=retriever,
        reranker=object(),
        rerank_function=fake_rerank_function,
    )
    return result, retriever


def test_strong_first_attempt_stops_immediately():
    result, retriever = run_retry([STRONG])

    assert result["should_answer"] is True
    assert len(result["attempts"]) == 1
    assert retriever.call_count == 1
    assert result["used_query"] == QUERY


def test_later_strong_attempt_returns_its_results_and_query():
    result, retriever = run_retry([WEAK, STRONG, WEAK])

    assert result["should_answer"] is True
    assert result["results"] == STRONG
    assert len(result["attempts"]) == 2
    assert retriever.call_count == 2
    assert result["used_query"] == result["attempts"][1]["query"]


def test_all_three_weak_attempts_refuse_and_record_every_attempt():
    result, retriever = run_retry([WEAK, WEAK, WEAK])

    assert result["should_answer"] is False
    assert len(result["attempts"]) == 3
    assert retriever.call_count == 3


def test_result_contains_guardrail_contract_keys():
    result, _ = run_retry([STRONG])

    assert {"results", "confidence", "attempts", "used_query"} <= result.keys()

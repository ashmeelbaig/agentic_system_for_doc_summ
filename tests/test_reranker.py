from src.reranker import EvidenceReranker


class DummyCrossEncoder:
    def predict(self, pairs):
        scores = []

        for query, text in pairs:
            if "important" in text:
                scores.append(0.95)
            else:
                scores.append(0.10)

        return scores


def test_reranker_orders_chunks_by_score():
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "This is less relevant text.",
        },
        {
            "chunk_id": "chunk_2",
            "text": "This is important evidence.",
        },
    ]

    reranker = EvidenceReranker(model=DummyCrossEncoder())

    results = reranker.rerank(
        query="What is important?",
        retrieved_chunks=chunks,
        top_k=2,
    )

    assert results[0]["chunk_id"] == "chunk_2"
    assert results[0]["rerank_score"] == 0.95
    assert results[0]["original_rank"] == 2


def test_reranker_returns_empty_list_for_no_chunks():
    reranker = EvidenceReranker(model=DummyCrossEncoder())

    results = reranker.rerank(
        query="test query",
        retrieved_chunks=[],
    )

    assert results == []


def test_reranker_rejects_empty_query():
    reranker = EvidenceReranker(model=DummyCrossEncoder())

    try:
        reranker.rerank(
            query="",
            retrieved_chunks=[{"text": "some text"}],
        )
        assert False
    except ValueError:
        assert True
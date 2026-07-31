from main import rerank_candidate_evidence


class DummyReranker:
    def rerank(self, query, retrieved_chunks, top_k=4):
        reranked = []

        for item in retrieved_chunks:
            updated_item = dict(item)

            if "claim verification" in updated_item["text"].lower():
                updated_item["rerank_score"] = 0.95
            else:
                updated_item["rerank_score"] = 0.20

            reranked.append(updated_item)

        reranked = sorted(
            reranked,
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]


def test_rerank_candidate_evidence_returns_top_reranked_chunks():
    candidate_chunks = [
        {
            "chunk_id": "doc_p1_c0",
            "source": "doc.pdf",
            "page_number": 1,
            "text": "This chunk is about document loading.",
            "score": 0.88,
        },
        {
            "chunk_id": "doc_p2_c0",
            "source": "doc.pdf",
            "page_number": 2,
            "text": "This chunk explains claim verification using evidence.",
            "score": 0.72,
        },
    ]

    results = rerank_candidate_evidence(
        query="How does claim verification work?",
        candidate_chunks=candidate_chunks,
        reranker=DummyReranker(),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "doc_p2_c0"
    assert results[0]["rerank_score"] == 0.95
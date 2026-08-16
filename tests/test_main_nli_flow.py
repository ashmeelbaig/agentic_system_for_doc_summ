from main import verify_claims_with_selected_verifier


class DummyVerifier:
    def verify_claims(self, claims, retrieved_chunks):
        return [
            {
                "claim": claims[0],
                "label": "Supported",
                "nli_label": "ENTAILMENT",
                "nli_score": 0.94,
                "evidence": retrieved_chunks[0]["text"],
                "chunk_id": retrieved_chunks[0]["chunk_id"],
                "source": retrieved_chunks[0]["source"],
                "page_number": retrieved_chunks[0]["page_number"],
            }
        ]


def test_verify_claims_with_selected_verifier():
    claims = ["The system uses FAISS for retrieval."]

    retrieved_chunks = [
        {
            "chunk_id": "doc_p1_c0",
            "source": "doc.pdf",
            "page_number": 1,
            "text": "The system uses FAISS for retrieval.",
            "score": 0.88,
            "rerank_score": 0.95,
        }
    ]

    results = verify_claims_with_selected_verifier(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
        verifier=DummyVerifier(),
    )

    assert len(results) == 1
    assert results[0]["label"] == "Supported"
    assert results[0]["nli_label"] == "ENTAILMENT"
    assert results[0]["chunk_id"] == "doc_p1_c0"
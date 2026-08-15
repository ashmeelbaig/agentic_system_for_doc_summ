from src.nli_verifier import NLIClaimVerifier


class DummyNLIPipeline:
    def __call__(self, inputs, truncation=True):
        results = []

        for item in inputs:
            text = item.lower()

            if "does not use faiss" in text:
                results.append({"label": "CONTRADICTION", "score": 0.91})
            elif "uses faiss" in text:
                results.append({"label": "ENTAILMENT", "score": 0.94})
            else:
                results.append({"label": "NEUTRAL", "score": 0.76})

        return results


def test_nli_verifier_labels_supported_claim():
    verifier = NLIClaimVerifier(nli_pipeline=DummyNLIPipeline())

    claims = ["The system uses FAISS for retrieval."]

    retrieved_chunks = [
        {
            "chunk_id": "doc_p1_c0",
            "source": "doc.pdf",
            "page_number": 1,
            "text": "The system uses FAISS for retrieval and stores embeddings in a vector index.",
            "score": 0.88,
            "rerank_score": 0.95,
        }
    ]

    results = verifier.verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
    )

    assert len(results) == 1
    assert results[0]["label"] == "Supported"
    assert results[0]["nli_label"] == "ENTAILMENT"
    assert results[0]["chunk_id"] == "doc_p1_c0"
    assert results[0]["source"] == "doc.pdf"
    assert results[0]["page_number"] == 1


def test_nli_verifier_labels_contradicted_claim():
    verifier = NLIClaimVerifier(nli_pipeline=DummyNLIPipeline())

    claims = ["The system does not use FAISS for retrieval."]

    retrieved_chunks = [
        {
            "chunk_id": "doc_p1_c0",
            "source": "doc.pdf",
            "page_number": 1,
            "text": "The system uses FAISS for retrieval and stores embeddings in a vector index.",
            "score": 0.88,
            "rerank_score": 0.95,
        }
    ]

    results = verifier.verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
    )

    assert len(results) == 1
    assert results[0]["label"] == "Contradicted"
    assert results[0]["nli_label"] == "CONTRADICTION"


def test_nli_verifier_labels_not_enough_evidence():
    verifier = NLIClaimVerifier(nli_pipeline=DummyNLIPipeline())

    claims = ["The system supports audio input."]

    retrieved_chunks = [
        {
            "chunk_id": "doc_p2_c0",
            "source": "doc.pdf",
            "page_number": 2,
            "text": "The system currently supports PDF text extraction and metadata-aware chunking.",
            "score": 0.70,
            "rerank_score": 0.80,
        }
    ]

    results = verifier.verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
    )

    assert len(results) == 1
    assert results[0]["label"] == "Not enough evidence"
    assert results[0]["nli_label"] == "NEUTRAL"


def test_nli_verifier_returns_empty_list_for_no_claims():
    verifier = NLIClaimVerifier(nli_pipeline=DummyNLIPipeline())

    results = verifier.verify_claims(
        claims=[],
        retrieved_chunks=[],
    )

    assert results == []
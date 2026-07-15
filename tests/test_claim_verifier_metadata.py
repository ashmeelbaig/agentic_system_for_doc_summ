import numpy as np

from src.claim_verifier import ClaimVerifier


class DummyVerifierModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []

        for text in texts:
            text = text.lower()

            if "faiss" in text or "vector search" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return np.array(vectors, dtype="float32")


def test_prepare_evidence_sentences_supports_metadata_chunks():
    verifier = ClaimVerifier.__new__(ClaimVerifier)
    verifier.supported_threshold = 0.55
    verifier.partial_threshold = 0.40

    retrieved_chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "The system uses FAISS vector search to retrieve relevant evidence from technical documents.",
            "score": 0.91,
        }
    ]

    evidence_items = verifier._prepare_evidence_sentences(retrieved_chunks)

    assert len(evidence_items) == 1
    assert evidence_items[0]["chunk_id"] == "sample_p1_c0"
    assert evidence_items[0]["source"] == "sample.pdf"
    assert evidence_items[0]["page_number"] == 1
    assert evidence_items[0]["retrieval_score"] == 0.91
    assert "FAISS vector search" in evidence_items[0]["sentence"]


def test_verify_claims_returns_metadata_evidence_reference():
    verifier = ClaimVerifier(
        embedding_model=DummyVerifierModel(),
        supported_threshold=0.55,
        partial_threshold=0.40,
    )

    claims = [
        "The system uses FAISS vector search to retrieve evidence."
    ]

    retrieved_chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "The system uses FAISS vector search to retrieve relevant evidence from technical documents.",
            "score": 0.91,
        }
    ]

    results = verifier.verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
    )

    assert len(results) == 1
    assert results[0]["label"] == "Supported"
    assert results[0]["chunk_id"] == "sample_p1_c0"
    assert results[0]["source"] == "sample.pdf"
    assert results[0]["page_number"] == 1
    assert "FAISS vector search" in results[0]["evidence"]
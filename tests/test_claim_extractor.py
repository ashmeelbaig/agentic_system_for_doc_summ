from src.claim_extractor import extract_claims


def test_extract_claims_from_answer():
    answer = (
        "The system uses FAISS for semantic retrieval. "
        "The generated answer is verified against retrieved evidence chunks."
    )

    claims = extract_claims(answer)

    assert isinstance(claims, list)
    assert len(claims) >= 1
    assert any("FAISS" in claim for claim in claims)
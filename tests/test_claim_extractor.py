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


def test_compound_risk_claim_is_split_into_atomic_claims():
    answer = (
        "GenAI systems have risks from adversarial attacks, misuse of foundation "
        "model capabilities, and privacy issues."
    )

    claims = extract_claims(answer)

    assert claims == [
        "GenAI systems have risks from adversarial attacks.",
        "GenAI systems have risks from misuse of foundation model capabilities.",
        "GenAI systems have privacy risks.",
    ]


def test_observed_compound_claim_splits_each_risk_type():
    answer = (
        "The retrieved documents discuss risks associated with generative AI systems, "
        "including risks from adversarial attacks, misuse of foundation model "
        "capabilities, and potential negative impacts on humans."
    )

    claims = extract_claims(answer)

    assert len(claims) == 3
    assert any("risks from adversarial attacks" in claim for claim in claims)
    assert any("risks from misuse of foundation model capabilities" in claim for claim in claims)
    assert any("potential negative impacts on humans" in claim for claim in claims)

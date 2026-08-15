from src.scoring import calculate_faithfulness_score


def test_faithfulness_score_supports_nli_labels():
    verification_results = [
        {
            "claim": "Claim 1",
            "label": "Supported",
        },
        {
            "claim": "Claim 2",
            "label": "Contradicted",
        },
        {
            "claim": "Claim 3",
            "label": "Not enough evidence",
        },
    ]

    score = calculate_faithfulness_score(verification_results)

    assert score["total_claims"] == 3
    assert score["supported_claims"] == 1
    assert score["contradicted_claims"] == 1
    assert score["not_enough_evidence_claims"] == 1
    assert score["faithfulness_score"] == 1 / 3


def test_faithfulness_score_still_supports_old_similarity_labels():
    verification_results = [
        {
            "claim": "Claim 1",
            "label": "Supported",
        },
        {
            "claim": "Claim 2",
            "label": "Partially supported",
        },
        {
            "claim": "Claim 3",
            "label": "Unsupported",
        },
    ]

    score = calculate_faithfulness_score(verification_results)

    assert score["total_claims"] == 3
    assert score["supported_claims"] == 1
    assert score["partially_supported_claims"] == 1
    assert score["unsupported_claims"] == 1
    assert score["faithfulness_score"] == 1 / 3
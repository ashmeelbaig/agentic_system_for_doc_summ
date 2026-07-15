from src.scoring import calculate_faithfulness_score


def test_faithfulness_score():
    verification_results = [
        {"claim": "Claim 1", "label": "Supported"},
        {"claim": "Claim 2", "label": "Supported"},
        {"claim": "Claim 3", "label": "Unsupported"},
    ]

    score = calculate_faithfulness_score(verification_results)

    assert isinstance(score, dict)
    assert score["total_claims"] == 3
    assert score["supported_claims"] == 2
    assert score["unsupported_claims"] == 1
    assert score["faithfulness_score"] == 2 / 3
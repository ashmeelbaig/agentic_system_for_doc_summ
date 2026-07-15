from typing import List, Dict, Any


def calculate_faithfulness_score(
    verification_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate a simple faithfulness score from claim verification results.

    Args:
        verification_results: List of claim verification results.

    Returns:
        Dictionary containing score summary.
    """

    if not verification_results:
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "partially_supported_claims": 0,
            "unsupported_claims": 0,
            "faithfulness_score": 0.0
        }

    total_claims = len(verification_results)

    supported_claims = sum(
        1 for result in verification_results
        if result["label"] == "Supported"
    )

    partially_supported_claims = sum(
        1 for result in verification_results
        if result["label"] == "Partially supported"
    )

    unsupported_claims = sum(
        1 for result in verification_results
        if result["label"] == "Unsupported"
    )

    faithfulness_score = supported_claims / total_claims

    return {
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "partially_supported_claims": partially_supported_claims,
        "unsupported_claims": unsupported_claims,
        "faithfulness_score": faithfulness_score
    }


def print_faithfulness_score(score_summary: Dict[str, Any]) -> None:
    """
    Print faithfulness score in terminal.
    """

    print("\nFaithfulness Score")
    print("=" * 70)

    print(f"Total claims: {score_summary['total_claims']}")
    print(f"Supported claims: {score_summary['supported_claims']}")
    print(f"Partially supported claims: {score_summary['partially_supported_claims']}")
    print(f"Unsupported claims: {score_summary['unsupported_claims']}")
    print(f"Final faithfulness score: {score_summary['faithfulness_score']:.2f}")
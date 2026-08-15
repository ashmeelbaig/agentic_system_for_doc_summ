from typing import List, Dict, Any


def calculate_faithfulness_score(verification_results):
    """
    Calculate faithfulness score from claim verification results.

    Supports both:
    - semantic similarity labels:
      Supported, Partially supported, Unsupported

    - NLI labels:
      Supported, Contradicted, Not enough evidence
    """

    total_claims = len(verification_results)

    if total_claims == 0:
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "partially_supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "not_enough_evidence_claims": 0,
            "faithfulness_score": 0.0,
        }

    supported_claims = 0
    partially_supported_claims = 0
    unsupported_claims = 0
    contradicted_claims = 0
    not_enough_evidence_claims = 0

    for result in verification_results:
        label = result.get("label", "")

        if label == "Supported":
            supported_claims += 1
        elif label == "Partially supported":
            partially_supported_claims += 1
        elif label == "Unsupported":
            unsupported_claims += 1
        elif label == "Contradicted":
            contradicted_claims += 1
        elif label == "Not enough evidence":
            not_enough_evidence_claims += 1
        else:
            unsupported_claims += 1

    faithfulness_score = supported_claims / total_claims

    return {
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "partially_supported_claims": partially_supported_claims,
        "unsupported_claims": unsupported_claims,
        "contradicted_claims": contradicted_claims,
        "not_enough_evidence_claims": not_enough_evidence_claims,
        "faithfulness_score": faithfulness_score,
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
from typing import Dict, Any, List, Tuple


def create_baseline_result(
    query: str,
    answer: str,
    retrieved_chunks: List[Tuple[int, str, float]]
) -> Dict[str, Any]:
    """
    Create a simple standard RAG baseline result.

    The baseline only contains:
    1. User question
    2. Generated answer
    3. Retrieved evidence count

    It does not include claim extraction, verification, or faithfulness scoring.
    """

    return {
        "system_type": "Standard RAG Baseline",
        "query": query,
        "answer": answer,
        "retrieved_evidence_count": len(retrieved_chunks),
        "claim_verification": "Not available in baseline",
        "faithfulness_score": "Not available in baseline"
    }


def print_baseline_result(baseline_result: Dict[str, Any]) -> None:
    """
    Print baseline result in terminal.
    """

    print("\n" + "=" * 80)
    print("Baseline RAG Output")
    print("=" * 80)

    print(f"System type: {baseline_result['system_type']}")
    print(f"Retrieved evidence chunks: {baseline_result['retrieved_evidence_count']}")

    print("\nAnswer:")
    print(baseline_result["answer"])

    print("\nClaim verification: Not available")
    print("Faithfulness score: Not available")
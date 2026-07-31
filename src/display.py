from typing import List, Dict, Any


def print_header(title: str) -> None:
    """
    Print a clean section header.
    """

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title: str) -> None:
    """
    Print a clean subsection header.
    """

    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def shorten_text(text: str, max_chars: int = 350) -> str:
    """
    Shorten long text for terminal display.
    """

    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."

def normalize_evidence_item(item: Any) -> Dict[str, Any]:
    """
    Normalize retrieved evidence for display.

    Supports:
    - old tuple format: (chunk_index, chunk_text, score)
    - new metadata dictionary format
    """

    if isinstance(item, dict):
        return {
            "chunk_index": item.get("chunk_index"),
            "chunk_id": item.get("chunk_id"),
            "source": item.get("source"),
            "page_number": item.get("page_number"),
            "text": item.get("text", ""),
            "score": float(item.get("score", 0.0) or 0.0),
        }

    if isinstance(item, tuple) and len(item) == 3:
        chunk_index, chunk_text, score = item

        return {
            "chunk_index": chunk_index,
            "chunk_id": None,
            "source": None,
            "page_number": None,
            "text": chunk_text,
            "score": float(score),
        }

    raise TypeError(
        "Evidence item must be either a metadata dictionary or a tuple of "
        "(chunk_index, chunk_text, score)."
    )


def print_document_status(pdf_name: str, total_chars: int, total_chunks: int) -> None:
    """
    Print document processing status.
    """

    print_header("Document Processing")
    print(f"Selected PDF: {pdf_name}")
    print(f"Extracted characters: {total_chars}")
    print(f"Total chunks created: {total_chunks}")


def print_evidence_summary(results: List[Any]) -> None:
    """
    Print short retrieved evidence summary.

    Supports both old tuple-based evidence and new metadata-aware evidence.
    """

    print_header("Retrieved Evidence")

    if not results:
        print("No evidence retrieved.")
        return

    for rank, item in enumerate(results, start=1):
        evidence = normalize_evidence_item(item)

        print(f"\nEvidence {rank}")

        if evidence["chunk_id"] is not None:
            print(f"Chunk ID: {evidence['chunk_id']}")
        else:
            print(f"Chunk index: {evidence['chunk_index']}")

        if evidence["source"]:
            print(f"Source: {evidence['source']}")

        if evidence["page_number"] is not None:
            print(f"Page number: {evidence['page_number']}")

        print(f"Similarity score: {evidence['score']:.4f}")
        print(f"Preview: {shorten_text(evidence['text'], max_chars=400)}")


def print_generated_answer(answer: str) -> None:
    """
    Print generated answer.
    """

    print_header("Generated Answer")
    print(answer)


def print_claim_table(verification_results: List[Dict[str, Any]]) -> None:
    """
    Print claim verification results in a clean terminal format.
    """

    print_header("Claim Verification")

    if not verification_results:
        print("No claims were verified.")
        return

    for index, result in enumerate(verification_results, start=1):
        print(f"\nClaim {index}")
        print(f"Claim: {result['claim']}")
        print(f"Label: {result['label']}")
        print(f"Similarity score: {result['score']:.4f}")
        if result.get("chunk_id"):
            print(f"Evidence chunk: {result['chunk_id']}")
        else:
            print(f"Evidence chunk: {result.get('chunk_index')}")

        if result.get("source"):
            print(f"Source: {result['source']}")

        if result.get("page_number") is not None:
            print(f"Page number: {result['page_number']}")
        print(f"Best evidence: {shorten_text(result['evidence'], max_chars=300)}")


def print_score_summary(score_summary: Dict[str, Any]) -> None:
    """
    Print final faithfulness score.
    """

    print_header("Faithfulness Score")
    print(f"Total claims: {score_summary['total_claims']}")
    print(f"Supported claims: {score_summary['supported_claims']}")
    print(f"Partially supported claims: {score_summary['partially_supported_claims']}")
    print(f"Unsupported claims: {score_summary['unsupported_claims']}")
    print(f"Final faithfulness score: {score_summary['faithfulness_score']:.2f}")
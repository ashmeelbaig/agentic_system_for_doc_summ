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

    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def normalize_evidence_item(item: Any) -> Dict[str, Any]:
    """
    Normalize retrieved evidence for display.

    Supports:
    - old tuple format: (chunk_index, chunk_text, score)
    - metadata dictionary format
    - reranked metadata dictionary format
    """

    if isinstance(item, dict):
        similarity_score = item.get("similarity_score", item.get("score", 0.0))

        return {
            "chunk_index": item.get("chunk_index"),
            "chunk_id": item.get("chunk_id"),
            "source": item.get("source"),
            "page_number": item.get("page_number"),
            "text": item.get("text", ""),
            "similarity_score": float(similarity_score or 0.0),
            "rerank_score": item.get("rerank_score"),
            "original_rank": item.get("original_rank"),
        }

    if isinstance(item, tuple) and len(item) == 3:
        chunk_index, chunk_text, score = item

        return {
            "chunk_index": chunk_index,
            "chunk_id": None,
            "source": None,
            "page_number": None,
            "text": chunk_text,
            "similarity_score": float(score),
            "rerank_score": None,
            "original_rank": None,
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
    Print retrieved evidence summary.

    Supports:
    - old tuple-based evidence
    - metadata-aware evidence
    - reranked evidence
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

        if evidence["original_rank"] is not None:
            print(f"Original FAISS rank: {evidence['original_rank']}")

        print(f"Similarity score: {evidence['similarity_score']:.4f}")

        if evidence["rerank_score"] is not None:
            print(f"Rerank score: {float(evidence['rerank_score']):.4f}")

        print(f"Preview: {shorten_text(evidence['text'], max_chars=400)}")


def print_generated_answer(answer: str) -> None:
    """
    Print generated answer.
    """

    print_header("Generated Answer")
    print(answer)


def print_claim_table(verification_results: List[Dict[str, Any]]) -> None:
    """
    Print claim verification results.

    Supports:
    - semantic similarity verifier results with "score"
    - NLI verifier results with "nli_score" and "nli_label"
    """

    print_header("Claim Verification")

    if not verification_results:
        print("No claims were verified.")
        return

    for index, result in enumerate(verification_results, start=1):
        print(f"\nClaim {index}")
        print(f"Claim: {result.get('claim')}")
        print(f"Label: {result.get('label')}")

        if "nli_score" in result:
            print(f"NLI label: {result.get('nli_label', 'N/A')}")
            print(f"NLI score: {float(result.get('nli_score', 0.0)):.4f}")
        elif "score" in result:
            print(f"Similarity score: {float(result.get('score', 0.0)):.4f}")
        else:
            print("Verification score: N/A")

        if result.get("chunk_id"):
            print(f"Evidence chunk: {result['chunk_id']}")
        elif result.get("chunk_index") is not None:
            print(f"Evidence chunk: {result.get('chunk_index')}")
        else:
            print("Evidence chunk: N/A")

        if result.get("source"):
            print(f"Source: {result['source']}")

        if result.get("page_number") is not None:
            print(f"Page number: {result['page_number']}")

        print(
            "Best evidence: "
            f"{shorten_text(result.get('evidence', ''), max_chars=300)}"
        )


def print_score_summary(score_summary: Dict[str, Any]) -> None:
    """
    Print final faithfulness score.
    """

    print_header("Faithfulness Score")

    print(f"Total claims: {score_summary.get('total_claims', 0)}")
    print(f"Supported claims: {score_summary.get('supported_claims', 0)}")
    print(
        "Partially supported claims: "
        f"{score_summary.get('partially_supported_claims', 0)}"
    )
    print(f"Unsupported claims: {score_summary.get('unsupported_claims', 0)}")

    if "contradicted_claims" in score_summary:
        print(f"Contradicted claims: {score_summary['contradicted_claims']}")

    if "not_enough_evidence_claims" in score_summary:
        print(
            "Not enough evidence claims: "
            f"{score_summary['not_enough_evidence_claims']}"
        )

    print(
        "Final faithfulness score: "
        f"{float(score_summary.get('faithfulness_score', 0.0)):.2f}"
    )
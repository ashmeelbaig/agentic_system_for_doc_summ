import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

def normalize_retrieved_evidence_item(item: Any) -> Dict[str, Any]:
    """
    Normalize retrieved evidence before saving to JSON.

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
            "similarity_score": float(item.get("score", 0.0) or 0.0),
            "text": item.get("text", ""),
            "rerank_score": item.get("rerank_score"),
        }

    if isinstance(item, tuple) and len(item) == 3:
        chunk_index, chunk_text, score = item

        return {
            "chunk_index": chunk_index,
            "chunk_id": None,
            "source": None,
            "page_number": None,
            "similarity_score": float(score),
            "text": chunk_text,
            "rerank_score": None,
        }

    raise TypeError(
        "Retrieved evidence must be either a metadata dictionary or a tuple of "
        "(chunk_index, chunk_text, score)."
    )

def save_result_to_json(
    output_dir: str,
    pdf_name: str,
    query: str,
    answer: str,
    retrieved_chunks: List[Any],
    claims: List[str],
    verification_results: List[Dict[str, Any]],
    score_summary: Dict[str, Any],
    baseline_result: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save one complete prototype result to a JSON file.

    The saved result includes:
    1. Standard RAG baseline
    2. Claim grounded RAG output
    3. Retrieved evidence
    4. Extracted claims
    5. Claim verification results
    6. Faithfulness score
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = make_safe_filename(query)

    file_name = f"result_{timestamp}_{safe_query}.json"
    file_path = output_path / file_name

    retrieved_evidence = [
        normalize_retrieved_evidence_item(item)
        for item in retrieved_chunks
    ]

    result_data = {
        "pdf_name": pdf_name,
        "query": query,

        "baseline_rag": baseline_result,

        "claim_grounded_rag": {
            "generated_answer": answer,
            "retrieved_evidence": retrieved_evidence,
            "extracted_claims": claims,
            "verification_results": verification_results,
            "faithfulness_score": score_summary
        }
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(result_data, file, indent=4, ensure_ascii=False)

    return file_path


def make_safe_filename(text: str, max_length: int = 40) -> str:
    """
    Convert query text into a safe short filename.
    """

    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = text.strip("_")

    if not text:
        text = "query"

    return text[:max_length]
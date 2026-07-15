import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


def save_result_to_json(
    output_dir: str,
    pdf_name: str,
    query: str,
    answer: str,
    retrieved_chunks: List[Tuple[int, str, float]],
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

    retrieved_evidence = []

    for chunk_index, chunk_text, score in retrieved_chunks:
        retrieved_evidence.append(
            {
                "chunk_index": chunk_index,
                "similarity_score": score,
                "text": chunk_text
            }
        )

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
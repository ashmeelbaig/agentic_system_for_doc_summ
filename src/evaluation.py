import json
from pathlib import Path
from typing import Dict, Any


def summarize_result_file(result_file: Path) -> Dict[str, Any]:
    """
    Summarize one saved JSON result file.

    This is used for lightweight evaluation and report preparation.
    """

    result_file = Path(result_file)

    with open(result_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    generator = data.get("generator") or {}
    claim_grounded = data.get("claim_grounded_rag") or {}
    score_summary = claim_grounded.get("faithfulness_score") or {}

    retrieved_evidence = claim_grounded.get("retrieved_evidence") or []
    extracted_claims = claim_grounded.get("extracted_claims") or []

    return {
        "file_name": result_file.name,
        "pdf_name": data.get("pdf_name"),
        "query": data.get("query"),

        "generator_mode": generator.get("mode"),
        "generator_model": generator.get("model_name"),

        "retrieved_evidence_count": len(retrieved_evidence),
        "total_claims": score_summary.get("total_claims", len(extracted_claims)),
        "supported_claims": score_summary.get("supported_claims", 0),
        "partially_supported_claims": score_summary.get("partially_supported_claims", 0),
        "unsupported_claims": score_summary.get("unsupported_claims", 0),
        "contradicted_claims": score_summary.get("contradicted_claims", 0),
        "not_enough_evidence_claims": score_summary.get("not_enough_evidence_claims", 0),
        "faithfulness_score": score_summary.get("faithfulness_score", 0.0),
    }
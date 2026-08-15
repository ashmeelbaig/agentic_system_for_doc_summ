import json
from pathlib import Path
from typing import Dict, Any
import csv


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

def save_evaluation_summary_csv(
    summaries,
    output_csv_path
):
    """
    Save evaluation summaries to a CSV file.
    """

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_name",
        "pdf_name",
        "query",
        "generator_mode",
        "generator_model",
        "retrieved_evidence_count",
        "total_claims",
        "supported_claims",
        "partially_supported_claims",
        "unsupported_claims",
        "contradicted_claims",
        "not_enough_evidence_claims",
        "faithfulness_score",
    ]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for summary in summaries:
            writer.writerow(
                {
                    field: summary.get(field)
                    for field in fieldnames
                }
            )

    return output_csv_path
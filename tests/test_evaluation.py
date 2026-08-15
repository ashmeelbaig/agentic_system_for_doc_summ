import json

from src.evaluation import summarize_result_file


def test_summarize_result_file_extracts_key_metrics(tmp_path):
    result_file = tmp_path / "result_test.json"

    data = {
        "pdf_name": "sample.pdf",
        "query": "How does the system verify claims?",
        "generator": {
            "mode": "quality",
            "model_name": "google/flan-t5-base",
        },
        "baseline_rag": {
            "system_type": "Standard RAG Baseline",
            "retrieved_evidence_count": 4,
        },
        "claim_grounded_rag": {
            "generated_answer": "The system verifies claims using evidence.",
            "retrieved_evidence": [
                {"chunk_id": "doc_p1_c0"},
                {"chunk_id": "doc_p2_c0"},
            ],
            "extracted_claims": [
                "The system verifies claims using evidence."
            ],
            "verification_results": [
                {
                    "claim": "The system verifies claims using evidence.",
                    "label": "Supported",
                    "nli_label": "ENTAILMENT",
                    "nli_score": 0.94,
                }
            ],
            "faithfulness_score": {
                "total_claims": 1,
                "supported_claims": 1,
                "partially_supported_claims": 0,
                "unsupported_claims": 0,
                "contradicted_claims": 0,
                "not_enough_evidence_claims": 0,
                "faithfulness_score": 1.0,
            },
        },
    }

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(data, file)

    summary = summarize_result_file(result_file)

    assert summary["pdf_name"] == "sample.pdf"
    assert summary["query"] == "How does the system verify claims?"
    assert summary["generator_mode"] == "quality"
    assert summary["generator_model"] == "google/flan-t5-base"
    assert summary["retrieved_evidence_count"] == 2
    assert summary["total_claims"] == 1
    assert summary["supported_claims"] == 1
    assert summary["contradicted_claims"] == 0
    assert summary["not_enough_evidence_claims"] == 0
    assert summary["faithfulness_score"] == 1.0
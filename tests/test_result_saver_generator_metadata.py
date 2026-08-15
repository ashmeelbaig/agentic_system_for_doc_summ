import json

from src.result_saver import save_result_to_json


def test_save_result_to_json_includes_generator_metadata(tmp_path):
    saved_file = save_result_to_json(
        output_dir=str(tmp_path),
        pdf_name="sample.pdf",
        query="How does the system answer questions?",
        answer="The system answers using retrieved evidence.",
        retrieved_chunks=[],
        claims=[],
        verification_results=[],
        score_summary={
            "total_claims": 0,
            "supported_claims": 0,
            "partially_supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "not_enough_evidence_claims": 0,
            "faithfulness_score": 0.0,
        },
        baseline_result=None,
        generator_metadata={
            "mode": "quality",
            "model_name": "google/flan-t5-base",
        },
    )

    with open(saved_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["generator"]["mode"] == "quality"
    assert data["generator"]["model_name"] == "google/flan-t5-base"
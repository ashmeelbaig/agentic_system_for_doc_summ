import json

from src.result_saver import save_multi_model_result_to_json


def test_multi_model_result_saver_records_mode_and_provider(tmp_path):
    saved = save_multi_model_result_to_json(
        str(tmp_path),
        "document.pdf",
        "Question?",
        {"retrieved_evidence": []},
        {"local/model": {"status": "success", "provider": "local_transformers"}},
        [],
        generator_mode="multi_model",
    )

    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["generator_mode"] == "multi_model"
    assert data["model_results"]["local/model"]["provider"] == "local_transformers"

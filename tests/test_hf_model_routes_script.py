import json

from scripts import test_hf_model_routes


class FakeClient:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]

    def chat_completion(self, **kwargs):
        if self.model.startswith("Qwen/"):
            raise RuntimeError("provider unavailable for chat")
        return {"choices": [{"message": {"content": "A short chat answer."}}]}

    def text_generation(self, prompt, **kwargs):
        return "A short generated answer."


def test_route_script_uses_fallback_and_never_emits_token(monkeypatch, tmp_path, capsys):
    secret = "route-test-secret-never-expose"
    output_path = tmp_path / "route-report.json"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(test_hf_model_routes, "InferenceClient", FakeClient)
    monkeypatch.setattr(test_hf_model_routes, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(test_hf_model_routes, "load_dotenv", lambda: None)

    assert test_hf_model_routes.main() == 0

    printed = capsys.readouterr().out
    saved = output_path.read_text(encoding="utf-8")
    report = json.loads(saved)
    assert secret not in printed
    assert secret not in saved
    assert [item["model_name"] for item in report["models"]] == test_hf_model_routes.MODEL_IDS
    assert report["models"][0]["method"] == "text_generation"
    assert [attempt["method"] for attempt in report["models"][0]["attempts"]] == [
        "chat_completion",
        "text_generation",
    ]
    assert report["models"][1]["method"] == "chat_completion"

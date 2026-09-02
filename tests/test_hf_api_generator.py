import pytest

from src.generators.hf_api_generator import (
    HuggingFaceAPIGenerator,
    HuggingFaceGenerationError,
)


class FakeClient:
    def text_generation(self, prompt, **kwargs):
        return "Answer: A concise grounded answer from the evidence."


def test_hf_generator_reads_token_from_environment(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setenv("HF_TOKEN", "test-token-never-print")
    monkeypatch.setattr(
        "src.generators.hf_api_generator.InferenceClient", fake_client
    )

    generator = HuggingFaceAPIGenerator("test/model")
    answer = generator.generate_answer(
        "What is described?", [{"text": "The evidence describes a grounded fact."}]
    )

    assert captured["token"] == "test-token-never-print"
    assert answer == "A concise grounded answer from the evidence."


def test_llama_uses_plain_completion_prompt(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    generator = HuggingFaceAPIGenerator(
        "meta-llama/Llama-2-7b-hf", client=FakeClient()
    )
    prompt = generator._build_prompt("Question?", [{"text": "Evidence."}], None)
    assert "Question: Question?" in prompt
    assert "Answer:" in prompt


class RecordingClient:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def text_generation(self, prompt, **kwargs):
        self.calls.append(("text_generation", kwargs))
        if self.fail_first:
            raise RuntimeError("provider expects another task")
        return "Text generation answer."

    def text2text_generation(self, prompt, **kwargs):
        self.calls.append(("text2text_generation", kwargs))
        return "Text2text answer."

    def chat_completion(self, **kwargs):
        self.calls.append(("chat_completion", kwargs))
        return {"choices": [{"message": {"content": "Chat answer."}}]}


def test_qwen_uses_text_generation_first(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "placeholder-secret")
    client = RecordingClient()
    generator = HuggingFaceAPIGenerator("Qwen/Qwen2.5-7B-Instruct", client=client)

    assert generator.generate_answer("Question?", [{"text": "Evidence."}]) == "Text generation answer."
    assert [call[0] for call in client.calls] == ["text_generation"]
    assert client.calls[0][1]["max_new_tokens"] == 180
    assert client.calls[0][1]["temperature"] == 0.1
    assert client.calls[0][1]["do_sample"] is False


def test_flan_uses_text2text_generation_first(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "placeholder-secret")
    client = RecordingClient()
    generator = HuggingFaceAPIGenerator("google/flan-t5-large", client=client)

    assert generator.generate_answer("Question?", [{"text": "Evidence."}]) == "Text2text answer."
    assert [call[0] for call in client.calls] == ["text2text_generation"]


def test_qwen_falls_back_to_chat_completion(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "placeholder-secret")
    client = RecordingClient(fail_first=True)
    generator = HuggingFaceAPIGenerator("Qwen/Qwen2.5-7B-Instruct", client=client)

    assert generator.generate_answer("Question?", [{"text": "Evidence."}]) == "Chat answer."
    assert generator.attempted_methods == ["text_generation", "chat_completion"]


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ],
)
def test_weak_chat_models_use_chat_completion_first(monkeypatch, model_id):
    monkeypatch.setenv("HF_TOKEN", "placeholder-secret")
    client = RecordingClient()
    generator = HuggingFaceAPIGenerator(model_id, client=client)

    assert generator.model_id == model_id
    assert generator.generate_answer("Question?", [{"text": "Evidence."}]) == "Chat answer."
    assert [call[0] for call in client.calls] == ["chat_completion"]


def test_weak_chat_model_falls_back_to_text_generation(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "placeholder-secret")
    client = RecordingClient()
    client.chat_completion = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("chat task not supported")
    )
    generator = HuggingFaceAPIGenerator(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", client=client
    )

    assert generator.generate_answer("Question?", [{"text": "Evidence."}]) == "Text generation answer."
    assert generator.attempted_methods == ["chat_completion", "text_generation"]
    assert generator.attempt_failures[0]["method"] == "chat_completion"


def test_token_is_never_in_controlled_error(monkeypatch):
    secret = "highly-sensitive-placeholder"
    monkeypatch.setenv("HF_TOKEN", secret)
    client = RecordingClient(fail_first=True)
    client.chat_completion = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError(f"401 Authorization Bearer {secret}")
    )
    generator = HuggingFaceAPIGenerator("Qwen/Qwen2.5-7B-Instruct", client=client)

    with pytest.raises(HuggingFaceGenerationError) as error:
        generator.generate_answer("Question?", [{"text": "Evidence."}])
    assert secret not in str(error.value)
    assert "access was denied" in str(error.value)


def test_missing_token_has_clear_controlled_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(HuggingFaceGenerationError, match="HF_TOKEN is not configured"):
        HuggingFaceAPIGenerator("Qwen/Qwen2.5-7B-Instruct", client=RecordingClient())


def test_empty_provider_routing_error_is_classified(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "placeholder-secret")
    client = RecordingClient(fail_first=True)
    client.chat_completion = lambda **kwargs: (_ for _ in ()).throw(StopIteration())
    generator = HuggingFaceAPIGenerator("Qwen/Qwen2.5-7B-Instruct", client=client)

    with pytest.raises(HuggingFaceGenerationError, match="task method"):
        generator.generate_answer("Question?", [{"text": "Evidence."}])
    assert generator.attempt_failures == [
        {
            "method": "text_generation",
            "error": "Hugging Face rejected the generation task method for this model.",
        },
        {
            "method": "chat_completion",
            "error": "Hugging Face provider is unavailable for this model.",
        },
    ]


def test_model_task_provider_mismatch_is_provider_error():
    error = HuggingFaceAPIGenerator._controlled_error(
        ValueError(
            "Model example/model is not supported for task text-generation "
            "and provider example-provider"
        )
    )
    assert str(error) == "Hugging Face provider is unavailable for this model."


@pytest.mark.parametrize(
    "message",
    [
        "provider is unavailable",
        "no provider configured",
        "model_not_supported",
        "not supported by any provider",
        "this is a non-serverless model",
    ],
)
def test_provider_unavailable_messages_are_classified(message):
    error = HuggingFaceAPIGenerator._controlled_error(RuntimeError(message))
    assert str(error) == "Hugging Face provider is unavailable for this model."


def test_wrapped_stop_iteration_is_classified_as_provider_unavailable():
    try:
        try:
            raise StopIteration()
        except StopIteration as exc:
            raise RuntimeError() from exc
    except RuntimeError as exc:
        error = HuggingFaceAPIGenerator._controlled_error(exc)

    assert str(error) == "Hugging Face provider is unavailable for this model."


def test_unrecognized_provider_error_remains_unclassified():
    error = HuggingFaceAPIGenerator._controlled_error(RuntimeError("unexpected failure"))
    assert str(error) == "Hugging Face generation failed for an unclassified provider error."

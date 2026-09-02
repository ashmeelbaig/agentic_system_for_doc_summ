import torch
import pytest

from src.generators.local_transformers_generator import (
    LocalTransformersGenerationError,
    LocalTransformersGenerator,
)


class FakeTokenizer:
    def __init__(self, use_chat=True):
        self.use_chat = use_chat
        self.chat_calls = []
        self.prompt = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        if not self.use_chat:
            raise ValueError("no chat template")
        return "CHAT TEMPLATE PROMPT"

    def __call__(self, prompt, return_tensors):
        self.prompt = prompt
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

    def decode(self, tokens, skip_special_tokens):
        return "Concise local answer."


class FakeModel:
    def parameters(self):
        return iter([torch.tensor([0])])

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


def test_local_generator_uses_chat_template(monkeypatch):
    tokenizer = FakeTokenizer(use_chat=True)
    generator = LocalTransformersGenerator(
        "Qwen/Qwen2.5-0.5B-Instruct", tokenizer=tokenizer, model=FakeModel()
    )

    answer = generator.generate_answer("Question?", [{"text": "Evidence."}])

    assert answer == "Concise local answer."
    assert tokenizer.prompt == "CHAT TEMPLATE PROMPT"
    assert tokenizer.chat_calls[0][1]["add_generation_prompt"] is True


def test_local_generator_falls_back_to_plain_prompt(monkeypatch):
    tokenizer = FakeTokenizer(use_chat=False)
    generator = LocalTransformersGenerator(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer=tokenizer,
        model=FakeModel(),
    )

    generator.generate_answer("Question?", [{"text": "Shared evidence."}])

    assert "Question: Question?" in tokenizer.prompt
    assert "Shared evidence." in tokenizer.prompt
    assert tokenizer.prompt.endswith("Answer:")


def test_local_generator_error_never_contains_hf_token(monkeypatch):
    secret = "local-model-secret-never-expose"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(
        "src.generators.local_transformers_generator.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    generator = LocalTransformersGenerator("Qwen/Qwen2.5-0.5B-Instruct")

    with pytest.raises(LocalTransformersGenerationError) as error:
        generator.generate_answer("Question?", [{"text": "Evidence."}])

    assert secret not in str(error.value)

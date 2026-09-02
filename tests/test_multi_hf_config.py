from pathlib import Path

from main import get_generator_mode, print_multi_hf_startup_check
from src.generators.factory import get_configured_model_ids


def test_env_example_exists_and_secret_files_are_ignored():
    example = Path(".env.example")
    ignored = Path(".gitignore").read_text(encoding="utf-8")

    assert example.exists()
    assert "HF_TOKEN=your_huggingface_token_here" in example.read_text(encoding="utf-8")
    assert ".env" in ignored
    assert ".env.local" in ignored
    assert "*.env" in ignored


def test_factory_reads_configured_model_ids(monkeypatch):
    monkeypatch.setenv("HF_MODEL_IDS", "model/a, model/b,model/a")
    assert get_configured_model_ids() == ["model/a", "model/b"]


def test_factory_uses_individual_model_overrides(monkeypatch):
    monkeypatch.delenv("HF_MODEL_IDS", raising=False)
    monkeypatch.setenv("LLAMA_MODEL_ID", "organization/new-llama")
    assert "organization/new-llama" in get_configured_model_ids()


def test_factory_uses_provider_compatible_qwen_default(monkeypatch):
    monkeypatch.delenv("HF_MODEL_IDS", raising=False)
    monkeypatch.delenv("QWEN_MODEL_ID", raising=False)

    assert "Qwen/Qwen2.5-7B-Instruct" in get_configured_model_ids()
    assert "Qwen/Qwen2.5-7B-Instruct-1M" not in get_configured_model_ids()


def test_multi_hf_is_a_supported_generator_mode(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "multi_hf")
    assert get_generator_mode() == "multi_hf"


def test_startup_check_prints_presence_only(monkeypatch, capsys):
    secret = "startup-secret-placeholder"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("HF_MODEL_IDS", "Qwen/Qwen2.5-7B-Instruct")
    print_multi_hf_startup_check()
    output = capsys.readouterr().out
    assert "token_loaded=True" in output
    assert "model_ids_loaded=True" in output
    assert secret not in output

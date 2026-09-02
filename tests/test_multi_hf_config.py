from pathlib import Path

from main import get_generator_mode, print_multi_hf_startup_check
from src.generators.factory import (
    get_all_configured_models,
    get_configured_model_ids,
    get_hf_model_ids,
    get_local_model_ids,
    get_weak_model_ids,
)


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


def test_factory_reads_weak_model_ids(monkeypatch):
    monkeypatch.setenv(
        "WEAK_HF_MODEL_IDS",
        " Qwen/Qwen2.5-0.5B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0 ",
    )
    assert get_weak_model_ids() == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]


def test_factory_uses_exact_default_weak_model_ids(monkeypatch):
    monkeypatch.delenv("WEAK_HF_MODEL_IDS", raising=False)
    assert get_weak_model_ids() == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]


def test_multi_model_configuration_combines_hosted_and_local(monkeypatch):
    hosted = (
        "Qwen/Qwen2.5-Coder-32B-Instruct,"
        "meta-llama/Llama-3.1-8B-Instruct:nscale"
    )
    local = (
        "Qwen/Qwen2.5-0.5B-Instruct,"
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    monkeypatch.setenv("HF_MODEL_IDS", hosted)
    monkeypatch.setenv("LOCAL_MODEL_IDS", local)

    assert get_hf_model_ids() == hosted.split(",")
    assert get_local_model_ids() == local.split(",")
    assert get_all_configured_models() == [
        *({"model_id": value, "provider_type": "hosted_hf"} for value in hosted.split(",")),
        *({"model_id": value, "provider_type": "local_transformers"} for value in local.split(",")),
    ]


def test_factory_uses_individual_model_overrides(monkeypatch):
    monkeypatch.delenv("HF_MODEL_IDS", raising=False)
    monkeypatch.setenv("LLAMA_MODEL_ID", "organization/new-llama")
    assert "organization/new-llama" in get_configured_model_ids()


def test_factory_uses_provider_compatible_qwen_default(monkeypatch):
    monkeypatch.delenv("HF_MODEL_IDS", raising=False)
    monkeypatch.delenv("QWEN_MODEL_ID", raising=False)

    assert "Qwen/Qwen2.5-Coder-32B-Instruct" in get_configured_model_ids()
    assert "Qwen/Qwen2.5-7B-Instruct-1M" not in get_configured_model_ids()


def test_default_active_models_are_only_the_two_hosted_models(monkeypatch):
    monkeypatch.delenv("HF_MODEL_IDS", raising=False)
    monkeypatch.delenv("QWEN_MODEL_ID", raising=False)
    monkeypatch.delenv("LLAMA_MODEL_ID", raising=False)

    assert get_configured_model_ids() == [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct:nscale",
    ]
    assert "Qwen/Qwen2.5-0.5B-Instruct" not in get_configured_model_ids()
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" not in get_configured_model_ids()


def test_empty_local_model_ids_configures_no_local_generators(monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL_IDS", "  ")
    monkeypatch.delenv("HF_MODEL_IDS", raising=False)

    assert get_local_model_ids() == []
    configured = get_all_configured_models()
    assert len(configured) == 2
    assert all(item["provider_type"] == "hosted_hf" for item in configured)


def test_env_example_keeps_weak_models_inactive():
    contents = Path(".env.example").read_text(encoding="utf-8")
    assert "GENERATOR_MODE=multi_hf" in contents
    assert "LOCAL_MODEL_IDS=\n" in contents
    assert "HF_MODEL_IDS=Qwen/Qwen2.5-Coder-32B-Instruct,meta-llama/Llama-3.1-8B-Instruct:nscale" in contents


def test_multi_hf_is_a_supported_generator_mode(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "multi_hf")
    assert get_generator_mode() == "multi_hf"


def test_multi_model_is_a_supported_generator_mode(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "multi_model")
    assert get_generator_mode() == "multi_model"


def test_startup_check_prints_presence_only(monkeypatch, capsys):
    secret = "startup-secret-placeholder"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("HF_MODEL_IDS", "Qwen/Qwen2.5-7B-Instruct")
    print_multi_hf_startup_check()
    output = capsys.readouterr().out
    assert "token_loaded=True" in output
    assert "model_ids_loaded=True" in output
    assert secret not in output

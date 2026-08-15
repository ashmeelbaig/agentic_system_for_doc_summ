from main import get_generator_model_name


def test_get_generator_model_name_fast_mode(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "fast")

    model_name = get_generator_model_name()

    assert model_name == "google/flan-t5-small"


def test_get_generator_model_name_quality_mode(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "quality")

    model_name = get_generator_model_name()

    assert model_name == "google/flan-t5-base"


def test_get_generator_model_name_defaults_to_quality(monkeypatch):
    monkeypatch.delenv("GENERATOR_MODE", raising=False)

    model_name = get_generator_model_name()

    assert model_name == "google/flan-t5-base"
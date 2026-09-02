import pytest

from main import get_generator_mode, get_generator_model_name


def test_get_generator_mode_returns_quality_by_default(monkeypatch):
    monkeypatch.delenv("GENERATOR_MODE", raising=False)

    mode = get_generator_mode()

    assert mode == "quality"


def test_get_generator_mode_falls_back_to_quality_for_invalid_mode(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "invalid")

    mode = get_generator_mode()

    assert mode == "quality"


@pytest.mark.parametrize(
    ("mode", "model_name"),
    [
        ("fast", "google/flan-t5-small"),
        ("quality", "google/flan-t5-base"),
        ("quality_plus", "google/flan-t5-large"),
    ],
)
def test_generator_mode_selects_expected_model(monkeypatch, mode, model_name):
    monkeypatch.setenv("GENERATOR_MODE", mode)

    assert get_generator_mode() == mode
    assert get_generator_model_name() == model_name

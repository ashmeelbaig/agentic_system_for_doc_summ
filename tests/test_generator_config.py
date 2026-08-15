from main import get_generator_mode


def test_get_generator_mode_returns_quality_by_default(monkeypatch):
    monkeypatch.delenv("GENERATOR_MODE", raising=False)

    mode = get_generator_mode()

    assert mode == "quality"


def test_get_generator_mode_falls_back_to_quality_for_invalid_mode(monkeypatch):
    monkeypatch.setenv("GENERATOR_MODE", "invalid")

    mode = get_generator_mode()

    assert mode == "quality"
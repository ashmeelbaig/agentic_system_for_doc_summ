from pathlib import Path


def test_evaluate_outputs_script_exists():
    script_path = Path("scripts/evaluate_outputs.py")

    assert script_path.exists()
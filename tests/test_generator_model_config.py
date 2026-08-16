import inspect

from src.generator import AnswerGenerator


def test_answer_generator_uses_stronger_default_model():
    signature = inspect.signature(AnswerGenerator.__init__)
    default_model = signature.parameters["model_name"].default

    assert default_model == "google/flan-t5-base"
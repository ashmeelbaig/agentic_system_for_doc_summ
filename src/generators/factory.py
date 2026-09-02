import os
from typing import List

from src.generators.base import BaseAnswerGenerator
from src.generators.hf_api_generator import HuggingFaceAPIGenerator


DEFAULT_MODEL_ENV = (
    ("FLAN_LARGE_MODEL_ID", "google/flan-t5-large"),
    ("FLAN_XL_MODEL_ID", "google/flan-t5-xl"),
    ("LLAMA_MODEL_ID", "meta-llama/Llama-2-7b-hf"),
    ("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
)


def _default_model_ids() -> List[str]:
    return [os.getenv(name, default).strip() for name, default in DEFAULT_MODEL_ENV]


def get_configured_model_ids() -> List[str]:
    configured = os.getenv("HF_MODEL_IDS", "")
    if not configured.strip():
        return _default_model_ids()

    model_ids = []
    for value in configured.split(","):
        model_id = value.strip()
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids or _default_model_ids()


def get_generator(model_id: str) -> BaseAnswerGenerator:
    return HuggingFaceAPIGenerator(model_id=model_id)

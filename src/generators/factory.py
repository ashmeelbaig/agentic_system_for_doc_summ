import os
from typing import List

from src.generators.base import BaseAnswerGenerator
from src.generators.hf_api_generator import HuggingFaceAPIGenerator


DEFAULT_MODEL_ENV = (
    ("QWEN_MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    ("LLAMA_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct:nscale"),
)

DEFAULT_WEAK_MODEL_IDS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

DEFAULT_HOSTED_MODEL_IDS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct:nscale",
]


def _parse_model_ids(value: str) -> List[str]:
    model_ids = []
    for item in value.split(","):
        model_id = item.strip()
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids


def _default_model_ids() -> List[str]:
    return [os.getenv(name, default).strip() for name, default in DEFAULT_MODEL_ENV]


def get_configured_model_ids() -> List[str]:
    configured = os.getenv("HF_MODEL_IDS", "")
    if not configured.strip():
        return _default_model_ids()

    model_ids = _parse_model_ids(configured)
    return model_ids or _default_model_ids()


def get_weak_model_ids() -> List[str]:
    """Return the optional weak-model experiment preset."""
    configured = os.getenv("WEAK_HF_MODEL_IDS", "")
    model_ids = _parse_model_ids(configured)
    return model_ids or list(DEFAULT_WEAK_MODEL_IDS)


def get_hf_model_ids() -> List[str]:
    configured = _parse_model_ids(os.getenv("HF_MODEL_IDS", ""))
    return configured or list(DEFAULT_HOSTED_MODEL_IDS)


def get_local_model_ids() -> List[str]:
    configured = _parse_model_ids(os.getenv("LOCAL_MODEL_IDS", ""))
    return configured


def get_all_configured_models() -> List[dict]:
    return [
        *(
            {"model_id": model_id, "provider_type": "hosted_hf"}
            for model_id in get_hf_model_ids()
        ),
        *(
            {"model_id": model_id, "provider_type": "local_transformers"}
            for model_id in get_local_model_ids()
        ),
    ]


def get_generator(
    model_id: str, provider_type: str = "hosted_hf"
) -> BaseAnswerGenerator:
    if provider_type == "local_transformers":
        # Keep Transformers out of normal hosted-only startup. Importing here also
        # guarantees no local model can be initialized without explicit dispatch.
        from src.generators.local_transformers_generator import (
            LocalTransformersGenerator,
        )

        return LocalTransformersGenerator(model_id=model_id)
    if provider_type == "hosted_hf":
        return HuggingFaceAPIGenerator(
            model_id=model_id,
            provider=os.getenv("HF_PROVIDER", "auto").strip() or "auto",
        )
    raise ValueError("Unsupported generator provider type.")

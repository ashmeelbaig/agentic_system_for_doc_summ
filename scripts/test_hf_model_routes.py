"""Smoke-test the active hosted Hugging Face model routes."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


MODEL_IDS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct:nscale",
]
# Optional future weak-model experiments (not active here):
# Qwen/Qwen2.5-0.5B-Instruct
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
PROMPT = "Answer in one sentence: What is Retrieval Augmented Generation?"
OUTPUT_PATH = Path("outputs/hf_model_route_test.json")


def _short_answer(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _chat_text(response: Any) -> str:
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        if isinstance(response, dict):
            return response["choices"][0]["message"]["content"]
        raise ValueError("unsupported chat response") from None


def _controlled_error(exc: Exception) -> str:
    message = str(exc).lower()
    if any(term in message for term in ("401", "403", "unauthorized", "forbidden")):
        return "Hugging Face model access was denied."
    if "timeout" in message or "timed out" in message:
        return "Hugging Face request timed out."
    if "429" in message or "rate limit" in message:
        return "Hugging Face rate limit reached."
    if any(term in message for term in ("provider", "not supported", "unavailable")):
        return "Hugging Face route is unavailable for this model."
    return "Hugging Face request failed."


def run_model_route(model_id: str, token: str) -> dict:
    try:
        client = InferenceClient(model=model_id, token=token, provider="auto")
    except TypeError:  # Compatibility with older huggingface_hub versions.
        client = InferenceClient(model=model_id, token=token)

    result = {"model_name": model_id, "status": "failed", "attempts": []}
    methods = ("chat_completion", "text_generation")
    for method in methods:
        print(f"model={model_id} method={method} status=attempting")
        try:
            if method == "chat_completion":
                response = client.chat_completion(
                    messages=[{"role": "user", "content": PROMPT}],
                    max_tokens=80,
                    temperature=0.1,
                )
                answer = _chat_text(response)
            else:
                answer = client.text_generation(
                    PROMPT,
                    max_new_tokens=80,
                    temperature=0.1,
                    do_sample=False,
                    return_full_text=False,
                )
            answer = _short_answer(answer)
            if not answer:
                raise ValueError("empty response")
            result.update(status="success", method=method, answer=answer)
            result.pop("error", None)
            result["attempts"].append({"method": method, "status": "success"})
            print(f"model={model_id} method={method} status=success answer={answer}")
            break
        except Exception as exc:
            error = _controlled_error(exc)
            result["attempts"].append(
                {"method": method, "status": "failed", "error": error}
            )
            result["error"] = error
            print(f"model={model_id} method={method} status=failed error={error}")
    return result


def main() -> int:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not configured; no requests were made.")
        return 1

    report = {
        "prompt": PROMPT,
        "models": [run_model_route(model, token) for model in MODEL_IDS],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={OUTPUT_PATH}")
    return 0 if all(item["status"] == "success" for item in report["models"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

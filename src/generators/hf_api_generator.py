import os
from typing import Any, List, Optional

from huggingface_hub import InferenceClient

from src.generators.base import BaseAnswerGenerator


REFUSAL_ANSWER = (
    "The retrieved documents do not provide enough reliable evidence "
    "to answer this question."
)


class HuggingFaceGenerationError(RuntimeError):
    """A controlled remote-generation failure that contains no credentials."""


class HuggingFaceAPIGenerator(BaseAnswerGenerator):
    def __init__(
        self,
        model_id: str,
        provider: str = "auto",
        timeout: float = 60.0,
        client=None,
    ):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise HuggingFaceGenerationError(
                "HF_TOKEN is not configured. Add it to the environment or local .env file."
            )

        self.model_id = model_id
        self.provider = "huggingface_api"
        self.hf_provider = provider
        self.attempted_methods: List[str] = []
        self.attempt_failures: List[dict] = []
        if client is not None:
            self.client = client
        else:
            try:
                self.client = InferenceClient(
                    model=model_id,
                    token=token,
                    provider=provider,
                    timeout=timeout,
                )
            except TypeError:  # Compatibility with older huggingface_hub clients.
                self.client = InferenceClient(
                    model=model_id,
                    token=token,
                    timeout=timeout,
                )

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        if isinstance(chunk, dict):
            return str(chunk.get("text", ""))
        if isinstance(chunk, tuple) and len(chunk) == 3:
            return str(chunk[1])
        return str(chunk)

    def _build_prompt(
        self,
        query: str,
        evidence_chunks: List[Any],
        instruction: Optional[str],
    ) -> str:
        evidence = "\n\n".join(
            self._chunk_text(chunk).strip()
            for chunk in evidence_chunks
            if self._chunk_text(chunk).strip()
        )
        revision = f"\nAdditional instruction: {instruction}\n" if instruction else ""

        # This plain completion prompt is intentional: the configured Llama model is
        # a base model and must not be assumed to support a chat template. It also
        # works as an instruction prompt for Qwen and text2text FLAN models.
        return (
            "Answer the question using only the evidence below.\n"
            "Give a direct answer in 1 to 3 sentences.\n"
            "Do not include URLs unless the question explicitly asks for them.\n"
            "Do not add unsupported information.\n"
            f"If the evidence is insufficient, answer exactly: {REFUSAL_ANSWER}\n"
            f"{revision}"
            f"Evidence:\n{evidence}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    @staticmethod
    def _clean_answer(text: Any, prompt: str) -> str:
        answer = str(text or "").strip()
        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()
        if answer.lower().startswith("answer:"):
            answer = answer.split(":", 1)[1].strip()
        return answer

    @staticmethod
    def _controlled_error(exc: Exception) -> HuggingFaceGenerationError:
        # Map only to fixed messages: raw HTTP errors may contain credentials.
        message = str(exc).lower()
        exception_name = type(exc).__name__.lower()
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None) or getattr(
            exc, "status_code", None
        )
        if status_code == 429 or "429" in message or "rate limit" in message or "too many requests" in message:
            detail = "Hugging Face rate limit reached."
        elif "timeout" in exception_name or "timeout" in message or "timed out" in message:
            detail = "Hugging Face generation timed out."
        elif status_code in (401, 403) or any(term in message for term in ("401", "403", "unauthorized", "forbidden", "gated", "access denied")):
            detail = "Hugging Face model access was denied. Check token permissions and model access."
        elif status_code == 404 or any(term in message for term in ("404", "not found", "does not exist", "unknown model")):
            detail = "Hugging Face model was not found. Check the model ID."
        elif (
            "not supported for task" in message and "provider" in message
        ) or status_code in (502, 503, 504) or exception_name == "stopiteration" or any(term in message for term in ("loading", "503", "unavailable", "no provider", "provider error")):
            detail = "Hugging Face provider is unavailable for this model."
        elif status_code in (400, 405, 422) or any(term in message for term in ("task", "method", "not supported", "unsupported", "attribute")):
            detail = "Hugging Face rejected the generation task method for this model."
        else:
            detail = "Hugging Face generation failed for an unclassified provider error."
        return HuggingFaceGenerationError(detail)

    def _method_order(self) -> List[str]:
        if "flan-t5" in self.model_id.lower():
            return ["text2text_generation", "text_generation"]
        return ["text_generation", "chat_completion"]

    def _invoke(self, method: str, prompt: str) -> Any:
        function = getattr(self.client, method, None)
        if not callable(function):
            raise AttributeError(f"{method} method is not supported")
        if method == "chat_completion":
            response = function(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=180,
                temperature=0.1,
            )
            try:
                return response.choices[0].message.content
            except (AttributeError, IndexError, TypeError):
                if isinstance(response, dict):
                    return response["choices"][0]["message"]["content"]
                raise ValueError("chat completion returned an unsupported response")
        return function(
            prompt,
            max_new_tokens=180,
            temperature=0.1,
            do_sample=False,
            return_full_text=False,
        )

    def generate_answer(
        self,
        query: str,
        evidence_chunks: List[Any],
        instruction: Optional[str] = None,
    ) -> str:
        prompt = self._build_prompt(query, evidence_chunks, instruction)
        controlled_errors: List[HuggingFaceGenerationError] = []
        self.attempted_methods = []
        self.attempt_failures = []
        for method in self._method_order():
            self.attempted_methods.append(method)
            try:
                answer = self._clean_answer(self._invoke(method, prompt), prompt)
                if not answer:
                    raise ValueError("provider returned an empty answer")
                return answer
            except Exception as exc:
                controlled = self._controlled_error(exc)
                controlled_errors.append(controlled)
                self.attempt_failures.append(
                    {"method": method, "error": str(controlled)}
                )

        # Authentication and service failures affect every fallback and therefore
        # outrank a task-method failure from an earlier attempt.
        selected = None
        for marker in ("access was denied", "rate limit", "timed out", "not found"):
            selected = next(
                (error for error in controlled_errors if marker in str(error).lower()),
                None,
            )
            if selected:
                break
        if selected is None:
            selected = next(
                (
                    error
                    for error in controlled_errors
                    if "unclassified" not in str(error).lower()
                ),
                controlled_errors[-1] if controlled_errors else self._controlled_error(RuntimeError()),
            )
        raise selected from None

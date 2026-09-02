import gc
import os
from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generators.base import BaseAnswerGenerator
from src.generators.hf_api_generator import REFUSAL_ANSWER


class LocalTransformersGenerationError(RuntimeError):
    """A controlled local-generation failure with no credential details."""


class LocalTransformersGenerator(BaseAnswerGenerator):
    """Lazily load one causal language model from the Hugging Face cache."""

    provider = "local_transformers"

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        tokenizer=None,
        model=None,
    ):
        self.model_id = model_id
        self.device_setting = device or os.getenv("LOCAL_MODEL_DEVICE", "auto")
        self.dtype_setting = dtype or os.getenv("LOCAL_MODEL_DTYPE", "auto")
        self.tokenizer = tokenizer
        self.model = model
        self.device = None

    def _load(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            self.device = self._model_device()
            return

        kwargs = {}
        token = os.getenv("HF_TOKEN")
        if token:
            kwargs["token"] = token

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            model_kwargs = dict(kwargs)
            if self.dtype_setting == "auto":
                model_kwargs["torch_dtype"] = "auto"
            else:
                dtype = getattr(torch, self.dtype_setting, None)
                if dtype is None:
                    raise ValueError("unsupported LOCAL_MODEL_DTYPE")
                model_kwargs["torch_dtype"] = dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **model_kwargs
            )
            self.device = self._selected_device()
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.release()
            raise LocalTransformersGenerationError(
                "Local Transformers model could not be loaded."
            ) from None

    def _selected_device(self):
        if self.device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_setting)

    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return self._selected_device()

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        if isinstance(chunk, dict):
            return str(chunk.get("text", ""))
        if isinstance(chunk, tuple) and len(chunk) == 3:
            return str(chunk[1])
        return str(chunk)

    def _instruction_prompt(
        self, query: str, evidence_chunks: List[Any], instruction: Optional[str]
    ) -> str:
        evidence = "\n\n".join(
            text.strip()
            for text in (self._chunk_text(chunk) for chunk in evidence_chunks)
            if text.strip()
        )
        revision = f"\nAdditional instruction: {instruction}" if instruction else ""
        return (
            "Answer using only the evidence. Give a direct answer in 1 to 3 sentences.\n"
            f"If evidence is insufficient, answer exactly: {REFUSAL_ANSWER}"
            f"{revision}\nEvidence:\n{evidence}\n\nQuestion: {query}\nAnswer:"
        )

    def _format_prompt(self, instruction_prompt: str) -> str:
        chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(chat_template):
            try:
                return chat_template(
                    [{"role": "user", "content": instruction_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (AttributeError, TypeError, ValueError):
                pass
        return instruction_prompt

    def generate_answer(
        self,
        query: str,
        evidence_chunks: List[Any],
        instruction: Optional[str] = None,
    ) -> str:
        try:
            self._load()
            prompt = self._format_prompt(
                self._instruction_prompt(query, evidence_chunks, instruction)
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {
                name: tensor.to(self.device) if hasattr(tensor, "to") else tensor
                for name, tensor in inputs.items()
            }
            input_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=180,
                    temperature=0.1,
                    do_sample=False,
                )
            answer = self.tokenizer.decode(
                output[0][input_length:], skip_special_tokens=True
            ).strip()
            if not answer:
                raise ValueError("empty local generation")
            return answer
        except LocalTransformersGenerationError:
            raise
        except Exception:
            raise LocalTransformersGenerationError(
                "Local Transformers generation failed."
            ) from None

    def release(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

"""Reusable backend pipeline for CLI and UI entry points."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.answer_revision_agent import decide_answer_revision, final_safety_gate
from src.claim_extractor import extract_claims
from src.document_collection import prepare_metadata_chunks_from_pdfs
from src.generators.factory import get_all_configured_models, get_configured_model_ids
from src.generators.multi_model_runner import run_all_generators
from src.nli_verifier import NLIClaimVerifier
from src.reranker import EvidenceReranker
from src.retrieval_retry import confidence_to_dict, retrieve_with_retries
from src.result_saver import (
    normalize_retrieved_evidence_item,
    save_multi_model_result_to_json,
)
from src.retriever import FaissRetriever
from src.safety_guardrails import (
    check_user_query_safety,
    detect_prompt_injection,
    sanitize_evidence_text,
)


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
REFUSAL_ANSWER = (
    "The retrieved documents do not provide enough reliable evidence "
    "to answer this question."
)


class PipelineConfigurationError(RuntimeError):
    """A safe configuration error suitable for CLI or UI display."""


def available_pdf_paths(data_dir: Path = DATA_DIR) -> List[Path]:
    return sorted(data_dir.glob("*.pdf")) if data_dir.exists() else []


def _resolve_pdf_paths(selected_pdf_paths: List[str]) -> List[Path]:
    paths = [Path(value) for value in (selected_pdf_paths or [])]
    if not paths:
        raise PipelineConfigurationError("Please upload at least one PDF file.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PipelineConfigurationError("One or more selected PDF files do not exist.")
    return paths


def sanitize_retrieved_evidence(results):
    """Sanitize document instructions while preserving evidence metadata."""
    sanitized = []
    matches = []
    for chunk in results:
        if isinstance(chunk, dict):
            text = str(chunk.get("text", ""))
            updated = dict(chunk)
            updated["text"] = sanitize_evidence_text(text)
        elif isinstance(chunk, tuple) and len(chunk) == 3:
            chunk_index, text, score = chunk
            text = str(text)
            updated = (chunk_index, sanitize_evidence_text(text), score)
        else:
            text = ""
            updated = chunk
        for pattern in detect_prompt_injection(text)["matched_patterns"]:
            if pattern not in matches:
                matches.append(pattern)
        sanitized.append(updated)
    return sanitized, matches


def _rerank(query, candidate_chunks, reranker, top_k=4):
    return reranker.rerank(
        query=query, retrieved_chunks=candidate_chunks, top_k=top_k
    )


def run_models_for_evidence(
    query: str,
    evidence_chunks,
    retrieval_confidence: Dict[str, Any],
    generator_mode: str,
    claim_verifier,
) -> Dict[str, Any]:
    """Run independently guarded models over one already-retrieved evidence set."""
    if generator_mode == "multi_hf":
        models = get_configured_model_ids()
        local_models_skipped = False
    elif generator_mode == "multi_model":
        models = get_all_configured_models()
        local_models_skipped = not any(
            item["provider_type"] == "local_transformers" for item in models
        )
    else:
        raise PipelineConfigurationError(
            "The reusable pipeline currently supports multi_hf and multi_model modes."
        )

    result = run_all_generators(
        query=query,
        evidence_chunks=evidence_chunks,
        model_ids=models,
        claim_extractor=extract_claims,
        claim_verifier=claim_verifier,
        revision_function=decide_answer_revision,
        final_safety_gate=final_safety_gate,
        retrieval_confidence=retrieval_confidence,
    )
    result["local_models_skipped"] = local_models_skipped
    return result


def _save_result(
    pdf_name: str,
    query: str,
    generator_mode: str,
    retrieval: Dict[str, Any],
    model_results: Dict[str, Any],
    model_comparison: List[Dict[str, Any]],
    save_output: bool,
) -> Dict[str, Any]:
    result = {
        "pdf_name": pdf_name,
        "query": query,
        "generator_mode": generator_mode,
        "retrieval": retrieval,
        "model_results": model_results,
        "model_comparison": model_comparison,
        "saved_output_path": None,
    }
    if save_output:
        saved_path = save_multi_model_result_to_json(
            output_dir=str(OUTPUT_DIR),
            pdf_name=pdf_name,
            query=query,
            retrieval=retrieval,
            model_results=model_results,
            model_comparison=model_comparison,
            generator_mode=generator_mode,
        )
        result["saved_output_path"] = str(saved_path)
    return result


def answer_question(
    query: str,
    selected_pdf_paths: List[str],
    generator_mode: Optional[str] = None,
    save_output: bool = True,
) -> dict:
    """Run the complete claim-grounded multi-model pipeline for one question."""
    load_dotenv()
    query = (query or "").strip()
    if not query:
        raise ValueError("Please enter a question.")

    pdf_paths = _resolve_pdf_paths(selected_pdf_paths)

    mode = (generator_mode or os.getenv("GENERATOR_MODE", "multi_hf")).strip().lower()
    if mode not in {"multi_hf", "multi_model"}:
        raise PipelineConfigurationError(
            "The Streamlit pipeline supports multi_hf or multi_model mode."
        )
    if not os.getenv("HF_TOKEN", "").strip():
        raise PipelineConfigurationError(
            "Hugging Face token is missing. Please add HF_TOKEN to your .env file."
        )

    documents = prepare_metadata_chunks_from_pdfs(pdf_paths, chunk_size=700, overlap=120)
    pdf_name = pdf_paths[0].name if len(pdf_paths) == 1 else "Multiple PDFs"

    query_safety = check_user_query_safety(query)
    if not query_safety["is_safe"]:
        retrieval = {
            "used_query": query,
            "retrieval_confidence": {"label": "not_run", "reason": query_safety["reason"]},
            "retrieval_attempts": [],
            "retrieved_evidence": [],
            "query_safety": query_safety,
        }
        return _save_result(pdf_name, query, mode, retrieval, {}, [], save_output)

    retriever = FaissRetriever()
    retriever.build_index(documents["chunks"])
    reranker = EvidenceReranker()
    retrieval_result = retrieve_with_retries(
        original_query=query,
        retriever=retriever,
        reranker=reranker,
        rerank_function=_rerank,
        max_attempts=3,
        retrieve_top_k=12,
        rerank_top_k=4,
    )
    evidence, injection_matches = sanitize_retrieved_evidence(
        retrieval_result["results"]
    )
    confidence = confidence_to_dict(retrieval_result["confidence"])
    retrieval = {
        "used_query": retrieval_result["used_query"],
        "retrieval_confidence": confidence,
        "retrieval_attempts": retrieval_result["attempts"],
        "retrieved_evidence": [
            normalize_retrieved_evidence_item(item) for item in evidence
        ],
        "document_prompt_injection_detected": bool(injection_matches),
        "prompt_injection_matches": injection_matches,
    }

    if not retrieval_result["should_answer"]:
        refusal = {
            model_id: {
                "status": "success",
                "model_name": model_id,
                "provider": "huggingface_api",
                "draft_answer": REFUSAL_ANSWER,
                "candidate_final_answer": REFUSAL_ANSWER,
                "final_answer": REFUSAL_ANSWER,
                "is_refusal_answer": True,
                "extracted_claims": [],
                "atomic_claims": [],
                "verification_results": [],
                "faithfulness_score": {"faithfulness_score": 0.0, "total_claims": 0},
                "revision_decision": {"decision": "refuse"},
                "final_safety_gate": {"is_safe": False, "action": "refuse"},
                "latency_seconds": 0.0,
            }
            for model_id in get_configured_model_ids()
        }
        comparison = [
            {
                "model_name": model_id,
                "status": item["status"],
                "provider": item["provider"],
                "faithfulness_score": 0.0,
                "revision_decision": "refuse",
                "safety_action": "refuse",
                "is_refusal_answer": True,
                "latency_seconds": 0.0,
            }
            for model_id, item in refusal.items()
        ]
        return _save_result(
            pdf_name, query, mode, retrieval, refusal, comparison, save_output
        )

    verifier = NLIClaimVerifier()
    generated = run_models_for_evidence(query, evidence, confidence, mode, verifier)
    return _save_result(
        pdf_name,
        query,
        mode,
        retrieval,
        generated["model_results"],
        generated["model_comparison"],
        save_output,
    )

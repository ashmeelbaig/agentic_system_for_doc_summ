from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Tuple

try:
    from crewai import Agent as CrewAIAgent  # type: ignore
except ImportError:  # CrewAI is optional for this prototype.
    CrewAIAgent = None

from src.answer_revision_agent import (
    build_revision_query,
    decide_answer_revision,
    final_safety_gate,
)
from src.claim_extractor import extract_claims
from src.retrieval_retry import confidence_to_dict, retrieve_with_retries
from src.safety_guardrails import (
    check_user_query_safety,
    detect_prompt_injection,
    sanitize_evidence_text,
)
from src.scoring import calculate_faithfulness_score


CREWAI_AVAILABLE = CrewAIAgent is not None
REFUSAL_ANSWER = (
    "The retrieved documents do not provide enough reliable evidence "
    "to answer this question."
)
UNSAFE_QUERY_REFUSAL = (
    "I cannot help with this request because it asks for unsafe or "
    "non-document behaviour."
)


@dataclass(frozen=True)
class AgentRole:
    name: str
    responsibility: str


AGENT_WORKFLOW: Tuple[AgentRole, ...] = (
    AgentRole("Safety Agent", "Check query safety and final answer safety."),
    AgentRole("Retrieval Agent", "Retrieve, rerank, and retry weak evidence."),
    AgentRole("Answer Agent", "Generate an evidence-grounded answer."),
    AgentRole("Verification Agent", "Extract and verify claims, then score them."),
    AgentRole("Revision Agent", "Request one focused revision when required."),
)


def _empty_score_summary() -> Dict[str, Any]:
    return calculate_faithfulness_score([])


def _decision_dict(decision: Any) -> Dict[str, Any]:
    return asdict(decision)


def _sanitize_results(results):
    sanitized_results = []
    matches = []

    for chunk in results:
        if isinstance(chunk, dict):
            text = str(chunk.get("text", ""))
            sanitized_chunk = dict(chunk)
            sanitized_chunk["text"] = sanitize_evidence_text(text)
        elif isinstance(chunk, tuple) and len(chunk) == 3:
            chunk_index, text, score = chunk
            text = str(text)
            sanitized_chunk = (chunk_index, sanitize_evidence_text(text), score)
        else:
            text = ""
            sanitized_chunk = chunk

        for pattern in detect_prompt_injection(text)["matched_patterns"]:
            if pattern not in matches:
                matches.append(pattern)
        sanitized_results.append(sanitized_chunk)

    return sanitized_results, matches


def _refusal_result(
    query: str,
    answer: str,
    reason: str,
    retrieval_attempts=None,
    retrieval_confidence=None,
) -> Dict[str, Any]:
    return {
        "query": query,
        "draft_answer": "",
        "final_answer": answer,
        "retrieval_attempts": retrieval_attempts or [],
        "retrieval_confidence": retrieval_confidence,
        "revision_decision": None,
        "final_safety_gate": {
            "is_safe": False,
            "reason": reason,
            "action": "refuse",
        },
        "claims": [],
        "verification_results": [],
        "score_summary": _empty_score_summary(),
        "is_refused": True,
    }


def run_crewai_claim_grounded_workflow(
    query,
    retriever,
    reranker,
    answer_generator,
    claim_verifier,
    rerank_function: Callable,
):
    """Run the existing deterministic RAG guardrails as an agent workflow."""

    query_safety = check_user_query_safety(query)
    if not query_safety["is_safe"]:
        result = _refusal_result(
            query, UNSAFE_QUERY_REFUSAL, query_safety["reason"]
        )
        result["query_safety"] = query_safety
        return result

    retrieval = retrieve_with_retries(
        original_query=query,
        retriever=retriever,
        reranker=reranker,
        rerank_function=rerank_function,
        max_attempts=3,
        retrieve_top_k=12,
        rerank_top_k=4,
    )
    confidence = confidence_to_dict(retrieval["confidence"])

    if not retrieval["should_answer"]:
        return _refusal_result(
            query=query,
            answer=REFUSAL_ANSWER,
            reason="Retrieval confidence remained low after three attempts.",
            retrieval_attempts=retrieval["attempts"],
            retrieval_confidence=confidence,
        )

    results, injection_matches = _sanitize_results(retrieval["results"])
    draft_answer = answer_generator.generate_answer(query, results)
    claims = extract_claims(draft_answer)
    verification_results = claim_verifier.verify_claims(claims, results)
    score_summary = calculate_faithfulness_score(verification_results)
    revision = decide_answer_revision(
        query, draft_answer, verification_results, score_summary
    )

    candidate_final_answer = draft_answer
    final_claims = claims
    final_verification_results = verification_results
    final_score_summary = score_summary

    if revision.decision == "revise":
        revision_query = build_revision_query(query, revision.instruction)
        candidate_final_answer = answer_generator.generate_answer(
            revision_query, results
        )
        final_claims = extract_claims(candidate_final_answer)
        final_verification_results = claim_verifier.verify_claims(
            final_claims, results
        )
        final_score_summary = calculate_faithfulness_score(
            final_verification_results
        )

    safety = final_safety_gate(
        query,
        candidate_final_answer,
        final_verification_results,
        final_score_summary,
    )
    final_answer = candidate_final_answer if safety.is_safe else REFUSAL_ANSWER

    return {
        "query": query,
        "draft_answer": draft_answer,
        "final_answer": final_answer,
        "retrieval_attempts": retrieval["attempts"],
        "retrieval_confidence": confidence,
        "revision_decision": _decision_dict(revision),
        "final_safety_gate": _decision_dict(safety),
        "claims": final_claims,
        "verification_results": final_verification_results,
        "score_summary": final_score_summary,
        "is_refused": not safety.is_safe,
        "document_prompt_injection_detected": bool(injection_matches),
        "prompt_injection_matches": injection_matches,
    }

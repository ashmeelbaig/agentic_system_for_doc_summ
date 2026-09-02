from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable

from src.answer_revision_agent import is_refusal_answer
from src.generators.factory import get_generator
from src.scoring import calculate_faithfulness_score


REFUSAL_ANSWER = (
    "The retrieved documents do not provide enough reliable evidence "
    "to answer this question."
)


def _verify(answer, evidence_chunks, claim_extractor, claim_verifier):
    if is_refusal_answer(answer):
        return [], [], calculate_faithfulness_score([])
    claims = claim_extractor(answer)
    results = claim_verifier.verify_claims(claims, evidence_chunks)
    return claims, results, calculate_faithfulness_score(results)


def _decision_dict(value):
    return asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value)


def run_all_generators(
    query,
    evidence_chunks,
    model_ids: Iterable[str],
    claim_extractor,
    claim_verifier,
    revision_function,
    final_safety_gate,
    retrieval_confidence=None,
    generator_factory: Callable[[str], Any] = get_generator,
) -> Dict[str, Any]:
    """Run independent answer guardrails for many models over shared evidence."""
    model_results = {}

    for model_id in model_ids:
        generator = None
        try:
            generator = generator_factory(model_id)
            draft = generator.generate_answer(query, evidence_chunks)
            claims, verification, score = _verify(
                draft, evidence_chunks, claim_extractor, claim_verifier
            )
            revision = revision_function(
                query,
                draft,
                verification,
                score,
                retrieval_confidence,
            )

            candidate = draft
            final_claims = claims
            final_verification = verification
            final_score = score

            if revision.decision == "revise":
                candidate = generator.generate_answer(
                    query, evidence_chunks, instruction=revision.instruction
                )
                final_claims, final_verification, final_score = _verify(
                    candidate, evidence_chunks, claim_extractor, claim_verifier
                )

            safety = final_safety_gate(
                query, candidate, final_verification, final_score
            )
            final_answer = candidate if safety.is_safe else REFUSAL_ANSWER
            model_results[model_id] = {
                "status": "success",
                "model_name": model_id,
                "provider": getattr(generator, "provider", "huggingface_api"),
                "attempted_methods": list(getattr(generator, "attempted_methods", [])),
                "attempt_failures": list(getattr(generator, "attempt_failures", [])),
                "draft_answer": draft,
                "candidate_final_answer": candidate,
                "final_answer": final_answer,
                "is_refusal_answer": is_refusal_answer(final_answer),
                "extracted_claims": final_claims,
                "atomic_claims": final_claims,
                "verification_results": final_verification,
                "faithfulness_score": final_score,
                "revision_decision": _decision_dict(revision),
                "final_safety_gate": _decision_dict(safety),
            }
        except Exception as exc:
            # Generator exceptions are intentionally normalized by the API adapter.
            # Keep this fallback generic so raw errors can never leak credentials.
            model_results[model_id] = {
                "status": "failed",
                "model_name": model_id,
                "provider": getattr(generator, "provider", "huggingface_api"),
                "attempted_methods": list(
                    getattr(generator, "attempted_methods", [])
                ),
                "attempt_failures": list(
                    getattr(generator, "attempt_failures", [])
                ),
                "error": str(exc) if exc.__class__.__name__ == "HuggingFaceGenerationError" else "Model generation failed.",
            }

    comparison = []
    for model_id, result in model_results.items():
        score = result.get("faithfulness_score", {})
        revision = result.get("revision_decision", {})
        safety = result.get("final_safety_gate", {})
        comparison.append({
            "model_name": model_id,
            "status": result["status"],
            "faithfulness_score": score.get("faithfulness_score"),
            "revision_decision": revision.get("decision"),
            "safety_action": safety.get("action"),
            "is_refusal_answer": result.get("is_refusal_answer"),
        })

    return {"model_results": model_results, "model_comparison": comparison}

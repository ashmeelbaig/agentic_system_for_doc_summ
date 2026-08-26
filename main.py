from pathlib import Path
import os

from src.retrieval_retry import retrieve_with_retries, confidence_to_dict
from src.safety_guardrails import (
    check_user_query_safety,
    detect_prompt_injection,
    sanitize_evidence_text,
)
from src.answer_revision_agent import (
    build_revision_query,
    decide_answer_revision,
    final_safety_gate,
)
from src.document_collection import prepare_metadata_chunks_from_pdfs
from src.document_loader import load_pdf_pages
from src.chunker import chunk_pages_with_metadata
from src.retriever import FaissRetriever
from src.reranker import EvidenceReranker
from src.generator import AnswerGenerator
from src.claim_extractor import extract_claims
from src.nli_verifier import NLIClaimVerifier
from src.scoring import calculate_faithfulness_score
from src.result_saver import save_result_to_json
from src.baseline import create_baseline_result, print_baseline_result
from src.display import (
    print_header,
    print_document_status,
    print_evidence_summary,
    print_generated_answer,
    print_claim_table,
    print_score_summary,
)


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def list_pdf_files(data_dir: Path):
    """
    List all PDF files in the data folder.
    """

    if not data_dir.exists():
        raise FileNotFoundError("Data folder does not exist.")

    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the data folder.")

    return pdf_files


def get_selected_pdf_paths(pdf_files, choice: str):
    """
    Convert terminal PDF selection into selected PDF path list.

    Choice '0' means use all PDFs.
    Choice '1', '2', etc. means use one selected PDF.
    """

    if choice == "0":
        return pdf_files

    if not choice.isdigit():
        raise ValueError("Invalid PDF selection.")

    selected_index = int(choice)

    if 1 <= selected_index <= len(pdf_files):
        return [pdf_files[selected_index - 1]]

    raise ValueError("Invalid PDF selection.")


def choose_pdf_files(pdf_files):
    """
    Allow the user to select one PDF file or all PDF files from terminal.
    """

    print_header("Available PDF Files")

    print("0. Use all PDFs")

    for index, pdf_file in enumerate(pdf_files, start=1):
        print(f"{index}. {pdf_file.name}")

    while True:
        choice = input("\nSelect a PDF by number, or enter 0 for all PDFs: ")

        try:
            return get_selected_pdf_paths(pdf_files, choice)
        except ValueError:
            print("Invalid selection. Please enter a valid number.")


def prepare_metadata_chunks_from_pdf(
    pdf_path: Path,
    chunk_size: int = 700,
    overlap: int = 120,
):
    """
    Load PDF pages and create metadata-aware chunks.
    """

    pages = load_pdf_pages(str(pdf_path))

    chunks = chunk_pages_with_metadata(
        pages=pages,
        source=pdf_path.name,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    total_chars = sum(len(page.get("text", "")) for page in pages)

    return pages, chunks, total_chars


def prepare_metadata_chunks_from_selected_pdfs(
    selected_pdfs,
    chunk_size: int = 700,
    overlap: int = 120,
):
    """
    Prepare metadata-aware chunks from one or multiple selected PDFs.
    """

    result = prepare_metadata_chunks_from_pdfs(
        pdf_paths=selected_pdfs,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if len(selected_pdfs) == 1:
        display_name = selected_pdfs[0].name
    else:
        display_name = "Multiple PDFs"

    result["display_name"] = display_name

    return result


def get_generator_mode():
    """
    Return selected generator mode.

    Supported modes:
    - fast
    - quality
    """

    mode = os.getenv("GENERATOR_MODE", "quality").lower().strip()

    if mode in {"fast", "quality"}:
        return mode

    return "quality"


def get_generator_model_name():
    """
    Select the answer generation model based on GENERATOR_MODE.

    Supported modes:
    - fast: lightweight local testing
    - quality: stronger answer generation
    """

    mode = get_generator_mode()

    if mode == "fast":
        return "google/flan-t5-small"

    return "google/flan-t5-base"


def rerank_candidate_evidence(
    query: str,
    candidate_chunks,
    reranker,
    top_k: int = 4,
):
    """
    Rerank FAISS candidate evidence and return strongest chunks.
    """

    return reranker.rerank(
        query=query,
        retrieved_chunks=candidate_chunks,
        top_k=top_k,
    )


def verify_claims_with_selected_verifier(
    claims,
    retrieved_chunks,
    verifier,
):
    """
    Verify claims using the selected verifier.
    """

    return verifier.verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
    )


def make_empty_score_summary():
    """
    Score summary used when the system refuses before generation.
    """

    return {
        "total_claims": 0,
        "supported_claims": 0,
        "partially_supported_claims": 0,
        "unsupported_claims": 0,
        "contradicted_claims": 0,
        "not_enough_evidence_claims": 0,
        "faithfulness_score": 0.0,
    }


def print_retrieval_attempts(attempts):
    """
    Print retrieval retry attempt details.
    """

    print("\nRetrieval Attempts")
    print("------------------")

    for attempt in attempts:
        confidence = attempt["confidence"]

        print(f"Attempt {attempt['attempt']}:")
        print(f"Query: {attempt['query']}")
        print(f"Confidence label: {confidence['label']}")
        print(f"Reason: {confidence['reason']}")
        print(f"Top rerank score: {confidence['top_rerank_score']:.4f}")
        print(f"Average rerank score: {confidence['avg_rerank_score']:.4f}")
        print(f"Keyword coverage: {confidence['keyword_coverage']:.4f}")
        print("")


def main():
    print_header("Claim Grounded Agentic RAG Prototype")

    pdf_files = list_pdf_files(DATA_DIR)
    selected_pdfs = choose_pdf_files(pdf_files)

    print("\nProcessing selected PDF. Please wait...")

    document_result = prepare_metadata_chunks_from_selected_pdfs(
        selected_pdfs=selected_pdfs,
        chunk_size=700,
        overlap=120,
    )

    chunks = document_result["chunks"]
    total_chars = document_result["total_chars"]
    display_name = document_result["display_name"]

    print_document_status(
        pdf_name=display_name,
        total_chars=total_chars,
        total_chunks=len(chunks),
    )

    print("\nLoading retrieval model and building FAISS index...")
    retriever = FaissRetriever()
    retriever.build_index(chunks)

    print("\nLoading evidence reranker...")
    reranker = EvidenceReranker()

    generator_model_name = get_generator_model_name()

    print(f"\nLoading answer generation model: {generator_model_name}")
    answer_generator = AnswerGenerator(model_name=generator_model_name)

    print("\nLoading NLI claim verifier...")
    claim_verifier = NLIClaimVerifier()

    print_header("System Ready")
    print("You can now ask questions about the selected document.")
    print("Type 'exit' to stop the prototype.")

    while True:
        query = input("\nAsk a question: ").strip()

        if query.lower() == "exit":
            print("\nExiting prototype.")
            break

        if not query:
            print("Please enter a valid question.")
            continue

        query_safety = check_user_query_safety(query)

        if not query_safety["is_safe"]:
            answer = (
                "I cannot help with this request because it asks for unsafe or "
                "non-document behaviour."
            )
            score_summary = make_empty_score_summary()
            baseline_result = create_baseline_result(query, answer, [])
            baseline_result["safety_guardrail"] = query_safety
            baseline_result["is_refused"] = True
            baseline_result["refusal_reason"] = query_safety["reason"]

            print_generated_answer(answer)
            saved_file = save_result_to_json(
                output_dir=str(OUTPUT_DIR),
                pdf_name=display_name,
                query=query,
                answer=answer,
                retrieved_chunks=[],
                claims=[],
                verification_results=[],
                score_summary=score_summary,
                baseline_result=baseline_result,
                generator_metadata={
                    "mode": get_generator_mode(),
                    "model_name": generator_model_name,
                },
            )
            print("\nResult saved successfully.")
            print(f"Saved file: {saved_file}")
            continue

        # ---------------------------------------------------------
        # 1. Retrieval guardrail with three attempts
        # Attempt 1: original query
        # Attempt 2: rewritten query
        # Attempt 3: keyword query
        # ---------------------------------------------------------
        retrieval_result = retrieve_with_retries(
            original_query=query,
            retriever=retriever,
            reranker=reranker,
            rerank_function=rerank_candidate_evidence,
            max_attempts=3,
            retrieve_top_k=12,
            rerank_top_k=4,
        )

        results = retrieval_result["results"]
        retrieval_confidence = retrieval_result["confidence"]
        retrieval_confidence_dict = confidence_to_dict(retrieval_confidence)
        retrieval_attempts = retrieval_result["attempts"]
        used_query = retrieval_result["used_query"]

        prompt_injection_matches = []
        sanitized_results = []

        for chunk in results:
            if isinstance(chunk, dict):
                chunk_text = str(chunk.get("text", ""))
                sanitized_chunk = dict(chunk)
                sanitized_chunk["text"] = sanitize_evidence_text(chunk_text)
            elif isinstance(chunk, tuple) and len(chunk) == 3:
                chunk_index, chunk_text, score = chunk
                chunk_text = str(chunk_text)
                sanitized_chunk = (
                    chunk_index,
                    sanitize_evidence_text(chunk_text),
                    score,
                )
            else:
                chunk_text = ""
                sanitized_chunk = chunk

            detection = detect_prompt_injection(chunk_text)
            for pattern in detection["matched_patterns"]:
                if pattern not in prompt_injection_matches:
                    prompt_injection_matches.append(pattern)

            sanitized_results.append(sanitized_chunk)

        results = sanitized_results

        print_retrieval_attempts(retrieval_attempts)

        print("\nSelected Evidence After Retrieval Guardrail")
        print("------------------------------------------")
        print(f"Used query: {used_query}")
        print(f"Final retrieval confidence: {retrieval_confidence.label}")
        print_evidence_summary(results)

        # ---------------------------------------------------------
        # 2. Refuse before generation if retrieval is still weak
        # ---------------------------------------------------------
        if not retrieval_result["should_answer"]:
            answer = (
                "The retrieved documents do not provide enough reliable evidence "
                "to answer this question."
            )

            baseline_result = create_baseline_result(
                query=query,
                answer=answer,
                retrieved_chunks=results,
            )

            if isinstance(baseline_result, dict):
                baseline_result["retrieval_confidence"] = retrieval_confidence_dict
                baseline_result["retrieval_attempts"] = retrieval_attempts
                baseline_result["used_query"] = used_query
                baseline_result["is_refused"] = True
                baseline_result["document_prompt_injection_detected"] = bool(
                    prompt_injection_matches
                )
                baseline_result["prompt_injection_matches"] = prompt_injection_matches
                baseline_result["refusal_reason"] = (
                    "Retrieval confidence remained low after three attempts."
                )
                baseline_result["draft_answer"] = answer
                baseline_result["candidate_final_answer"] = answer
                baseline_result["final_answer"] = answer
                baseline_result["final_safety_gate"] = {
                    "is_safe": False,
                    "reason": "Retrieval confidence remained low after three attempts.",
                    "action": "refuse",
                }
                baseline_result["final_verification_summary"] = {
                    "total_claims": 0,
                    "supported_claims": 0,
                    "contradicted_claims": 0,
                    "not_enough_evidence_claims": 0,
                    "faithfulness_score": 0.0,
                }
                baseline_result["revision_decision"] = {
                    "decision": "refuse",
                    "reason": "Retrieval confidence remained low after three attempts.",
                    "instruction": "No answer was generated because evidence was not strong enough.",
                    "answer_focus": "not_applicable",
                    "should_reverify": False,
                }

            print_baseline_result(baseline_result)
            print_generated_answer(answer)

            claims = []
            verification_results = []
            score_summary = make_empty_score_summary()

            print_score_summary(score_summary)

            saved_file = save_result_to_json(
                output_dir=str(OUTPUT_DIR),
                pdf_name=display_name,
                query=query,
                answer=answer,
                retrieved_chunks=results,
                claims=claims,
                verification_results=verification_results,
                score_summary=score_summary,
                baseline_result=baseline_result,
                generator_metadata={
                    "mode": get_generator_mode(),
                    "model_name": generator_model_name,
                },
            )

            print("\nResult saved successfully.")
            print(f"Saved file: {saved_file}")

            continue

        # ---------------------------------------------------------
        # 3. Generate draft answer
        # ---------------------------------------------------------
        answer = answer_generator.generate_answer(
            query=query,
            retrieved_chunks=results,
        )

        baseline_result = create_baseline_result(
            query=query,
            answer=answer,
            retrieved_chunks=results,
        )

        if isinstance(baseline_result, dict):
            baseline_result["retrieval_confidence"] = retrieval_confidence_dict
            baseline_result["retrieval_attempts"] = retrieval_attempts
            baseline_result["used_query"] = used_query
            baseline_result["is_refused"] = False
            baseline_result["document_prompt_injection_detected"] = bool(
                prompt_injection_matches
            )
            baseline_result["prompt_injection_matches"] = prompt_injection_matches

        print_baseline_result(baseline_result)
        print_generated_answer(answer)

        # ---------------------------------------------------------
        # 4. Extract claims
        # ---------------------------------------------------------
        claims = extract_claims(answer)

        # ---------------------------------------------------------
        # 5. Verify draft claims with NLI
        # ---------------------------------------------------------
        verification_results = verify_claims_with_selected_verifier(
            claims=claims,
            retrieved_chunks=results,
            verifier=claim_verifier,
        )

        print_claim_table(verification_results)

        # ---------------------------------------------------------
        # 6. Calculate draft faithfulness score
        # ---------------------------------------------------------
        score_summary = calculate_faithfulness_score(verification_results)

        print_score_summary(score_summary)

        # ---------------------------------------------------------
        # 7. Answer Revision Agent V1
        # ---------------------------------------------------------
        revision_decision = decide_answer_revision(
            query=query,
            answer=answer,
            verification_results=verification_results,
            score_summary=score_summary,
        )

        print("\nAnswer Revision Decision")
        print("------------------------")
        print(f"Decision: {revision_decision.decision}")
        print(f"Reason: {revision_decision.reason}")
        print(f"Answer focus: {revision_decision.answer_focus}")

        draft_answer = answer
        final_answer = answer
        final_claims = claims
        final_verification_results = verification_results
        final_score_summary = score_summary

        # ---------------------------------------------------------
        # 8. One revision attempt for efficiency
        # ---------------------------------------------------------
        if revision_decision.decision == "revise":
            print("\nRevising answer once using revision instructions...")

            revision_query = build_revision_query(
                original_query=query,
                instruction=revision_decision.instruction,
            )

            revised_answer = answer_generator.generate_answer(
                query=revision_query,
                retrieved_chunks=results,
            )

            print_generated_answer(revised_answer)

            revised_claims = extract_claims(revised_answer)

            revised_verification_results = verify_claims_with_selected_verifier(
                claims=revised_claims,
                retrieved_chunks=results,
                verifier=claim_verifier,
            )

            print_claim_table(revised_verification_results)

            revised_score_summary = calculate_faithfulness_score(
                revised_verification_results
            )

            print_score_summary(revised_score_summary)

            final_answer = revised_answer
            final_claims = revised_claims
            final_verification_results = revised_verification_results
            final_score_summary = revised_score_summary

        # ---------------------------------------------------------
        # 9. Final safety gate
        # ---------------------------------------------------------
        candidate_final_answer = final_answer
        final_safety_decision = final_safety_gate(
            query=query,
            final_answer=candidate_final_answer,
            final_verification_results=final_verification_results,
            final_score_summary=final_score_summary,
        )

        print("\nFinal Safety Gate")
        print("-----------------")
        print(f"Action: {final_safety_decision.action}")
        print(f"Reason: {final_safety_decision.reason}")

        if not final_safety_decision.is_safe:
            final_answer = (
                "The retrieved documents do not provide enough reliable evidence "
                "to answer this question."
            )
            print_generated_answer(final_answer)

        # ---------------------------------------------------------
        # 10. Add revision and final safety metadata
        # ---------------------------------------------------------
        if isinstance(baseline_result, dict):
            baseline_result["draft_answer"] = draft_answer
            baseline_result["candidate_final_answer"] = candidate_final_answer
            baseline_result["final_answer"] = final_answer
            baseline_result["final_safety_gate"] = {
                "is_safe": final_safety_decision.is_safe,
                "reason": final_safety_decision.reason,
                "action": final_safety_decision.action,
            }
            baseline_result["is_refused"] = not final_safety_decision.is_safe

            baseline_result["revision_decision"] = {
                "decision": revision_decision.decision,
                "reason": revision_decision.reason,
                "instruction": revision_decision.instruction,
                "answer_focus": revision_decision.answer_focus,
                "should_reverify": revision_decision.should_reverify,
            }

            baseline_result["final_verification_summary"] = {
                "total_claims": final_score_summary.get("total_claims", 0),
                "supported_claims": final_score_summary.get("supported_claims", 0),
                "contradicted_claims": final_score_summary.get(
                    "contradicted_claims", 0
                ),
                "not_enough_evidence_claims": final_score_summary.get(
                    "not_enough_evidence_claims", 0
                ),
                "faithfulness_score": final_score_summary.get(
                    "faithfulness_score", 0.0
                ),
            }

        # ---------------------------------------------------------
        # 11. Save final result
        # ---------------------------------------------------------
        saved_file = save_result_to_json(
            output_dir=str(OUTPUT_DIR),
            pdf_name=display_name,
            query=query,
            answer=final_answer,
            retrieved_chunks=results,
            claims=final_claims,
            verification_results=final_verification_results,
            score_summary=final_score_summary,
            baseline_result=baseline_result,
            generator_metadata={
                "mode": get_generator_mode(),
                "model_name": generator_model_name,
            },
        )

        print("\nResult saved successfully.")
        print(f"Saved file: {saved_file}")


if __name__ == "__main__":
    main()

from pathlib import Path

from src.document_loader import load_pdf_text
from src.chunker import chunk_text
from src.retriever import FaissRetriever
from src.generator import AnswerGenerator
from src.claim_extractor import extract_claims
from src.claim_verifier import ClaimVerifier
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


def choose_pdf_file(pdf_files):
    """
    Allow the user to select a PDF file from terminal.
    """

    print_header("Available PDF Files")

    for index, pdf_file in enumerate(pdf_files, start=1):
        print(f"{index}. {pdf_file.name}")

    while True:
        choice = input("\nSelect a PDF by number: ")

        if choice.isdigit():
            selected_index = int(choice)

            if 1 <= selected_index <= len(pdf_files):
                return pdf_files[selected_index - 1]

        print("Invalid selection. Please enter a valid number.")


def main():
    print_header("Claim Grounded Agentic RAG Prototype")

    pdf_files = list_pdf_files(DATA_DIR)
    selected_pdf = choose_pdf_file(pdf_files)

    print("\nProcessing selected PDF. Please wait...")

    extracted_text = load_pdf_text(str(selected_pdf))

    chunks = chunk_text(
        extracted_text,
        chunk_size=700,
        overlap=120
    )

    print_document_status(
        pdf_name=selected_pdf.name,
        total_chars=len(extracted_text),
        total_chunks=len(chunks)
    )

    print("\nLoading retrieval model and building FAISS index...")
    retriever = FaissRetriever()
    retriever.build_index(chunks)

    print("\nLoading lightweight open source LLM...")
    answer_generator = AnswerGenerator()

    print("\nCreating claim verifier...")
    claim_verifier = ClaimVerifier(embedding_model=retriever.model)

    print_header("System Ready")
    print("You can now ask questions about the selected document.")
    print("Type 'exit' to stop the prototype.")

    while True:
        query = input("\nAsk a question: ")

        if query.lower().strip() == "exit":
            print("\nExiting prototype.")
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        results = retriever.retrieve(query, top_k=3)

        print_evidence_summary(results)

        answer = answer_generator.generate_answer(
            query=query,
            retrieved_chunks=results
        )

        baseline_result = create_baseline_result(
            query=query,
            answer=answer,
            retrieved_chunks=results
        )

        print_baseline_result(baseline_result)

        print_generated_answer(answer)

        claims = extract_claims(answer)

        verification_results = claim_verifier.verify_claims(
            claims=claims,
            retrieved_chunks=results
        )

        print_claim_table(verification_results)

        score_summary = calculate_faithfulness_score(verification_results)

        print_score_summary(score_summary)

        saved_file = save_result_to_json(
            output_dir=str(OUTPUT_DIR),
            pdf_name=selected_pdf.name,
            query=query,
            answer=answer,
            retrieved_chunks=results,
            claims=claims,
            verification_results=verification_results,
            score_summary=score_summary,
            baseline_result=baseline_result
        )

        print("\nResult saved successfully.")
        print(f"Saved file: {saved_file}")


if __name__ == "__main__":
    main()
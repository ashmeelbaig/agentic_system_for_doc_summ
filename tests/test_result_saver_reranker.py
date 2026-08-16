import json

from src.result_saver import save_result_to_json


def test_save_result_to_json_includes_rerank_score(tmp_path):
    retrieved_chunks = [
        {
            "chunk_id": "doc_p2_c0",
            "source": "doc.pdf",
            "page_number": 2,
            "text": "This chunk explains claim verification using evidence.",
            "score": 0.72,
            "rerank_score": 0.95,
        }
    ]

    saved_file = save_result_to_json(
        output_dir=str(tmp_path),
        pdf_name="doc.pdf",
        query="How does claim verification work?",
        answer="The system verifies claims using retrieved evidence.",
        retrieved_chunks=retrieved_chunks,
        claims=["The system verifies claims using retrieved evidence."],
        verification_results=[],
        score_summary={
            "total_claims": 1,
            "supported_claims": 1,
            "partially_supported_claims": 0,
            "unsupported_claims": 0,
            "faithfulness_score": 1.0,
        },
        baseline_result=None,
    )

    with open(saved_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    evidence = data["claim_grounded_rag"]["retrieved_evidence"][0]

    assert evidence["similarity_score"] == 0.72
    assert evidence["rerank_score"] == 0.95
    assert evidence["chunk_id"] == "doc_p2_c0"
    assert evidence["source"] == "doc.pdf"
    assert evidence["page_number"] == 2
import json

from src.result_saver import save_result_to_json


def test_save_result_to_json_supports_metadata_evidence(tmp_path):
    retrieved_chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "This is metadata-aware retrieved evidence.",
            "score": 0.91,
        }
    ]

    saved_file = save_result_to_json(
        output_dir=str(tmp_path),
        pdf_name="sample.pdf",
        query="How does retrieval work?",
        answer="The system retrieves evidence using FAISS.",
        retrieved_chunks=retrieved_chunks,
        claims=["The system retrieves evidence using FAISS."],
        verification_results=[
            {
                "claim": "The system retrieves evidence using FAISS.",
                "label": "Supported",
                "score": 0.88,
                "evidence": "This is metadata-aware retrieved evidence.",
                "chunk_id": "sample_p1_c0",
                "source": "sample.pdf",
                "page_number": 1,
            }
        ],
        score_summary={
            "total_claims": 1,
            "supported_claims": 1,
            "partially_supported_claims": 0,
            "unsupported_claims": 0,
            "faithfulness_score": 1.0,
        },
        baseline_result={
            "system_type": "Standard RAG Baseline",
            "retrieved_evidence_count": 1,
        },
    )

    with open(saved_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    evidence = data["claim_grounded_rag"]["retrieved_evidence"][0]

    assert evidence["chunk_id"] == "sample_p1_c0"
    assert evidence["source"] == "sample.pdf"
    assert evidence["page_number"] == 1
    assert evidence["similarity_score"] == 0.91
    assert evidence["text"] == "This is metadata-aware retrieved evidence."


def test_save_result_to_json_still_supports_old_tuple_format(tmp_path):
    retrieved_chunks = [
        (0, "This is old tuple-based evidence.", 0.82)
    ]

    saved_file = save_result_to_json(
        output_dir=str(tmp_path),
        pdf_name="sample.pdf",
        query="What is the evidence?",
        answer="The evidence is retrieved from the document.",
        retrieved_chunks=retrieved_chunks,
        claims=["The evidence is retrieved from the document."],
        verification_results=[],
        score_summary={
            "total_claims": 1,
            "supported_claims": 0,
            "partially_supported_claims": 0,
            "unsupported_claims": 1,
            "faithfulness_score": 0.0,
        },
        baseline_result=None,
    )

    with open(saved_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    evidence = data["claim_grounded_rag"]["retrieved_evidence"][0]

    assert evidence["chunk_index"] == 0
    assert evidence["chunk_id"] is None
    assert evidence["source"] is None
    assert evidence["page_number"] is None
    assert evidence["similarity_score"] == 0.82
    assert evidence["text"] == "This is old tuple-based evidence."
from src.display import print_claim_table


def test_print_claim_table_supports_nli_score(capsys):
    verification_results = [
        {
            "claim": "The system uses FAISS.",
            "label": "Supported",
            "nli_label": "ENTAILMENT",
            "nli_score": 0.94,
            "evidence": "The system uses FAISS for retrieval.",
            "chunk_id": "doc_p1_c0",
            "source": "doc.pdf",
            "page_number": 1,
        }
    ]

    print_claim_table(verification_results)

    captured = capsys.readouterr()

    assert "Claim Verification" in captured.out
    assert "The system uses FAISS." in captured.out
    assert "Supported" in captured.out
    assert "NLI label: ENTAILMENT" in captured.out
    assert "NLI score: 0.9400" in captured.out
    assert "doc_p1_c0" in captured.out
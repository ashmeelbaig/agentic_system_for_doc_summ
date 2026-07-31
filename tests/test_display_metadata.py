from src.display import print_evidence_summary, print_claim_table


def test_print_evidence_summary_supports_metadata_chunks(capsys):
    results = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "This is retrieved evidence from page one.",
            "score": 0.91,
        }
    ]

    print_evidence_summary(results)

    captured = capsys.readouterr()

    assert "Retrieved Evidence" in captured.out
    assert "sample_p1_c0" in captured.out
    assert "sample.pdf" in captured.out
    assert "Page number: 1" in captured.out
    assert "0.9100" in captured.out
    assert "This is retrieved evidence from page one." in captured.out


def test_print_evidence_summary_still_supports_old_tuple_format(capsys):
    results = [
        (0, "This is old tuple based evidence.", 0.82)
    ]

    print_evidence_summary(results)

    captured = capsys.readouterr()

    assert "Chunk index: 0" in captured.out
    assert "0.8200" in captured.out
    assert "This is old tuple based evidence." in captured.out


def test_print_claim_table_supports_metadata_references(capsys):
    verification_results = [
        {
            "claim": "The system uses FAISS.",
            "label": "Supported",
            "score": 0.88,
            "evidence": "The system uses FAISS vector search.",
            "chunk_index": None,
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
        }
    ]

    print_claim_table(verification_results)

    captured = capsys.readouterr()

    assert "Claim Verification" in captured.out
    assert "The system uses FAISS." in captured.out
    assert "Supported" in captured.out
    assert "sample_p1_c0" in captured.out
    assert "sample.pdf" in captured.out
    assert "Page number: 1" in captured.out
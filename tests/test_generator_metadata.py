from src.generator import AnswerGenerator


def test_build_context_supports_metadata_evidence():
    generator = AnswerGenerator.__new__(AnswerGenerator)

    retrieved_chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "This is evidence from page one.",
            "score": 0.91,
        },
        {
            "chunk_id": "sample_p2_c0",
            "source": "sample.pdf",
            "page_number": 2,
            "text": "This is evidence from page two.",
            "score": 0.82,
        },
    ]

    context = generator._build_context(
        retrieved_chunks=retrieved_chunks,
        max_context_words=100,
    )

    assert "sample_p1_c0" in context
    assert "Page 1" in context
    assert "This is evidence from page one." in context
    assert "sample_p2_c0" in context
    assert "Page 2" in context


def test_build_context_still_supports_old_tuple_format():
    generator = AnswerGenerator.__new__(AnswerGenerator)

    retrieved_chunks = [
        (0, "This is the first old chunk.", 0.91),
        (1, "This is the second old chunk.", 0.82),
    ]

    context = generator._build_context(
        retrieved_chunks=retrieved_chunks,
        max_context_words=100,
    )

    assert "[Chunk 0]" in context
    assert "This is the first old chunk." in context
    assert "[Chunk 1]" in context


def test_extractive_answer_supports_metadata_evidence():
    generator = AnswerGenerator.__new__(AnswerGenerator)

    retrieved_chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "The system uses FAISS vector search to retrieve relevant technical evidence from the document.",
            "score": 0.95,
        }
    ]

    answer = generator._generate_extractive_answer(
        query="How does the system retrieve evidence?",
        retrieved_chunks=retrieved_chunks,
    )

    assert "FAISS vector search" in answer
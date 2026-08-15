from src.generator import AnswerGenerator


def test_build_context_includes_source_page_and_scores():
    generator = AnswerGenerator.__new__(AnswerGenerator)

    retrieved_chunks = [
        {
            "chunk_id": "manual_p3_c1",
            "source": "manual.pdf",
            "page_number": 3,
            "text": "The system verifies generated claims against retrieved evidence.",
            "score": 0.81,
            "rerank_score": 0.94,
        }
    ]

    context = generator._build_context(
        retrieved_chunks=retrieved_chunks,
        max_context_words=100,
    )

    assert "manual_p3_c1" in context
    assert "manual.pdf" in context
    assert "Page 3" in context
    assert "The system verifies generated claims" in context


def test_build_prompt_contains_grounding_instruction():
    generator = AnswerGenerator.__new__(AnswerGenerator)

    context = "The system verifies generated claims against retrieved evidence."
    query = "How does the system verify answers?"

    prompt = generator._build_prompt(
        query=query,
        context=context,
    )

    assert "Use only the provided document context" in prompt
    assert "If the context is insufficient" in prompt
    assert query in prompt
    assert context in prompt
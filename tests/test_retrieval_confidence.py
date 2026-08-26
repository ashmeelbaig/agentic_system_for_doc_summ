from src.retrieval_confidence import assess_retrieval_confidence


def test_strong_scores_and_relevant_evidence_allow_answering():
    chunks = [
        {"text": "Solar panels convert sunlight into electrical energy.", "rerank_score": 2.2},
        {"text": "Sunlight is captured by photovoltaic cells.", "rerank_score": 1.8},
    ]

    confidence = assess_retrieval_confidence(
        "How do solar panels convert sunlight into energy?", chunks
    )

    assert confidence.label == "high"
    assert confidence.should_answer is True
    assert confidence.should_retry is False


def test_empty_chunks_have_low_confidence_and_request_retry():
    confidence = assess_retrieval_confidence("Any question", [])

    assert confidence.label == "low"
    assert confidence.should_answer is False
    assert confidence.should_retry is True


def test_weak_scores_and_unrelated_evidence_do_not_allow_answering():
    chunks = [{"text": "Penguins live in cold regions.", "rerank_score": 0.2}]

    confidence = assess_retrieval_confidence(
        "How does photosynthesis produce glucose?", chunks
    )

    assert confidence.label == "low"
    assert confidence.should_answer is False
    assert confidence.should_retry is True

import pytest

import main
from src import pipeline


class FakeRetriever:
    instances = []

    def __init__(self):
        self.retrieve_calls = 0
        self.chunks = None
        self.__class__.instances.append(self)

    def build_index(self, chunks):
        self.chunks = chunks

    def retrieve_evidence(self, query, top_k):
        self.retrieve_calls += 1
        return [
            {
                "chunk_id": "doc_p1_c0",
                "source": "doc.pdf",
                "page_number": 1,
                "text": "Retrieval augmented generation uses retrieved evidence to ground generated answers.",
                "score": 0.9,
            }
        ]


class FakeReranker:
    instances = []

    def __init__(self):
        self.calls = 0
        self.__class__.instances.append(self)

    def rerank(self, query, retrieved_chunks, top_k):
        self.calls += 1
        item = dict(retrieved_chunks[0])
        item["rerank_score"] = 2.0
        return [item]


def test_answer_question_is_importable():
    assert callable(pipeline.answer_question)
    assert callable(main.main)


def test_answer_question_requires_uploaded_pdf_paths(monkeypatch):
    monkeypatch.setattr(pipeline, "load_dotenv", lambda: None)

    with pytest.raises(pipeline.PipelineConfigurationError) as error:
        pipeline.answer_question("Question?", selected_pdf_paths=[])

    assert str(error.value) == "Please upload at least one PDF file."


def test_missing_token_uses_controlled_message(monkeypatch, tmp_path):
    pdf = tmp_path / "uploaded.pdf"
    pdf.write_bytes(b"mock")
    monkeypatch.setattr(pipeline, "load_dotenv", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(pipeline.PipelineConfigurationError) as error:
        pipeline.answer_question("Question?", [str(pdf)])

    assert str(error.value) == (
        "Hugging Face token is missing. Please add HF_TOKEN to your .env file."
    )


def test_answer_question_runs_shared_retrieval_and_returns_saved_result(
    monkeypatch, tmp_path
):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"mock")
    FakeRetriever.instances.clear()
    FakeReranker.instances.clear()
    monkeypatch.setenv("HF_TOKEN", "unit-test-secret")
    monkeypatch.setattr(pipeline, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "prepare_metadata_chunks_from_pdfs",
        lambda paths, **kwargs: {"chunks": [{"text": "Document chunk."}]},
    )
    monkeypatch.setattr(pipeline, "FaissRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline, "EvidenceReranker", FakeReranker)
    monkeypatch.setattr(pipeline, "NLIClaimVerifier", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "run_models_for_evidence",
        lambda *args, **kwargs: {
            "model_results": {"model/a": {"status": "success"}},
            "model_comparison": [{"model_name": "model/a", "status": "success"}],
        },
    )
    saved = tmp_path / "result.json"
    monkeypatch.setattr(
        pipeline, "save_multi_model_result_to_json", lambda **kwargs: saved
    )

    result = pipeline.answer_question(
        "What is retrieval augmented generation?",
        selected_pdf_paths=[str(pdf)],
        generator_mode="multi_hf",
    )

    assert FakeRetriever.instances[0].retrieve_calls == 1
    assert FakeReranker.instances[0].calls == 1
    assert result["model_results"]["model/a"]["status"] == "success"
    assert result["saved_output_path"] == str(saved)
    assert "unit-test-secret" not in str(result)


def test_answer_question_passes_uploaded_paths_and_can_skip_saving(monkeypatch, tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    received = []
    monkeypatch.setenv("HF_TOKEN", "unit-test-secret")
    monkeypatch.setattr(pipeline, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "prepare_metadata_chunks_from_pdfs",
        lambda paths, **kwargs: received.extend(paths) or {"chunks": [{"text": "x"}]},
    )
    monkeypatch.setattr(
        pipeline,
        "check_user_query_safety",
        lambda query: {"is_safe": False, "reason": "test refusal"},
    )
    monkeypatch.setattr(
        pipeline,
        "save_multi_model_result_to_json",
        lambda **kwargs: pytest.fail("save should not run"),
    )

    result = pipeline.answer_question(
        "Question?", [str(first), str(second)], save_output=False
    )

    assert received == [first, second]
    assert result["pdf_name"] == "Multiple PDFs"
    assert result["saved_output_path"] is None

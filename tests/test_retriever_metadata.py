from src.retriever import extract_chunk_texts
import numpy as np

class DummyEmbeddingModel:
    def __init__(self):
        self.last_inputs = None

    def encode(self, inputs, convert_to_numpy=True, show_progress_bar=False):
        self.last_inputs = inputs

        vectors = []

        for text in inputs:
            text = text.lower()

            if "first" in text or "query" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return np.array(vectors, dtype="float32")


def test_extract_chunk_texts_from_string_chunks():
    chunks = [
        "This is the first chunk.",
        "This is the second chunk.",
    ]

    texts = extract_chunk_texts(chunks)

    assert texts == chunks


def test_extract_chunk_texts_from_metadata_chunks():
    chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "This is the first metadata chunk.",
        },
        {
            "chunk_id": "sample_p2_c0",
            "source": "sample.pdf",
            "page_number": 2,
            "text": "This is the second metadata chunk.",
        },
    ]

    texts = extract_chunk_texts(chunks)

    assert texts == [
        "This is the first metadata chunk.",
        "This is the second metadata chunk.",
    ]


from src.retriever import format_retrieved_chunks


def test_format_retrieved_chunks_preserves_metadata():
    chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "This is the first chunk.",
        },
        {
            "chunk_id": "sample_p2_c0",
            "source": "sample.pdf",
            "page_number": 2,
            "text": "This is the second chunk.",
        },
    ]

    indices = [1, 0]
    scores = [0.91, 0.82]

    retrieved = format_retrieved_chunks(
        chunks=chunks,
        indices=indices,
        scores=scores,
    )

    assert isinstance(retrieved, list)
    assert len(retrieved) == 2

    assert retrieved[0]["chunk_id"] == "sample_p2_c0"
    assert retrieved[0]["source"] == "sample.pdf"
    assert retrieved[0]["page_number"] == 2
    assert retrieved[0]["text"] == "This is the second chunk."
    assert retrieved[0]["score"] == 0.91

    assert retrieved[1]["chunk_id"] == "sample_p1_c0"
    assert retrieved[1]["score"] == 0.82


import numpy as np

from src.retriever import FaissRetriever

def test_build_index_uses_text_from_metadata_chunks():
    dummy_model = DummyEmbeddingModel()

    retriever = FaissRetriever.__new__(FaissRetriever)
    retriever.model_name = "dummy-model"
    retriever.model = dummy_model
    retriever.index = None
    retriever.chunks = []

    chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "First metadata chunk text.",
        },
        {
            "chunk_id": "sample_p2_c0",
            "source": "sample.pdf",
            "page_number": 2,
            "text": "Second metadata chunk text.",
        },
    ]

    retriever.build_index(chunks)

    assert dummy_model.last_inputs == [
        "First metadata chunk text.",
        "Second metadata chunk text.",
    ]

    assert retriever.chunks == chunks
    assert retriever.index.ntotal == 2

def test_retrieve_evidence_returns_metadata_chunks():
    dummy_model = DummyEmbeddingModel()

    retriever = FaissRetriever.__new__(FaissRetriever)
    retriever.model_name = "dummy-model"
    retriever.model = dummy_model
    retriever.index = None
    retriever.chunks = []

    chunks = [
        {
            "chunk_id": "sample_p1_c0",
            "source": "sample.pdf",
            "page_number": 1,
            "text": "First metadata chunk text.",
        },
        {
            "chunk_id": "sample_p2_c0",
            "source": "sample.pdf",
            "page_number": 2,
            "text": "Second metadata chunk text.",
        },
    ]

    retriever.build_index(chunks)

    results = retriever.retrieve_evidence(
        query="first chunk query",
        top_k=1,
    )

    assert isinstance(results, list)
    assert len(results) == 1

    first_result = results[0]

    assert first_result["chunk_id"] == "sample_p1_c0"
    assert first_result["source"] == "sample.pdf"
    assert first_result["page_number"] == 1
    assert first_result["text"] == "First metadata chunk text."
    assert isinstance(first_result["score"], float)
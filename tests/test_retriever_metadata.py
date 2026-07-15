from src.retriever import extract_chunk_texts


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
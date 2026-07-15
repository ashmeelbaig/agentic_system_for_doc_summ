from src.chunker import chunk_pages_with_metadata


def test_chunk_pages_with_metadata_returns_required_fields():
    pages = [
        {
            "page_number": 1,
            "text": "This is page one of the technical document. " * 120,
        },
        {
            "page_number": 2,
            "text": "This is page two of the technical document. " * 120,
        },
    ]

    chunks = chunk_pages_with_metadata(
        pages=pages,
        source="sample_document.pdf",
        chunk_size=50,
        overlap=10,
    )

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    first_chunk = chunks[0]

    assert "chunk_id" in first_chunk
    assert "source" in first_chunk
    assert "page_number" in first_chunk
    assert "text" in first_chunk

    assert first_chunk["source"] == "sample_document.pdf"
    assert isinstance(first_chunk["page_number"], int)
    assert isinstance(first_chunk["text"], str)
    assert len(first_chunk["text"]) > 0


def test_chunk_pages_with_metadata_preserves_page_numbers():
    pages = [
        {
            "page_number": 1,
            "text": "Content from first page. " * 100,
        },
        {
            "page_number": 2,
            "text": "Content from second page. " * 100,
        },
    ]

    chunks = chunk_pages_with_metadata(
        pages=pages,
        source="sample_document.pdf",
        chunk_size=40,
        overlap=10,
    )

    page_numbers = {chunk["page_number"] for chunk in chunks}

    assert 1 in page_numbers
    assert 2 in page_numbers
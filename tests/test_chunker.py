from src.chunker import chunk_text


def test_chunk_text_returns_chunks():
    text = "This is a sample technical document. " * 300

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
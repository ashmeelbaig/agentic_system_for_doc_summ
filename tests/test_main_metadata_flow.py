import fitz

from main import prepare_metadata_chunks_from_pdf


def test_prepare_metadata_chunks_from_pdf_creates_metadata_chunks(tmp_path):
    pdf_path = tmp_path / "sample_main_test.pdf"

    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "This is page one of a technical document about retrieval and claim verification. " * 20
    )

    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "This is page two of a technical document about metadata and evidence grounding. " * 20
    )

    doc.save(pdf_path)
    doc.close()

    pages, chunks, total_chars = prepare_metadata_chunks_from_pdf(
        pdf_path=pdf_path,
        chunk_size=40,
        overlap=10
    )

    assert len(pages) == 2
    assert total_chars > 0
    assert len(chunks) > 0

    first_chunk = chunks[0]

    assert first_chunk["source"] == "sample_main_test.pdf"
    assert "chunk_id" in first_chunk
    assert "page_number" in first_chunk
    assert "text" in first_chunk
    assert isinstance(first_chunk["text"], str)
import fitz

from src.document_collection import prepare_metadata_chunks_from_pdfs


def create_test_pdf(pdf_path, text):
    doc = fitz.open()

    page = doc.new_page()
    page.insert_text((72, 72), text)

    doc.save(pdf_path)
    doc.close()


def test_prepare_metadata_chunks_from_pdfs_combines_multiple_documents(tmp_path):
    pdf_one = tmp_path / "document_one.pdf"
    pdf_two = tmp_path / "document_two.pdf"

    create_test_pdf(
        pdf_one,
        "This is the first technical PDF about retrieval and FAISS. " * 30
    )

    create_test_pdf(
        pdf_two,
        "This is the second technical PDF about claim verification and evidence grounding. " * 30
    )

    result = prepare_metadata_chunks_from_pdfs(
        pdf_paths=[pdf_one, pdf_two],
        chunk_size=40,
        overlap=10,
    )

    assert result["document_count"] == 2
    assert result["total_chars"] > 0
    assert len(result["chunks"]) > 0

    sources = {chunk["source"] for chunk in result["chunks"]}

    assert "document_one.pdf" in sources
    assert "document_two.pdf" in sources

    for chunk in result["chunks"]:
        assert "chunk_id" in chunk
        assert "source" in chunk
        assert "page_number" in chunk
        assert "text" in chunk


def test_prepare_metadata_chunks_from_pdfs_rejects_empty_list():
    try:
        prepare_metadata_chunks_from_pdfs(pdf_paths=[])
    except ValueError as error:
        assert "No PDF paths provided" in str(error)
    else:
        raise AssertionError("Expected ValueError for empty PDF list.")
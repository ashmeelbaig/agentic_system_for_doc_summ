import fitz

from main import prepare_metadata_chunks_from_selected_pdfs


def create_test_pdf(pdf_path, text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(pdf_path)
    doc.close()


def test_prepare_metadata_chunks_from_selected_pdfs_single_pdf(tmp_path):
    pdf_one = tmp_path / "single_document.pdf"

    create_test_pdf(
        pdf_one,
        "This PDF is about retrieval and claim verification. " * 30
    )

    result = prepare_metadata_chunks_from_selected_pdfs(
        selected_pdfs=[pdf_one],
        chunk_size=40,
        overlap=10,
    )

    assert result["document_count"] == 1
    assert result["display_name"] == "single_document.pdf"
    assert result["total_chars"] > 0
    assert len(result["chunks"]) > 0
    assert result["chunks"][0]["source"] == "single_document.pdf"


def test_prepare_metadata_chunks_from_selected_pdfs_multiple_pdfs(tmp_path):
    pdf_one = tmp_path / "document_one.pdf"
    pdf_two = tmp_path / "document_two.pdf"

    create_test_pdf(
        pdf_one,
        "This is the first PDF about FAISS retrieval. " * 30
    )

    create_test_pdf(
        pdf_two,
        "This is the second PDF about claim verification. " * 30
    )

    result = prepare_metadata_chunks_from_selected_pdfs(
        selected_pdfs=[pdf_one, pdf_two],
        chunk_size=40,
        overlap=10,
    )

    assert result["document_count"] == 2
    assert result["display_name"] == "Multiple PDFs"
    assert result["total_chars"] > 0
    assert len(result["chunks"]) > 0

    sources = {chunk["source"] for chunk in result["chunks"]}

    assert "document_one.pdf" in sources
    assert "document_two.pdf" in sources
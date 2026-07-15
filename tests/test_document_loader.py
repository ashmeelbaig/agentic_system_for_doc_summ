import fitz

from src.document_loader import load_pdf_pages


def test_load_pdf_pages_returns_page_metadata(tmp_path):
    pdf_path = tmp_path / "sample_test.pdf"

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "This is the first test page.")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "This is the second test page.")

    doc.save(pdf_path)
    doc.close()

    pages = load_pdf_pages(str(pdf_path))

    assert isinstance(pages, list)
    assert len(pages) == 2

    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2

    assert "text" in pages[0]
    assert "first test page" in pages[0]["text"]

    assert "text" in pages[1]
    assert "second test page" in pages[1]["text"]
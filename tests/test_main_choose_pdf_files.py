from pathlib import Path

from main import choose_pdf_files


def test_choose_pdf_files_returns_all_pdfs(monkeypatch):
    pdf_files = [
        Path("data/document_one.pdf"),
        Path("data/document_two.pdf"),
    ]

    monkeypatch.setattr("builtins.input", lambda _: "0")

    selected = choose_pdf_files(pdf_files)

    assert selected == pdf_files


def test_choose_pdf_files_returns_single_pdf(monkeypatch):
    pdf_files = [
        Path("data/document_one.pdf"),
        Path("data/document_two.pdf"),
    ]

    monkeypatch.setattr("builtins.input", lambda _: "2")

    selected = choose_pdf_files(pdf_files)

    assert selected == [Path("data/document_two.pdf")]
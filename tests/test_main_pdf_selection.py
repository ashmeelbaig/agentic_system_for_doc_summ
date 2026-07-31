from pathlib import Path

import pytest

from main import get_selected_pdf_paths


def test_get_selected_pdf_paths_returns_all_pdfs_for_zero_choice():
    pdf_files = [
        Path("data/document_one.pdf"),
        Path("data/document_two.pdf"),
    ]

    selected = get_selected_pdf_paths(pdf_files, choice="0")

    assert selected == pdf_files


def test_get_selected_pdf_paths_returns_single_pdf_for_valid_choice():
    pdf_files = [
        Path("data/document_one.pdf"),
        Path("data/document_two.pdf"),
    ]

    selected = get_selected_pdf_paths(pdf_files, choice="2")

    assert selected == [Path("data/document_two.pdf")]


def test_get_selected_pdf_paths_rejects_invalid_choice():
    pdf_files = [
        Path("data/document_one.pdf"),
        Path("data/document_two.pdf"),
    ]

    with pytest.raises(ValueError):
        get_selected_pdf_paths(pdf_files, choice="5")
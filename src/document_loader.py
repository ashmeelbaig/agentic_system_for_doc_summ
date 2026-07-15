from pathlib import Path
import fitz


def load_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Cleaned extracted text.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported in this prototype.")

    text_parts = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text()

            if page_text.strip():
                text_parts.append(f"\n[Page {page_number}]\n{page_text}")

    full_text = "\n".join(text_parts)

    if not full_text.strip():
        raise ValueError("No readable text was extracted from the PDF.")

    return clean_text(full_text)


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text using simple logic.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
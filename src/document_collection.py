from pathlib import Path
from typing import List, Dict, Any

from src.document_loader import load_pdf_pages
from src.chunker import chunk_pages_with_metadata


def prepare_metadata_chunks_from_pdfs(
    pdf_paths: List[Path],
    chunk_size: int = 700,
    overlap: int = 120
) -> Dict[str, Any]:
    """
    Prepare metadata-aware chunks from multiple PDF files.

    Args:
        pdf_paths: List of PDF file paths.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        Dictionary containing document count, total characters, and combined chunks.
    """

    if not pdf_paths:
        raise ValueError("No PDF paths provided.")

    all_chunks = []
    total_chars = 0

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)

        pages = load_pdf_pages(str(pdf_path))

        document_chars = sum(
            len(page.get("text", ""))
            for page in pages
        )

        total_chars += document_chars

        document_chunks = chunk_pages_with_metadata(
            pages=pages,
            source=pdf_path.name,
            chunk_size=chunk_size,
            overlap=overlap
        )

        all_chunks.extend(document_chunks)

    return {
        "document_count": len(pdf_paths),
        "total_chars": total_chars,
        "chunks": all_chunks
    }
from typing import List


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    """
    Split text into smaller chunks using word based chunking.

    Args:
        text: Full extracted document text.
        chunk_size: Number of words in each chunk.
        overlap: Number of words repeated between chunks.

    Returns:
        List of text chunks.
    """

    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


def print_chunk_summary(chunks: List[str], preview_count: int = 3) -> None:
    """
    Print a short summary of the created chunks.

    Args:
        chunks: List of text chunks.
        preview_count: Number of chunks to preview.
    """

    print(f"\nTotal chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks[:preview_count], start=1):
        print("\n" + "=" * 70)
        print(f"Chunk {index} preview")
        print("=" * 70)
        print(chunk[:600] + "...")
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


def chunk_pages_with_metadata(pages, source, chunk_size=700, overlap=120):
    """
    Split page-level text into word-based chunks and attach metadata.

    Args:
        pages (list): List of page dictionaries.
            Example:
            [
                {"page_number": 1, "text": "page text..."},
                {"page_number": 2, "text": "page text..."}
            ]

        source (str): Source PDF filename.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        list: List of metadata-aware chunk dictionaries.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    metadata_chunks = []

    for page in pages:
        page_number = page.get("page_number")
        text = page.get("text", "")

        if not text or not text.strip():
            continue

        words = text.split()
        start = 0
        chunk_counter = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunk = {
                    "chunk_id": f"{source}_p{page_number}_c{chunk_counter}",
                    "source": source,
                    "page_number": page_number,
                    "text": chunk_text,
                }

                metadata_chunks.append(chunk)

            if end >= len(words):
                break

            start = end - overlap
            chunk_counter += 1

    return metadata_chunks
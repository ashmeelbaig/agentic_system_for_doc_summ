from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    """
    Simple FAISS based retriever using a Hugging Face sentence transformer model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: List[str]) -> None:
        """
        Build FAISS index from document chunks.

        Args:
            chunks: List of text chunks.
        """

        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        self.chunks = chunks

        print("\nLoading embedding model and creating embeddings...")
        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        faiss.normalize_L2(embeddings)

        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings)

        print(f"FAISS index created successfully with {len(chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, str, float]]:
        """
        Retrieve the most relevant chunks for a user query.

        Args:
            query: User question.
            top_k: Number of chunks to retrieve.

        Returns:
            List of tuples containing chunk index, chunk text, and similarity score.
        """

        if self.index is None:
            raise ValueError("FAISS index has not been built yet.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                (
                    int(index),
                    self.chunks[index],
                    float(score)
                )
            )

        return results


def print_retrieved_chunks(results: List[Tuple[int, str, float]]) -> None:
    """
    Print retrieved chunks in terminal.
    """

    print("\nRetrieved Evidence")
    print("=" * 70)

    for rank, (chunk_index, chunk_text, score) in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Chunk index: {chunk_index}")
        print(f"Similarity score: {score:.4f}")
        print("-" * 70)
        print(chunk_text[:1000] + "...")
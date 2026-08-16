from typing import List, Dict, Any


class EvidenceReranker:
    """
    Rerank retrieved evidence chunks using a cross-encoder model.

    FAISS retrieves candidate chunks quickly.
    The reranker scores query-chunk pairs more carefully and selects
    the strongest evidence for answer generation and verification.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model=None,
    ):
        self.model_name = model_name

        if model is not None:
            self.model = model
        else:
            from sentence_transformers import CrossEncoder

            print(f"\nLoading reranker model: {model_name}")
            self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved chunks and return top_k strongest chunks.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not retrieved_chunks:
            return []

        pairs = [
            (query, chunk.get("text", ""))
            for chunk in retrieved_chunks
        ]

        rerank_scores = self.model.predict(pairs)

        reranked_chunks = []

        for original_rank, (chunk, rerank_score) in enumerate(
            zip(retrieved_chunks, rerank_scores),
            start=1,
        ):
            updated_chunk = dict(chunk)
            updated_chunk["original_rank"] = original_rank
            updated_chunk["rerank_score"] = float(rerank_score)

            reranked_chunks.append(updated_chunk)

        reranked_chunks = sorted(
            reranked_chunks,
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked_chunks[:top_k]
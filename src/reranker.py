from typing import List, Dict, Any


class EvidenceReranker:
    """
    Rerank retrieved evidence chunks using a cross-encoder model.

    In production, this can use a Hugging Face cross-encoder model.
    In tests, a dummy model can be injected to keep CI lightweight.
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
        Rerank retrieved evidence chunks for a user query.

        Args:
            query: User question.
            retrieved_chunks: Metadata-aware retrieved evidence chunks.
            top_k: Number of reranked chunks to return.

        Returns:
            List of evidence chunks sorted by rerank_score.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not retrieved_chunks:
            return []

        pairs = []

        for item in retrieved_chunks:
            text = item.get("text", "")
            pairs.append((query, text))

        rerank_scores = self.model.predict(pairs)

        reranked = []

        for original_rank, (item, rerank_score) in enumerate(
            zip(retrieved_chunks, rerank_scores),
            start=1,
        ):
            updated_item = dict(item)
            updated_item["original_rank"] = original_rank
            updated_item["rerank_score"] = float(rerank_score)

            reranked.append(updated_item)

        reranked = sorted(
            reranked,
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]
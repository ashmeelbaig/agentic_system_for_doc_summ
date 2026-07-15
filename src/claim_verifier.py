import re
from typing import List, Tuple, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class ClaimVerifier:
    """
    Verify extracted claims against retrieved evidence using semantic similarity.

    This is a lightweight deterministic verifier for the prototype.
    Later it can be replaced with an NLI model.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        supported_threshold: float = 0.55,
        partial_threshold: float = 0.40
    ):
        self.supported_threshold = supported_threshold
        self.partial_threshold = partial_threshold

        if embedding_model is not None:
            self.model = embedding_model
        else:
            print("\nLoading verifier embedding model...")
            self.model = SentenceTransformer(model_name)

    def verify_claims(
        self,
        claims: List[str],
        retrieved_chunks: List[Tuple[int, str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Verify each claim against evidence sentences from retrieved chunks.

        Args:
            claims: Extracted claims from generated answer.
            retrieved_chunks: List of retrieved chunks from FAISS.

        Returns:
            List of verification results.
        """

        if not claims:
            return []

        evidence_items = self._prepare_evidence_sentences(retrieved_chunks)

        if not evidence_items:
            return [
                {
                    "claim": claim,
                    "label": "Unsupported",
                    "score": 0.0,
                    "evidence": "No suitable evidence sentence found.",
                    "chunk_index": None,
                    "chunk_id": None,
                    "source": None,
                    "page_number": None,
                }
                for claim in claims
            ]

        evidence_texts = [item["sentence"] for item in evidence_items]

        claim_embeddings = self.model.encode(
            claims,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        evidence_embeddings = self.model.encode(
            evidence_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarity_matrix = np.dot(claim_embeddings, evidence_embeddings.T)

        results = []

        for claim_index, claim in enumerate(claims):
            best_evidence_index = int(np.argmax(similarity_matrix[claim_index]))
            best_score = float(similarity_matrix[claim_index][best_evidence_index])
            best_evidence = evidence_items[best_evidence_index]

            label = self._assign_label(best_score)

            results.append(
                {
                    "claim": claim,
                    "label": label,
                    "score": best_score,
                    "evidence": best_evidence["sentence"],
                    "chunk_index": best_evidence["chunk_index"],
                    "chunk_id": best_evidence["chunk_id"],
                    "source": best_evidence["source"],
                    "page_number": best_evidence["page_number"],
                }
            )

        return results

    def _normalize_retrieved_chunk(self, chunk):
        """
        Convert retrieved evidence into a common internal format.

        Supports:
        - old tuple format: (chunk_index, chunk_text, score)
        - new metadata dictionary format
        """

        if isinstance(chunk, dict):
            return {
                "chunk_index": chunk.get("chunk_index"),
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "page_number": chunk.get("page_number"),
                "text": chunk.get("text", ""),
                "score": float(chunk.get("score", 0.0) or 0.0),
            }

        if isinstance(chunk, tuple) and len(chunk) == 3:
            chunk_index, chunk_text, score = chunk

            return {
                "chunk_index": chunk_index,
                "chunk_id": f"chunk_{chunk_index}",
                "source": None,
                "page_number": None,
                "text": chunk_text,
                "score": float(score),
            }

        raise TypeError(
            "Retrieved chunk must be either a metadata dictionary or a tuple of "
            "(chunk_index, chunk_text, score)."
        )
    def _prepare_evidence_sentences(
        self,
        retrieved_chunks: List
    ) -> List[Dict[str, Any]]:
        """
        Split retrieved chunks into evidence sentences.

        Supports both old tuple-based retrieval results and new metadata-aware evidence.
        """

        evidence_items = []

        for chunk in retrieved_chunks:
            normalized_chunk = self._normalize_retrieved_chunk(chunk)

            chunk_index = normalized_chunk["chunk_index"]
            chunk_id = normalized_chunk["chunk_id"]
            source = normalized_chunk["source"]
            page_number = normalized_chunk["page_number"]
            chunk_text = normalized_chunk["text"]
            retrieval_score = normalized_chunk["score"]

            sentences = re.split(r"(?<=[.!?])\s+", chunk_text)

            for sentence in sentences:
                sentence = sentence.strip()

                if not self._is_valid_evidence_sentence(sentence):
                    continue

                evidence_items.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "source": source,
                        "page_number": page_number,
                        "sentence": sentence,
                        "retrieval_score": retrieval_score,
                    }
                )

        return evidence_items

    def _is_valid_evidence_sentence(self, sentence: str) -> bool:
        """
        Keep only useful evidence sentences.
        """

        if not sentence:
            return False

        words = sentence.split()

        if len(words) < 8:
            return False

        if len(words) > 90:
            return False

        alphabetic_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence)

        if len(alphabetic_words) < 6:
            return False

        return True

    def _assign_label(self, score: float) -> str:
        """
        Assign a support label based on similarity score.
        """

        if score >= self.supported_threshold:
            return "Supported"

        if score >= self.partial_threshold:
            return "Partially supported"

        return "Unsupported"


def print_verification_results(results: List[Dict[str, Any]]) -> None:
    """
    Print verification results in terminal.
    """

    print("\nClaim Verification Results")
    print("=" * 70)

    if not results:
        print("No claims were available for verification.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\nClaim {index}")
        print("-" * 70)
        print(f"Claim: {result['claim']}")
        print(f"Label: {result['label']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Evidence chunk: {result['chunk_index']}")
        print(f"Best evidence: {result['evidence'][:500]}")
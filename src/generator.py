from typing import List, Tuple

import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class AnswerGenerator:
    """
    Lightweight answer generator using an open source Hugging Face model.

    The model is used first. If the generated answer is too weak, the system
    falls back to a simple extractive answer from the retrieved evidence.
    """

    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name

        print(f"\nLoading answer generation model: {model_name}")
        print("This may take some time on first run...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        print(f"Answer generation model loaded successfully on: {self.device}")

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[int, str, float]],
        max_context_words: int = 320
    ) -> str:
        """
        Generate an answer using the query and retrieved evidence.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not retrieved_chunks:
            return "No relevant evidence was found in the document."

        context = self._build_context(retrieved_chunks, max_context_words)

        prompt = f"""
Use the document context to answer the question.
Write the answer in two complete sentences.
Do not answer with only a heading or title.
If the context does not contain the answer, say that the document context does not contain enough information.

Context:
{context}

Question:
{query}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )

        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        if self._is_weak_answer(answer):
            print("\nLLM answer was too weak. Using extractive fallback answer.")
            answer = self._generate_extractive_answer(query, retrieved_chunks)

        return answer

    def _normalize_retrieved_chunk(self, chunk):
        """
        Convert retrieved evidence into a common internal format.

        Supports:
        - old tuple format: (chunk_index, chunk_text, score)
        - new metadata dictionary format
        """

        if isinstance(chunk, dict):
            return {
                "chunk_id": chunk.get("chunk_id", "unknown_chunk"),
                "source": chunk.get("source"),
                "page_number": chunk.get("page_number"),
                "text": chunk.get("text", ""),
                "score": float(chunk.get("score", 0.0) or 0.0),
            }

        if isinstance(chunk, tuple) and len(chunk) == 3:
            chunk_index, chunk_text, score = chunk

            return {
                "chunk_id": chunk_index,
                "source": None,
                "page_number": None,
                "text": chunk_text,
                "score": float(score),
            }

        raise TypeError(
            "Retrieved chunk must be either a metadata dictionary or a tuple of "
            "(chunk_index, chunk_text, score)."
        )
    
    def _build_context(
        self,
        retrieved_chunks: List,
        max_context_words: int
    ) -> str:
        """
        Build context from retrieved chunks.

        Supports both old tuple-based retrieval results and new metadata-aware evidence.
        """

        all_text = []

        for chunk in retrieved_chunks:
            normalized_chunk = self._normalize_retrieved_chunk(chunk)

            chunk_id = normalized_chunk["chunk_id"]
            page_number = normalized_chunk["page_number"]
            chunk_text = normalized_chunk["text"]

            if page_number is not None:
                label = f"[Chunk {chunk_id} | Page {page_number}]"
            else:
                label = f"[Chunk {chunk_id}]"

            all_text.append(f"{label} {chunk_text}")

        combined_text = "\n\n".join(all_text)
        words = combined_text.split()

        return " ".join(words[:max_context_words])

    def _is_weak_answer(self, answer: str) -> bool:
        """
        Detect very short or poor answers.
        """

        if not answer or not answer.strip():
            return True

        words = answer.split()

        if len(words) < 8:
            return True

        if answer.count(",") >= 3 and len(words) < 15:
            return True

        if answer.startswith("“") and answer.endswith("”") and len(words) < 15:
            return True

        return False

    def _generate_extractive_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[int, str, float]],
        max_sentences: int = 3
    ) -> str:
        """
        Create a simple answer from the most relevant sentences in retrieved chunks.
        """

        query_words = set(self._clean_words(query))
        candidate_sentences = []

        for chunk in retrieved_chunks:
            normalized_chunk = self._normalize_retrieved_chunk(chunk)

            chunk_id = normalized_chunk["chunk_id"]
            chunk_text = normalized_chunk["text"]
            score = normalized_chunk["score"]

            sentences = re.split(r"(?<=[.!?])\s+", chunk_text)

            for sentence in sentences:
                sentence = sentence.strip()

                if len(sentence.split()) < 8:
                    continue

                sentence_words = set(self._clean_words(sentence))
                overlap = len(query_words.intersection(sentence_words))

                candidate_sentences.append(
                    {
                        "sentence": sentence,
                        "overlap": overlap,
                        "retrieval_score": score,
                        "chunk_id": chunk_id
                    }
                )

        if not candidate_sentences:
            return "The retrieved document evidence is not sufficient to generate a clear answer."

        candidate_sentences = sorted(
            candidate_sentences,
            key=lambda item: (item["overlap"], item["retrieval_score"]),
            reverse=True
        )

        selected_sentences = []
        seen = set()

        for item in candidate_sentences:
            sentence = item["sentence"]

            if sentence.lower() in seen:
                continue

            selected_sentences.append(sentence)
            seen.add(sentence.lower())

            if len(selected_sentences) >= max_sentences:
                break

        return " ".join(selected_sentences)

    def _clean_words(self, text: str) -> List[str]:
        """
        Convert text into simple lowercase words.
        """

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        stopwords = {
            "the", "and", "for", "with", "that", "this", "from",
            "what", "which", "are", "was", "were", "has", "have",
            "main", "document", "about", "into", "using"
        }

        return [word for word in words if word not in stopwords]
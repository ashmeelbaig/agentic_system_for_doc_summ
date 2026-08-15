import re
from typing import List, Dict, Any


class NLIClaimVerifier:
    """
    Verify claims using Natural Language Inference.

    The verifier compares each claim against retrieved evidence and classifies
    the relationship as entailment, contradiction, or neutral.

    Output labels:
    - Supported
    - Contradicted
    - Not enough evidence
    """

    def __init__(
        self,
        model_name: str = "typeform/distilbert-base-uncased-mnli",
        nli_pipeline=None,
    ):
        self.model_name = model_name

        if nli_pipeline is not None:
            self.nli_pipeline = nli_pipeline
        else:
            from transformers import pipeline

            print(f"\nLoading NLI claim verification model: {model_name}")
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
            )

    def verify_claims(
        self,
        claims: List[str],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Verify claims against retrieved evidence using NLI.
        """

        if not claims:
            return []

        evidence_items = self._prepare_evidence_sentences(retrieved_chunks)

        if not evidence_items:
            return [
                {
                    "claim": claim,
                    "label": "Not enough evidence",
                    "nli_label": "NEUTRAL",
                    "nli_score": 0.0,
                    "evidence": "No suitable evidence sentence found.",
                    "chunk_id": None,
                    "source": None,
                    "page_number": None,
                }
                for claim in claims
            ]

        results = []

        for claim in claims:
            best_result = self._find_best_nli_result(
                claim=claim,
                evidence_items=evidence_items,
            )

            results.append(best_result)

        return results

    def _find_best_nli_result(
        self,
        claim: str,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run NLI between one claim and all evidence sentences.
        """

        nli_inputs = []

        for item in evidence_items:
            premise = item["sentence"]
            hypothesis = claim

            nli_inputs.append(
                f"{premise} </s></s> {hypothesis}"
            )

        nli_outputs = self.nli_pipeline(
            nli_inputs,
            truncation=True,
        )

        best_index = 0
        best_priority_score = -1.0

        for index, output in enumerate(nli_outputs):
            raw_label = output.get("label", "")
            score = float(output.get("score", 0.0))
            normalized_label = self._normalize_nli_label(raw_label)

            priority_score = self._priority_score(
                normalized_label=normalized_label,
                score=score,
            )

            if priority_score > best_priority_score:
                best_priority_score = priority_score
                best_index = index

        best_output = nli_outputs[best_index]
        best_evidence = evidence_items[best_index]

        nli_label = self._normalize_nli_label(best_output.get("label", "NEUTRAL"))
        nli_score = float(best_output.get("score", 0.0))
        final_label = self._map_nli_to_claim_label(nli_label)

        return {
            "claim": claim,
            "label": final_label,
            "nli_label": nli_label,
            "nli_score": nli_score,
            "evidence": best_evidence["sentence"],
            "chunk_id": best_evidence.get("chunk_id"),
            "source": best_evidence.get("source"),
            "page_number": best_evidence.get("page_number"),
        }

    def _prepare_evidence_sentences(
        self,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Split retrieved evidence chunks into sentence-level evidence items.
        """

        evidence_items = []

        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "")

            sentences = re.split(r"(?<=[.!?])\s+", chunk_text)

            for sentence in sentences:
                sentence = sentence.strip()

                if not self._is_valid_evidence_sentence(sentence):
                    continue

                evidence_items.append(
                    {
                        "sentence": sentence,
                        "chunk_id": chunk.get("chunk_id"),
                        "source": chunk.get("source"),
                        "page_number": chunk.get("page_number"),
                        "retrieval_score": chunk.get("score"),
                        "rerank_score": chunk.get("rerank_score"),
                    }
                )

        return evidence_items

    def _is_valid_evidence_sentence(self, sentence: str) -> bool:
        """
        Keep useful evidence sentences only.
        """

        if not sentence:
            return False

        words = sentence.split()

        if len(words) < 6:
            return False

        if len(words) > 120:
            return False

        return True

    def _normalize_nli_label(self, label: str) -> str:
        """
        Normalize different model label formats.
        """

        label = label.upper()

        if "ENTAIL" in label:
            return "ENTAILMENT"

        if "CONTRADICT" in label:
            return "CONTRADICTION"

        if "NEUTRAL" in label:
            return "NEUTRAL"

        return "NEUTRAL"

    def _map_nli_to_claim_label(self, nli_label: str) -> str:
        """
        Map NLI labels to project-level claim labels.
        """

        if nli_label == "ENTAILMENT":
            return "Supported"

        if nli_label == "CONTRADICTION":
            return "Contradicted"

        return "Not enough evidence"

    def _priority_score(self, normalized_label: str, score: float) -> float:
        """
        Choose the strongest useful evidence.

        Entailment and contradiction are more informative than neutral.
        """

        if normalized_label == "ENTAILMENT":
            return 2.0 + score

        if normalized_label == "CONTRADICTION":
            return 2.0 + score

        return score
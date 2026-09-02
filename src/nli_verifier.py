import re
from typing import List, Dict, Any


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "has", "have", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "these", "this", "to", "was",
    "were", "what", "which", "with",
}

_NEGATION_WORDS = {"no", "not", "never", "none", "without", "neither", "nor"}


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
        entailment_threshold: float = 0.5,
        top_k_evidence: int = 3,
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.top_k_evidence = top_k_evidence

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
                    "nli_original_label": "NEUTRAL",
                    "nli_score": 0.0,
                    "support_override_applied": False,
                    "support_override_reason": "No suitable evidence sentence found.",
                    "claim_evidence_keyword_overlap": 0.0,
                    "matched_key_terms": [],
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
        """Rank evidence lexically, then run NLI on the strongest candidates."""

        ranked_evidence = self._rank_evidence_sentences(claim, evidence_items)
        candidates = ranked_evidence[:self.top_k_evidence]

        nli_inputs = []

        for item in candidates:
            premise = item["sentence"]
            hypothesis = claim

            nli_inputs.append(
                f"{premise} </s></s> {hypothesis}"
            )

        nli_outputs = self.nli_pipeline(
            nli_inputs,
            truncation=True,
        )

        normalized_outputs = [
            {
                "label": self._normalize_nli_label(output.get("label", "")),
                "score": float(output.get("score", 0.0)),
            }
            for output in nli_outputs
        ]
        for output in normalized_outputs:
            if (
                output["label"] == "ENTAILMENT"
                and output["score"] < self.entailment_threshold
            ):
                output["label"] = "NEUTRAL"

        # A supported candidate takes precedence even when another candidate has
        # a higher neutral score. Ranking order breaks ties in favor of the more
        # lexically relevant sentence.
        entailed = [
            index for index, output in enumerate(normalized_outputs)
            if output["label"] == "ENTAILMENT"
            and output["score"] >= self.entailment_threshold
        ]
        if entailed:
            best_index = max(
                entailed,
                key=lambda index: normalized_outputs[index]["score"],
            )
        else:
            best_index = max(
                range(len(normalized_outputs)),
                key=lambda index: self._priority_score(
                    normalized_outputs[index]["label"],
                    normalized_outputs[index]["score"],
                ),
            )

        best_output = normalized_outputs[best_index]
        best_evidence = candidates[best_index]

        nli_label = best_output["label"]
        nli_score = best_output["score"]
        final_label = self._map_nli_to_claim_label(nli_label)
        override = self._support_override(claim, best_evidence["sentence"], nli_label)
        if override["label"] is not None:
            final_label = override["label"]

        return {
            "claim": claim,
            "label": final_label,
            "nli_label": nli_label,
            "nli_original_label": nli_label,
            "nli_score": nli_score,
            "support_override_applied": override["label"] is not None,
            "support_override_reason": override["reason"],
            "claim_evidence_keyword_overlap": override["keyword_overlap"],
            "matched_key_terms": override["matched_key_terms"],
            "evidence": best_evidence["sentence"],
            "chunk_id": best_evidence.get("chunk_id"),
            "source": best_evidence.get("source"),
            "page_number": best_evidence.get("page_number"),
            "candidate_evidence_checked": [item["sentence"] for item in candidates],
            "selected_evidence_rank": best_index + 1,
        }

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())

    def _claim_keywords(self, claim: str) -> List[str]:
        return [
            token for token in self._tokens(claim)
            if token not in _STOP_WORDS and len(token) > 2
        ]

    @staticmethod
    def _claim_list_items(claim: str) -> List[str]:
        """Extract conspicuous enumerated terms without domain-specific values."""
        # Also recognize ordinary comma-separated lists after a colon or phrases
        # such as "including", while avoiding treating normal prose as a list.
        tail_match = re.search(r"(?:including|such as|:)\s+([^.!?]+)", claim, re.I)
        if tail_match and "," in tail_match.group(1):
            parts = re.split(r"\s*,\s*|\s+and\s+", tail_match.group(1), flags=re.I)
            items = [part.strip().lower() for part in parts if part.strip()]
            if len(items) >= 2:
                return items

        uppercase_items = re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b", claim)
        return (
            list(dict.fromkeys(item.lower() for item in uppercase_items))
            if len(uppercase_items) >= 2 else []
        )

    def _support_override(self, claim: str, evidence: str, nli_label: str) -> Dict[str, Any]:
        """Recover strong lexical support missed by NLI, especially enumerations."""
        claim_keywords = set(self._claim_keywords(claim))
        evidence_tokens = set(self._tokens(evidence))
        matched_keywords = claim_keywords & evidence_tokens
        keyword_overlap = len(matched_keywords) / max(len(claim_keywords), 1)

        uppercase_terms = list(dict.fromkeys(
            term.lower() for term in re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b", claim)
        ))
        matched_key_terms = [term for term in uppercase_terms if term in evidence_tokens]
        list_items = self._claim_list_items(claim)
        matched_list_items = [
            item for item in list_items
            if set(self._tokens(item)).issubset(evidence_tokens)
        ]

        result = {
            "label": None,
            "reason": "Override not applicable because the NLI result was not neutral.",
            "keyword_overlap": round(keyword_overlap, 4),
            "matched_key_terms": matched_key_terms,
        }
        if nli_label != "NEUTRAL":
            return result

        claim_negated = bool(set(self._tokens(claim)) & _NEGATION_WORDS)
        evidence_negated = bool(set(self._tokens(evidence)) & _NEGATION_WORDS)
        if claim_negated != evidence_negated:
            result["reason"] = "No override: claim and evidence have different negation polarity."
            return result

        if len(list_items) >= 2:
            list_overlap = len(matched_list_items) / len(list_items)
            if list_overlap == 1.0:
                result["label"] = "Supported"
                result["reason"] = "All claim list items are present in the evidence."
            elif len(matched_list_items) >= 2 and list_overlap >= 0.6:
                result["label"] = "Partially Supported"
                result["reason"] = "Most, but not all, claim list items are present in the evidence."
            else:
                result["reason"] = "No override: too few claim list items match the evidence."
            return result

        key_term_coverage = len(matched_key_terms) / max(len(uppercase_terms), 1)
        if keyword_overlap >= 0.85 and key_term_coverage == 1.0:
            result["label"] = "Supported"
            result["reason"] = "Evidence contains all key terms with very high keyword overlap."
        elif keyword_overlap >= 0.65 and (not uppercase_terms or key_term_coverage >= 0.8):
            result["label"] = "Partially Supported"
            result["reason"] = "Evidence has high keyword overlap and contains the important entities."
        else:
            result["reason"] = "No override: factual overlap is below the support threshold."
        return result

    def _rank_evidence_sentences(
        self,
        claim: str,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        claim_keywords = set(self._claim_keywords(claim))
        list_items = self._claim_list_items(claim)
        claim_text = " ".join(self._tokens(claim))

        def rank_key(item):
            sentence_text = " ".join(self._tokens(item["sentence"]))
            sentence_tokens = set(self._tokens(item["sentence"]))
            overlap_count = len(claim_keywords & sentence_tokens)
            overlap_ratio = overlap_count / max(len(claim_keywords), 1)
            list_count = sum(value in sentence_text for value in list_items)
            list_ratio = list_count / max(len(list_items), 1) if list_items else 0.0
            exact_claim = int(bool(claim_text) and claim_text in sentence_text)
            rerank_score = item.get("rerank_score")
            if rerank_score is None:
                rerank_score = item.get("retrieval_score")
            rerank_score = float(rerank_score or 0.0)
            return (
                list_ratio,
                list_count,
                overlap_ratio,
                overlap_count,
                exact_claim,
                rerank_score,
            )

        return sorted(evidence_items, key=rank_key, reverse=True)

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

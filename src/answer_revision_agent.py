import re
from dataclasses import dataclass
from typing import List, Dict, Any


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
    "and", "or", "how", "does", "do", "these", "this", "that", "with",
    "from", "by", "as", "it", "be", "was", "were", "about", "according",
    "explain", "describe", "discuss", "main", "role", "purpose"
}


@dataclass
class RevisionDecision:
    decision: str          # keep, revise, refuse
    reason: str
    instruction: str
    answer_focus: str      # good, weak
    should_reverify: bool


def _keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def _answer_focus(query: str, answer: str) -> str:
    query_terms = set(_keywords(query))
    answer_terms = set(_keywords(answer))

    if not answer.strip():
        return "weak"

    if not query_terms:
        return "good"

    coverage = len(query_terms.intersection(answer_terms)) / len(query_terms)

    answer_words = answer.split()

    metadata_like = any(
        token in answer.lower()
        for token in ["chunk_id", "page_number", "rerank_score", ".pdf_p"]
    )

    too_short = len(answer_words) < 6
    too_long = len(answer_words) > 180

    if metadata_like or too_short:
        return "weak"

    if coverage < 0.25:
        return "weak"

    if too_long and coverage < 0.50:
        return "weak"

    return "good"


def _count_labels(verification_results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "supported": 0,
        "contradicted": 0,
        "not_enough_evidence": 0,
        "other": 0,
    }

    for item in verification_results:
        label = str(item.get("label", "")).lower()

        if label == "supported":
            counts["supported"] += 1
        elif label == "contradicted":
            counts["contradicted"] += 1
        elif label in ["not enough evidence", "not_enough_evidence"]:
            counts["not_enough_evidence"] += 1
        else:
            counts["other"] += 1

    return counts


def decide_answer_revision(
    query: str,
    answer: str,
    verification_results: List[Dict[str, Any]],
    score_summary: Dict[str, Any],
) -> RevisionDecision:
    """
    Rule-based Answer Revision Agent V1.

    It decides whether the answer is ready for the user.
    It does not call an LLM.
    It only creates instructions for the Answer Agent if revision is needed.
    """

    focus = _answer_focus(query, answer)
    counts = _count_labels(verification_results)

    total_claims = int(score_summary.get("total_claims", len(verification_results)))
    faithfulness = float(score_summary.get("faithfulness_score", 0.0))

    # Case 1: no useful claims were extracted
    if total_claims == 0:
        return RevisionDecision(
            decision="revise",
            reason="No verifiable claims were extracted from the answer.",
            instruction=(
                "Rewrite the answer so it directly answers the question using only "
                "the provided evidence. Avoid metadata, headings, and unrelated text."
            ),
            answer_focus=focus,
            should_reverify=True,
        )

    # Case 2: contradicted claims are risky
    if counts["contradicted"] > 0:
        return RevisionDecision(
            decision="revise",
            reason="At least one generated claim contradicts the retrieved evidence.",
            instruction=(
                "Revise the answer by correcting or removing any contradicted claims. "
                "Only include statements that are supported by the provided evidence."
            ),
            answer_focus=focus,
            should_reverify=True,
        )

    # Case 3: unsupported claims need removal or refusal
    if counts["not_enough_evidence"] > 0:
        return RevisionDecision(
            decision="revise",
            reason="Some claims do not have enough evidence.",
            instruction=(
                "Revise the answer by removing claims that are not supported by the evidence. "
                "If the evidence is not enough to answer the question, state that the documents "
                "do not provide enough evidence."
            ),
            answer_focus=focus,
            should_reverify=True,
        )

    # Case 4: answer is supported but not focused
    if focus == "weak":
        return RevisionDecision(
            decision="revise",
            reason="The answer is supported but not focused enough on the question.",
            instruction=(
                "Rewrite the answer in a shorter and more focused form. "
                "Answer the user question directly and remove unnecessary details."
            ),
            answer_focus=focus,
            should_reverify=True,
        )

    # Case 5: low faithfulness
    if faithfulness < 0.75:
        return RevisionDecision(
            decision="revise",
            reason="The faithfulness score is below the accepted threshold.",
            instruction=(
                "Rewrite the answer using only supported evidence. "
                "Remove weak, unsupported, or unrelated statements."
            ),
            answer_focus=focus,
            should_reverify=True,
        )

    return RevisionDecision(
        decision="keep",
        reason="The answer is supported and focused enough to send to the user.",
        instruction="No revision needed.",
        answer_focus=focus,
        should_reverify=False,
    )


def build_revision_query(original_query: str, instruction: str) -> str:
    """
    Builds a revised query for the existing AnswerGenerator.
    This avoids changing the generator class for now.
    """

    return (
        f"{original_query}\n\n"
        f"Revision instruction: {instruction}\n"
        f"Use only the provided retrieved evidence. "
        f"Do not add information that is not supported by the evidence."
    )


@dataclass
class FinalSafetyDecision:
    is_safe: bool
    reason: str
    action: str   # send or refuse


def final_safety_gate(
    query: str,
    final_answer: str,
    final_verification_results: List[Dict[str, Any]],
    final_score_summary: Dict[str, Any],
) -> FinalSafetyDecision:
    """
    Final safety check after possible answer revision.

    If the final answer is still weak, contradicted, or unsupported,
    it should not be sent to the user.
    """

    focus = _answer_focus(query, final_answer)
    counts = _count_labels(final_verification_results)

    total_claims = int(
        final_score_summary.get("total_claims", len(final_verification_results))
    )
    faithfulness = float(final_score_summary.get("faithfulness_score", 0.0))

    if not final_answer.strip():
        return FinalSafetyDecision(
            is_safe=False,
            reason="Final answer is empty.",
            action="refuse",
        )

    if total_claims == 0:
        return FinalSafetyDecision(
            is_safe=False,
            reason="No verifiable claims were found in the final answer.",
            action="refuse",
        )

    if counts["contradicted"] > 0:
        return FinalSafetyDecision(
            is_safe=False,
            reason="Final answer still contains contradicted claims.",
            action="refuse",
        )

    if counts["not_enough_evidence"] > 0:
        return FinalSafetyDecision(
            is_safe=False,
            reason="Final answer still contains claims with not enough evidence.",
            action="refuse",
        )

    if faithfulness < 0.75:
        return FinalSafetyDecision(
            is_safe=False,
            reason="Final faithfulness score is below the safety threshold.",
            action="refuse",
        )

    if focus == "weak":
        return FinalSafetyDecision(
            is_safe=False,
            reason="Final answer is not focused enough on the user question.",
            action="refuse",
        )

    return FinalSafetyDecision(
        is_safe=True,
        reason="Final answer passed safety checks.",
        action="send",
    )
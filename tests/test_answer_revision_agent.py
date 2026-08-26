import pytest

from src.answer_revision_agent import (
    build_revision_query,
    decide_answer_revision,
    final_safety_gate,
)


QUERY = "How does photosynthesis use sunlight?"
FOCUSED_ANSWER = (
    "Photosynthesis uses sunlight to help plants convert water and carbon dioxide "
    "into stored chemical energy."
)
SUPPORTED = [{"claim": FOCUSED_ANSWER, "label": "supported"}]
GOOD_SCORE = {"total_claims": 1, "faithfulness_score": 1.0}


def test_supported_focused_answer_is_kept():
    decision = decide_answer_revision(QUERY, FOCUSED_ANSWER, SUPPORTED, GOOD_SCORE)

    assert decision.decision == "keep"
    assert decision.answer_focus == "good"
    assert decision.should_reverify is False


@pytest.mark.parametrize("label", ["contradicted", "not enough evidence"])
def test_risky_claim_is_revised(label):
    verification = [{"claim": FOCUSED_ANSWER, "label": label}]

    decision = decide_answer_revision(QUERY, FOCUSED_ANSWER, verification, GOOD_SCORE)

    assert decision.decision == "revise"
    assert decision.should_reverify is True


def test_no_claims_is_revised():
    decision = decide_answer_revision(
        QUERY, FOCUSED_ANSWER, [], {"total_claims": 0, "faithfulness_score": 0.0}
    )

    assert decision.decision == "revise"


def test_weak_focus_answer_is_revised():
    answer = "Volcanoes release molten rock during an eruption event."

    decision = decide_answer_revision(QUERY, answer, SUPPORTED, GOOD_SCORE)

    assert decision.decision == "revise"
    assert decision.answer_focus == "weak"


def test_build_revision_query_contains_query_and_instruction():
    instruction = "Remove unsupported claims."

    revision_query = build_revision_query(QUERY, instruction)

    assert QUERY in revision_query
    assert instruction in revision_query


def test_final_safety_gate_sends_supported_focused_answer():
    decision = final_safety_gate(QUERY, FOCUSED_ANSWER, SUPPORTED, GOOD_SCORE)

    assert decision.is_safe is True
    assert decision.action == "send"


@pytest.mark.parametrize(
    ("verification", "score"),
    [
        ([{"claim": FOCUSED_ANSWER, "label": "contradicted"}], GOOD_SCORE),
        (SUPPORTED, {"total_claims": 1, "faithfulness_score": 0.5}),
        ([], {"total_claims": 0, "faithfulness_score": 1.0}),
    ],
)
def test_final_safety_gate_refuses_unsafe_answer(verification, score):
    decision = final_safety_gate(QUERY, FOCUSED_ANSWER, verification, score)

    assert decision.is_safe is False
    assert decision.action == "refuse"

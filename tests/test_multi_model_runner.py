from dataclasses import dataclass

from src.answer_revision_agent import is_refusal_answer
from src.generators.multi_model_runner import run_all_generators


EVIDENCE = [{"text": "Plants use sunlight to produce chemical energy."}]
REFUSAL = "The retrieved documents do not provide enough reliable evidence to answer this question."


@dataclass
class Decision:
    decision: str
    reason: str = "test"
    instruction: str = "Rewrite directly."
    answer_focus: str = "good"
    should_reverify: bool = False


@dataclass
class Safety:
    is_safe: bool
    reason: str = "test"
    action: str = "send"


class FakeGenerator:
    provider = "huggingface_api"

    def __init__(self, model_id, answers, seen):
        self.model_id = model_id
        self.answers = list(answers)
        self.seen = seen
        self.attempted_methods = ["text_generation"]
        self.attempt_failures = []

    def generate_answer(self, query, evidence_chunks, instruction=None):
        self.seen.append(evidence_chunks)
        value = self.answers.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakeVerifier:
    def __init__(self):
        self.calls = 0

    def verify_claims(self, claims, evidence_chunks):
        self.calls += 1
        return [{"claim": claim, "label": "Supported"} for claim in claims]


def claims(answer):
    return [answer]


def keep(*args):
    return Decision("keep")


def revise_refusal(query, answer, verification, score, confidence):
    return Decision("revise") if is_refusal_answer(answer) else Decision("keep")


def safe(query, answer, verification, score):
    return Safety(True)


def test_all_models_reuse_same_evidence_and_output_is_separate():
    seen = []
    factory = lambda model: FakeGenerator(model, [f"Answer from {model}."], seen)
    result = run_all_generators(
        "Question", EVIDENCE, ["one", "two"], claims, FakeVerifier(), keep, safe,
        {"label": "high"}, factory,
    )

    assert set(result["model_results"]) == {"one", "two"}
    assert all(item is EVIDENCE for item in seen)
    assert len(result["model_comparison"]) == 2
    assert result["model_results"]["one"]["atomic_claims"] == ["Answer from one."]


def test_one_model_failure_does_not_stop_other_models():
    seen = []
    answers = {"bad": [RuntimeError("secret raw error")], "good": ["Grounded answer."]}
    factory = lambda model: FakeGenerator(model, answers[model], seen)
    result = run_all_generators(
        "Question", EVIDENCE, ["bad", "good"], claims, FakeVerifier(), keep, safe,
        {"label": "high"}, factory,
    )

    assert result["model_results"]["bad"]["status"] == "failed"
    assert result["model_results"]["bad"]["error"] == "Model generation failed."
    assert result["model_results"]["bad"]["attempted_methods"] == ["text_generation"]
    assert result["model_results"]["bad"]["attempt_failures"] == []
    assert result["model_results"]["good"]["status"] == "success"
    assert result["model_results"]["good"]["attempted_methods"] == ["text_generation"]


def test_refusal_bypasses_nli_and_revision_is_verified():
    seen = []
    verifier = FakeVerifier()
    factory = lambda model: FakeGenerator(
        model, [REFUSAL, "Plants use sunlight to produce chemical energy."], seen
    )
    result = run_all_generators(
        "Question", EVIDENCE, ["model"], claims, verifier, revise_refusal, safe,
        {"label": "high"}, factory,
    )

    model = result["model_results"]["model"]
    assert verifier.calls == 1
    assert model["revision_decision"]["decision"] == "revise"
    assert model["is_refusal_answer"] is False

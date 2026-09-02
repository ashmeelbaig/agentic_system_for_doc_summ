from src.nli_verifier import NLIClaimVerifier


class DummyNLIPipeline:
    def __call__(self, inputs, truncation=True):
        results = []

        for item in inputs:
            text = item.lower()

            if "does not use faiss" in text:
                results.append({"label": "CONTRADICTION", "score": 0.91})
            elif "uses faiss" in text:
                results.append({"label": "ENTAILMENT", "score": 0.94})
            else:
                results.append({"label": "NEUTRAL", "score": 0.76})

        return results


class AlwaysNeutralPipeline:
    def __call__(self, inputs, truncation=True):
        return [{"label": "NEUTRAL", "score": 0.999} for _ in inputs]


def test_nli_verifier_labels_supported_claim():
    verifier = NLIClaimVerifier(nli_pipeline=DummyNLIPipeline())

    claims = ["The system uses FAISS for retrieval."]

    retrieved_chunks = [
        {
            "chunk_id": "doc_p1_c0",
            "source": "doc.pdf",
            "page_number": 1,
            "text": "The system uses FAISS for retrieval and stores embeddings in a vector index.",
            "score": 0.88,
            "rerank_score": 0.95,
        }
    ]

    results = verifier.verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
    )

    assert len(results) == 1
    assert results[0]["label"] == "Supported"
    assert results[0]["nli_label"] == "ENTAILMENT"
    assert results[0]["chunk_id"] == "doc_p1_c0"
    assert results[0]["source"] == "doc.pdf"
    assert results[0]["page_number"] == 1


def test_nli_verifier_labels_contradicted_claim():
    verifier = NLIClaimVerifier(nli_pipeline=DummyNLIPipeline())

    claims = ["The system does not use FAISS for retrieval."]

    retrieved_chunks = [
        {
            "chunk_id": "doc_p1_c0",
            "source": "doc.pdf",
            "page_number": 1,
            "text": "The system uses FAISS for retrieval and stores embeddings in a vector index.",
            "score": 0.88,
            "rerank_score": 0.95,
        }
    ]

    results = verifier.verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
    )

    assert len(results) == 1
    assert results[0]["label"] == "Contradicted"
    assert results[0]["nli_label"] == "CONTRADICTION"


def test_nli_verifier_labels_not_enough_evidence():
    verifier = NLIClaimVerifier(nli_pipeline=DummyNLIPipeline())

    claims = ["The system supports audio input."]

    retrieved_chunks = [
        {
            "chunk_id": "doc_p2_c0",
            "source": "doc.pdf",
            "page_number": 2,
            "text": "The system currently supports PDF text extraction and metadata-aware chunking.",
            "score": 0.70,
            "rerank_score": 0.80,
        }
    ]

    results = verifier.verify_claims(
        claims=claims,
        retrieved_chunks=retrieved_chunks,
    )

    assert len(results) == 1
    assert results[0]["label"] == "Not enough evidence"
    assert results[0]["nli_label"] == "NEUTRAL"


def test_nli_verifier_returns_empty_list_for_no_claims():
    verifier = NLIClaimVerifier(nli_pipeline=DummyNLIPipeline())

    results = verifier.verify_claims(
        claims=[],
        retrieved_chunks=[],
    )

    assert results == []


class RiskFrameworkPipeline:
    def __call__(self, inputs, truncation=True):
        outputs = []
        for item in inputs:
            premise = item.split(" </s></s> ", 1)[0].lower()
            if all(term in premise for term in ("govern", "map", "measure", "manage")):
                outputs.append({"label": "ENTAILMENT", "score": 0.98})
            else:
                outputs.append({"label": "NEUTRAL", "score": 0.99})
        return outputs


def _risk_framework_chunks():
    return [
        {
            "chunk_id": "playbook",
            "source": "framework.pdf",
            "page_number": 2,
            "text": "Additional resources related to the Framework are included in the AI RMF Playbook at a public URL.",
            "rerank_score": 3.5,
        },
        {
            "chunk_id": "core-functions",
            "source": "framework.pdf",
            "page_number": 3,
            "text": "The Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE.",
            "rerank_score": 2.5,
        },
    ]


def test_list_claim_selects_sentence_containing_list_items():
    verifier = NLIClaimVerifier(nli_pipeline=RiskFrameworkPipeline())
    claim = "The main functions are GOVERN, MAP, MEASURE, and MANAGE."

    result = verifier.verify_claims([claim], _risk_framework_chunks())[0]

    assert result["label"] == "Supported"
    assert result["chunk_id"] == "core-functions"
    assert "GOVERN, MAP, MEASURE, and MANAGE" in result["evidence"]
    assert result["selected_evidence_rank"] == 1


def test_better_evidence_outranks_unrelated_playbook_sentence():
    verifier = NLIClaimVerifier(nli_pipeline=RiskFrameworkPipeline())
    claim = "The AI Risk Management Framework functions are GOVERN, MAP, MEASURE, and MANAGE."

    result = verifier.verify_claims([claim], _risk_framework_chunks())[0]

    assert "Playbook" not in result["evidence"]
    assert result["candidate_evidence_checked"][0].startswith("The Core")


def test_entailment_wins_when_another_top_candidate_is_neutral():
    verifier = NLIClaimVerifier(nli_pipeline=RiskFrameworkPipeline())
    claim = "The framework functions are GOVERN, MAP, MEASURE, and MANAGE."

    result = verifier.verify_claims([claim], _risk_framework_chunks())[0]

    assert len(result["candidate_evidence_checked"]) == 2
    assert result["nli_label"] == "ENTAILMENT"
    assert result["label"] == "Supported"


class IndependentRiskPipeline:
    def __call__(self, inputs, truncation=True):
        outputs = []
        for item in inputs:
            premise, hypothesis = item.lower().split(" </s></s> ", 1)
            supported = (
                "adversarial attacks" in premise and "adversarial attacks" in hypothesis
            ) or (
                "misuse of foundation model capabilities" in premise
                and "misuse of foundation model capabilities" in hypothesis
            )
            outputs.append({
                "label": "ENTAILMENT" if supported else "NEUTRAL",
                "score": 0.96,
            })
        return outputs


def test_atomic_risk_claims_are_supported_independently():
    from src.claim_extractor import extract_claims

    claims = extract_claims(
        "GenAI systems have risks from adversarial attacks, misuse of foundation "
        "model capabilities, and imaginary moon failures."
    )
    chunks = [
        {
            "text": "Risk profiles help assess risks from adversarial attacks.",
            "chunk_id": "attacks",
            "rerank_score": 2.0,
        },
        {
            "text": "Guidance addresses the misuse of foundation model capabilities.",
            "chunk_id": "misuse",
            "rerank_score": 1.9,
        },
        {
            "text": "This unrelated sentence describes ordinary software maintenance.",
            "chunk_id": "other",
            "rerank_score": 1.0,
        },
    ]

    results = NLIClaimVerifier(
        nli_pipeline=IndependentRiskPipeline()
    ).verify_claims(claims, chunks)

    assert [result["label"] for result in results] == [
        "Supported",
        "Supported",
        "Not enough evidence",
    ]
    assert results[0]["chunk_id"] == "attacks"
    assert results[1]["chunk_id"] == "misuse"


def test_neutral_nli_is_overridden_when_all_uppercase_list_items_match():
    claim = (
        "The main functions of the framework are the four high-level functions: "
        "GOVERN, MAP, MEASURE, and MANAGE."
    )
    chunks = [{"text": "The Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE."}]

    result = NLIClaimVerifier(nli_pipeline=AlwaysNeutralPipeline()).verify_claims(claims=[claim], retrieved_chunks=chunks)[0]

    assert result["label"] == "Supported"
    assert result["nli_original_label"] == "NEUTRAL"
    assert result["support_override_applied"] is True
    assert result["matched_key_terms"] == ["govern", "map", "measure", "manage"]


def test_neutral_nli_is_partially_supported_when_most_list_items_match():
    claim = "The controls include ALPHA, BRAVO, CHARLIE, and DELTA."
    chunks = [{"text": "The documented controls include ALPHA, BRAVO, and CHARLIE for this process."}]

    result = NLIClaimVerifier(nli_pipeline=AlwaysNeutralPipeline()).verify_claims([claim], chunks)[0]

    assert result["label"] == "Partially Supported"
    assert result["support_override_applied"] is True


def test_neutral_nli_supports_complete_generic_list_without_domain_logic():
    claim = "The release stages are ALPHA, BETA, GAMMA, and DELTA."
    chunks = [{"text": "The documented release stages are ALPHA, BETA, GAMMA, and DELTA."}]

    result = NLIClaimVerifier(nli_pipeline=AlwaysNeutralPipeline()).verify_claims([claim], chunks)[0]

    assert result["label"] == "Supported"
    assert "AI Risk Management Framework" not in result["support_override_reason"]


def test_neutral_nli_without_strong_overlap_is_not_overridden():
    claim = "The system supports encrypted audio uploads."
    chunks = [{"text": "The application extracts text and metadata from PDF documents."}]

    result = NLIClaimVerifier(nli_pipeline=AlwaysNeutralPipeline()).verify_claims([claim], chunks)[0]

    assert result["label"] == "Not enough evidence"
    assert result["support_override_applied"] is False
    assert result["claim_evidence_keyword_overlap"] < 0.65

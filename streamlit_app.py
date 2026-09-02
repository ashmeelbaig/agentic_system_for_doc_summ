"""Professional Streamlit UI for claim-grounded document analysis."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv

try:
    import streamlit as st
except ImportError:  # Keep formatting helpers importable in test environments.
    st = None

from src.generators.factory import get_configured_model_ids
from src.pipeline import PipelineConfigurationError, answer_question


PAGE_TITLE = "Claim-Grounded Agentic RAG for Technical Documents"
UPLOAD_ROOT = Path("outputs/streamlit_uploads")


def format_percentage(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "N/A"


def format_yes_no(value: Any) -> str:
    if value is None:
        return "N/A"
    return "Yes" if bool(value) else "No"


def format_model_comparison(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the human-readable model comparison table."""
    rows = []
    for item in result.get("model_comparison", []):
        latency = item.get("latency_seconds")
        rows.append(
            {
                "Model": item.get("model_name"),
                "Status": item.get("status"),
                "Faithfulness": format_percentage(item.get("faithfulness_score")),
                "Revision": item.get("revision_decision") or "N/A",
                "Safety Action": item.get("safety_action") or "N/A",
                "Refusal": format_yes_no(item.get("is_refusal_answer")),
                "Latency": f"{float(latency):.2f}s" if latency is not None else "N/A",
            }
        )
    return rows


def format_faithfulness(score: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields = (
        ("Total claims", "total_claims"),
        ("Supported claims", "supported_claims"),
        ("Partially supported", "partially_supported_claims"),
        ("Unsupported", "unsupported_claims"),
        ("Contradicted", "contradicted_claims"),
        ("Not enough evidence", "not_enough_evidence_claims"),
    )
    rows = [{"Metric": label, "Value": score.get(key, 0)} for label, key in fields]
    rows.append(
        {
            "Metric": "Faithfulness score",
            "Value": format_percentage(score.get("faithfulness_score")),
        }
    )
    return rows


def format_revision_decision(decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"Field": "Decision", "Value": decision.get("decision", "N/A")},
        {"Field": "Reason", "Value": decision.get("reason", "N/A")},
        {"Field": "Instruction", "Value": decision.get("instruction", "N/A")},
        {"Field": "Answer focus", "Value": decision.get("answer_focus", "N/A")},
        {
            "Field": "Should reverify",
            "Value": format_yes_no(decision.get("should_reverify")),
        },
    ]


def format_safety_gate(safety: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"Field": "Safe", "Value": format_yes_no(safety.get("is_safe"))},
        {"Field": "Reason", "Value": safety.get("reason", "N/A")},
        {"Field": "Action", "Value": safety.get("action", "N/A")},
    ]


def format_claim_verification(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        score = item.get("nli_score")
        rows.append(
            {
                "Claim": item.get("claim"),
                "Label": item.get("label"),
                "NLI label": item.get("nli_label"),
                "NLI score": round(float(score), 3) if score is not None else None,
                "Evidence": item.get("evidence"),
                "Source": item.get("source"),
                "Page": item.get("page_number"),
                "Override applied": format_yes_no(
                    item.get("support_override_applied")
                ),
            }
        )
    return rows


def save_uploaded_pdfs(uploaded_files, upload_root: Path = UPLOAD_ROOT):
    """Write uploaded PDFs to a unique temporary directory for one pipeline run."""
    upload_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run_", dir=upload_root))
    paths = []
    try:
        for index, uploaded in enumerate(uploaded_files, start=1):
            safe_name = Path(uploaded.name).name
            if Path(safe_name).suffix.lower() != ".pdf":
                raise ValueError("Only PDF uploads are supported.")
            file_directory = directory / str(index)
            file_directory.mkdir()
            path = file_directory / safe_name
            path.write_bytes(uploaded.getvalue())
            paths.append(str(path))
        return directory, paths
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def remove_uploaded_pdfs(directory: Path) -> None:
    if directory and directory.exists() and directory.parent.resolve() == UPLOAD_ROOT.resolve():
        shutil.rmtree(directory, ignore_errors=True)


def _clear_uploads() -> None:
    st.session_state["upload_widget_key"] += 1
    st.session_state.pop("last_result", None)


def _render_sidebar():
    mode = os.getenv("GENERATOR_MODE", "multi_hf").strip().lower()
    st.sidebar.title("Claim-Grounded RAG")
    st.sidebar.caption("Technical document analysis and model evaluation")
    st.sidebar.divider()
    st.sidebar.write(f"**Generator mode:** `{mode}`")
    st.sidebar.write("**Configured models**")
    for model_id in get_configured_model_ids():
        st.sidebar.code(model_id)
    token_loaded = bool(os.getenv("HF_TOKEN", "").strip())
    st.sidebar.write(f"**HF token status:** {'Loaded' if token_loaded else 'Missing'}")
    st.sidebar.divider()
    st.sidebar.subheader("Upload PDF documents")
    uploaded = st.sidebar.file_uploader(
        "Choose one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_upload_{st.session_state['upload_widget_key']}",
    )
    st.sidebar.write(f"Uploaded files: **{len(uploaded)}**")
    for item in uploaded:
        st.sidebar.write(f"- {item.name}")
    st.sidebar.button("Clear uploaded files", on_click=_clear_uploads)
    return mode, uploaded, token_loaded


def _render_summary(result):
    retrieval = result.get("retrieval", {})
    comparison = result.get("model_comparison", [])
    successful = sum(item.get("status") == "success" for item in comparison)
    failed = sum(item.get("status") == "failed" for item in comparison)
    values = (
        ("Retrieval confidence", retrieval.get("retrieval_confidence", {}).get("label", "N/A")),
        ("Retrieval attempts", len(retrieval.get("retrieval_attempts", []))),
        ("Evidence chunks", len(retrieval.get("retrieved_evidence", []))),
        ("Models", len(comparison)),
        ("Successful", successful),
        ("Failed", failed),
    )
    for column, (label, value) in zip(st.columns(6), values):
        column.metric(label, value)


def _render_model_tab(model_id: str, model: Dict[str, Any]) -> None:
    status = model.get("status", "failed")
    if status != "success":
        st.table([{"Field": "Status", "Value": status}])
        st.error(model.get("error", "Model generation failed."))
        return

    st.subheader("Final Answer")
    st.info(model.get("final_answer", "No final answer was returned."))
    score = model.get("faithfulness_score", {})
    revision = model.get("revision_decision", {})
    safety = model.get("final_safety_gate", {})
    overview = [
        {"Field": "Status", "Value": status},
        {"Field": "Faithfulness", "Value": format_percentage(score.get("faithfulness_score"))},
        {"Field": "Revision", "Value": revision.get("decision", "N/A")},
        {"Field": "Answer focus", "Value": revision.get("answer_focus", "N/A")},
        {"Field": "Safety action", "Value": safety.get("action", "N/A")},
        {"Field": "Refusal", "Value": format_yes_no(model.get("is_refusal_answer"))},
    ]
    st.table(overview)
    with st.expander("Draft answer"):
        st.write(model.get("draft_answer", "N/A"))
    with st.expander("Candidate final answer"):
        st.write(model.get("candidate_final_answer", "N/A"))

    left, middle, right = st.columns(3)
    with left:
        st.subheader("Faithfulness")
        st.table(format_faithfulness(score))
    with middle:
        st.subheader("Revision decision")
        st.table(format_revision_decision(revision))
    with right:
        st.subheader("Final safety gate")
        st.table(format_safety_gate(safety))
    st.write(f"**Refusal:** {format_yes_no(model.get('is_refusal_answer'))}")

    st.subheader("Claim Verification")
    claims = format_claim_verification(model.get("verification_results", []))
    if claims:
        st.dataframe(claims, use_container_width=True, hide_index=True)
    else:
        st.info("No claim-verification rows are available.")


def _render_models(result):
    st.header("Model Comparison")
    rows = format_model_comparison(result)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No models were run.")

    st.header("Model Results")
    models = result.get("model_results", {})
    if not models:
        st.info("No model results are available.")
        return
    tabs = st.tabs(list(models))
    for tab, (model_id, model) in zip(tabs, models.items()):
        with tab:
            _render_model_tab(model_id, model)


def _render_evidence(result):
    st.header("Retrieved Evidence")
    evidence = result.get("retrieval", {}).get("retrieved_evidence", [])
    if not evidence:
        st.info("No retrieved evidence is available.")
        return
    for index, item in enumerate(evidence, start=1):
        source = item.get("source") or "Unknown source"
        page = item.get("page_number")
        rerank = item.get("rerank_score")
        rerank_text = f"{float(rerank):.3f}" if rerank is not None else "N/A"
        with st.expander(f"{source} | Page {page or 'N/A'} | Rerank {rerank_text}"):
            details = [
                {"Field": "Chunk ID", "Value": item.get("chunk_id") or "N/A"},
                {"Field": "Source", "Value": source},
                {"Field": "Page number", "Value": page or "N/A"},
                {"Field": "Similarity score", "Value": item.get("similarity_score", "N/A")},
                {"Field": "Rerank score", "Value": rerank_text},
            ]
            st.table(details)
            st.write(item.get("text", ""))


def _render_download(result):
    saved_path = result.get("saved_output_path")
    if not saved_path:
        return
    st.success(f"Saved JSON: {saved_path}")
    path = Path(saved_path)
    data = path.read_bytes() if path.is_file() else json.dumps(result, indent=2).encode("utf-8")
    st.download_button(
        "Download JSON",
        data=data,
        file_name=path.name or "rag_result.json",
        mime="application/json",
    )


def render_app() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements.txt first.")

    load_dotenv()
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.caption("Compare grounded answers, claims, evidence, revision, and safety decisions.")
    st.session_state.setdefault("upload_widget_key", 0)
    mode, uploaded_files, token_loaded = _render_sidebar()

    st.subheader("Ask a question")
    query = st.text_area(
        "Question",
        placeholder="What does the uploaded document say about ...?",
        label_visibility="collapsed",
    )

    if st.button("Run Analysis", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        elif not query.strip():
            st.warning("Please enter a question.")
        elif not token_loaded:
            st.error(
                "Hugging Face token is missing. Please add HF_TOKEN to your .env file."
            )
        else:
            upload_directory = None
            try:
                upload_directory, paths = save_uploaded_pdfs(uploaded_files)
                with st.spinner("Running retrieval, generation, and verification..."):
                    st.session_state["last_result"] = answer_question(
                        query=query,
                        selected_pdf_paths=paths,
                        generator_mode=mode,
                        save_output=True,
                    )
            except (PipelineConfigurationError, ValueError) as exc:
                st.error(str(exc))
            except Exception:
                st.error(
                    "The RAG pipeline failed. Check the server logs and configuration."
                )
            finally:
                if upload_directory is not None:
                    remove_uploaded_pdfs(upload_directory)

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("Question")
        st.write(result.get("query", ""))
        _render_summary(result)
        _render_models(result)
        _render_evidence(result)
        _render_download(result)


if __name__ == "__main__":
    render_app()

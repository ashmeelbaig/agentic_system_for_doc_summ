import streamlit_app


class UploadedFile:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def getvalue(self):
        return self.content


def test_streamlit_module_imports_without_running_server():
    assert callable(streamlit_app.render_app)


def test_model_comparison_formatting_uses_expected_columns():
    result = {
        "model_comparison": [
            {
                "model_name": "Qwen/example",
                "provider": "huggingface_api",
                "status": "success",
                "faithfulness_score": 1.0,
                "revision_decision": "keep",
                "safety_action": "send",
                "is_refusal_answer": False,
                "latency_seconds": 0.25,
                "ignored": "value",
            }
        ]
    }

    assert streamlit_app.format_model_comparison(result) == [
        {
            "Model": "Qwen/example",
            "Status": "success",
            "Faithfulness": "100%",
            "Revision": "keep",
            "Safety Action": "send",
            "Refusal": "No",
            "Latency": "0.25s",
        }
    ]


def test_faithfulness_formatting_is_table_friendly():
    rows = streamlit_app.format_faithfulness(
        {"total_claims": 1, "supported_claims": 1, "faithfulness_score": 1.0}
    )
    assert rows[0] == {"Metric": "Total claims", "Value": 1}
    assert rows[-1] == {"Metric": "Faithfulness score", "Value": "100%"}


def test_revision_formatting_is_table_friendly():
    rows = streamlit_app.format_revision_decision(
        {
            "decision": "keep",
            "reason": "Supported.",
            "instruction": "No revision needed.",
            "answer_focus": "good",
            "should_reverify": False,
        }
    )
    assert {"Field": "Decision", "Value": "keep"} in rows
    assert {"Field": "Should reverify", "Value": "No"} in rows


def test_safety_formatting_is_table_friendly():
    rows = streamlit_app.format_safety_gate(
        {"is_safe": True, "reason": "Passed.", "action": "send"}
    )
    assert rows == [
        {"Field": "Safe", "Value": "Yes"},
        {"Field": "Reason", "Value": "Passed."},
        {"Field": "Action", "Value": "send"},
    ]


def test_uploaded_pdfs_are_temporary_and_preserve_safe_names(tmp_path):
    upload_root = tmp_path / "streamlit_uploads"
    directory, paths = streamlit_app.save_uploaded_pdfs(
        [
            UploadedFile("first.pdf", b"first"),
            UploadedFile("../second.pdf", b"second"),
        ],
        upload_root=upload_root,
    )

    assert [streamlit_app.Path(path).name for path in paths] == [
        "first.pdf",
        "second.pdf",
    ]
    assert [streamlit_app.Path(path).read_bytes() for path in paths] == [
        b"first",
        b"second",
    ]
    streamlit_app.shutil.rmtree(directory)

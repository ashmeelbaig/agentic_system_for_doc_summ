# Claim-Grounded Agentic RAG for Technical Documents

This repository contains a terminal-based Retrieval-Augmented Generation (RAG) prototype for answering questions about one or more PDF documents. It retrieves and reranks evidence, extracts atomic claims, verifies each claim with Natural Language Inference (NLI), optionally revises weak answers, and applies a final safety gate before returning the result.

The recommended configuration runs two hosted models through Hugging Face Inference Providers. Retrieval and reranking happen once per question, and the same evidence is passed to both models so their answers and faithfulness results can be compared fairly.

## Active model configuration

The default `.env.example` configuration uses `GENERATOR_MODE=multi_hf` and these hosted models:

| Role | Model | Execution |
|---|---|---|
| Answer generation | `Qwen/Qwen2.5-Coder-32B-Instruct` | Hugging Face Inference Providers |
| Answer generation | `meta-llama/Llama-3.1-8B-Instruct:nscale` | Hugging Face Inference Providers, preserving the `:nscale` suffix |
| Embedding and FAISS retrieval | `sentence-transformers/all-MiniLM-L6-v2` | Local Transformers cache |
| Evidence reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local Transformers cache |
| Claim verification | `typeform/distilbert-base-uncased-mnli` | Local Transformers pipeline |

The hosted answer generator uses `InferenceClient` with `provider="auto"`. Provider errors are normalized into controlled messages so credentials and raw request details are not written to logs or JSON. A failure from one hosted model does not prevent the other model from running.

Two smaller answer models are supported by the optional local generator abstraction but are not active by default:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

They are used only when `GENERATOR_MODE=multi_model` and `LOCAL_MODEL_IDS` is explicitly non-empty. Leaving `LOCAL_MODEL_IDS=` prevents their initialization and download.

## Processing workflow

```text
PDF file(s)
    -> page-aware text extraction and cleaning
    -> overlapping metadata-aware chunks
    -> sentence-transformer embeddings and FAISS index

User question
    -> query safety check
    -> retrieval attempt 1: original query
    -> retrieval attempt 2: rewritten query, when needed
    -> retrieval attempt 3: keyword query, when needed
    -> cross-encoder reranking
    -> retrieval-confidence decision
    -> document prompt-injection detection and sanitization
    -> one shared evidence set
         -> model A generation -> claims -> NLI -> revision -> safety gate
         -> model B generation -> claims -> NLI -> revision -> safety gate
    -> terminal comparison and one combined JSON result
```

Retrieval is not repeated for each answer model. By default, each attempt retrieves up to 12 candidates and reranks the best four. If retrieval confidence remains too low after the retry sequence, the system refuses before answer generation.

## Guardrails and verification

### Query and evidence safety

The application checks user queries for unsafe instructions before retrieval. Retrieved document text is also checked for prompt-injection patterns. Suspicious document instructions are treated as document content and sanitized before being passed to answer models.

### Atomic claim extraction

Generated answers are split into independently verifiable claims. The extractor includes special handling for compound lists, allowing separate verification of claims that would otherwise be grouped into one sentence.

### NLI verification

`NLIClaimVerifier` ranks candidate evidence sentences for every claim and uses `typeform/distilbert-base-uncased-mnli` to classify candidates as entailment, contradiction, or neutral. These labels map to:

- `Supported`
- `Contradicted`
- `Not enough evidence`

A conservative lexical-support override handles strongly matching lists and technical terms that the NLI model may classify as neutral.

### Faithfulness score

Faithfulness is calculated as:

```text
supported claims / total verified claims
```

The saved score also includes counts for partially supported, unsupported, contradicted, and insufficient-evidence claims.

### Answer Revision Agent

The revision agent is deterministic and rule-based; it does not make an additional remote model call by itself. It asks the current answer generator for one revised answer when the draft:

- contains contradicted or unsupported claims;
- has no useful claims;
- is poorly focused on the question;
- refuses despite high-confidence evidence; or
- has a faithfulness score below `0.75`.

Revised answers are extracted and verified again before continuing.

### Final safety gate

The final answer is sent only when it is non-empty, focused, contains verifiable claims, has no contradicted or insufficient-evidence claims, and reaches the required faithfulness threshold. Otherwise, the system returns a controlled insufficient-evidence response.

### Optional agent-role orchestration

`src/crewai_orchestrator.py` exposes the same deterministic guardrail sequence as named Safety, Retrieval, Answer, Verification, and Revision roles. CrewAI itself is optional, and this module does not replace the default workflow in `main.py`; it provides a separately testable orchestration boundary for future integration.

## Generator modes

| Mode | Answer generator behavior |
|---|---|
| `fast` | Local `google/flan-t5-small` |
| `quality` | Local `google/flan-t5-base` |
| `quality_plus` | Local `google/flan-t5-large` |
| `multi_hf` | Hosted models listed in `HF_MODEL_IDS`; recommended mode |
| `multi_model` | Hosted models plus explicitly configured `LOCAL_MODEL_IDS` |

The three FLAN modes use the legacy `AnswerGenerator`, including an extractive fallback for weak generations. They load models locally and may download them on first use.

In `multi_model`, local causal models are loaded one at a time through `LocalTransformersGenerator`. It uses a tokenizer chat template when available, falls back to a plain instruction prompt otherwise, and releases model/tokenizer references after each model pipeline. Garbage collection and CUDA cache cleanup are also requested. If no local IDs are configured, hosted models still run and the application prints:

```text
No local models configured. Skipping local Transformers generation.
```

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

On macOS or Linux, activate the environment with `source venv/bin/activate`.

PyTorch, FAISS, and sentence-transformers may require platform-specific installation choices for GPU acceleration. The default code also works on CPU, though model initialization and NLI verification can be slower.

## Configuration

Copy the example configuration and put the real token only in `.env`:

```powershell
Copy-Item .env.example .env
```

Recommended `.env` configuration:

```dotenv
GENERATOR_MODE=multi_hf

HF_TOKEN=your_huggingface_token_here
HF_MODEL_IDS=Qwen/Qwen2.5-Coder-32B-Instruct,meta-llama/Llama-3.1-8B-Instruct:nscale

LOCAL_MODEL_IDS=
LOCAL_MODEL_DEVICE=auto
LOCAL_MODEL_DTYPE=auto
```

`HF_TOKEN` is required for hosted generation. The token value is never intentionally printed or saved. `.env`, `.env.local`, and `*.env` are ignored by Git; do not place a real token in `.env.example`.

To enable optional local answer models later:

```dotenv
GENERATOR_MODE=multi_model
LOCAL_MODEL_IDS=Qwen/Qwen2.5-0.5B-Instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0
LOCAL_MODEL_DEVICE=auto
LOCAL_MODEL_DTYPE=auto
```

This optional configuration downloads model files into the normal Hugging Face cache on first use. It does not manually save them inside the repository.

## Running the application

Place PDF files in `data/`, then run:

```powershell
python main.py
```

The terminal lets you select one PDF or all available PDFs. Document loading, chunking, embedding-index creation, reranker initialization, and NLI initialization happen before the interactive question loop. Enter `exit` to stop.

For every answerable question in `multi_hf`, the application prints each model's provider, status, final answer, faithfulness score, revision decision, safety action, refusal status, and latency.

## How to run the Streamlit UI

The Streamlit interface supports text questions and uses the same hosted multi-model generation path as the CLI. PDFs are uploaded directly in the browser; they do not need to exist in `data/`.

1. Create `.env` from the example file:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Add your Hugging Face token to the local `.env`. Keep `GENERATOR_MODE=multi_hf` and do not commit this file.

3. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the UI:

   ```powershell
   streamlit run streamlit_app.py
   ```

Use the sidebar to upload one or more PDFs and inspect the generator mode, configured models, and token status. Enter a text question and select **Run Analysis**. Uploaded PDFs are written to an ignored temporary run directory under `outputs/streamlit_uploads/` and removed after processing. The page displays retrieval statistics, model comparison, model-specific tabs, structured revision and safety details, retrieved evidence, claim-verification tables, and a download button for the saved JSON result.

The current UI intentionally supports text questions only. It does not accept audio or images. Backend models are initialized for each submitted question, so repeated questions can be slower than the long-running CLI session.

## Output format

Results are written to `outputs/` with timestamped names. The directory is ignored by Git.

Multi-model output has this top-level structure:

```json
{
  "pdf_name": "document.pdf",
  "query": "...",
  "generator_mode": "multi_hf",
  "retrieval": {
    "used_query": "...",
    "retrieval_confidence": {},
    "retrieval_attempts": [],
    "retrieved_evidence": []
  },
  "model_results": {
    "Qwen/Qwen2.5-Coder-32B-Instruct": {},
    "meta-llama/Llama-3.1-8B-Instruct:nscale": {}
  },
  "model_comparison": []
}
```

Each successful model result includes:

- provider and status;
- attempted generation methods and controlled attempt failures;
- draft, revised candidate, and final answers;
- extracted atomic claims and verification results;
- faithfulness summary;
- revision and final-safety decisions; and
- end-to-end model-pipeline latency.

Failed models receive an isolated result containing their model name, provider, controlled error, attempted methods, attempt failures, and latency.

Single-model FLAN modes use the older `baseline_rag` and `claim_grounded_rag` JSON structure.

## Hosted model route smoke test

The standalone route test makes real Hugging Face requests to only the two active hosted models:

```powershell
python scripts/test_hf_model_routes.py
```

It tries `chat_completion` and then `text_generation`, prints sanitized status information, and writes:

```text
outputs/hf_model_route_test.json
```

The script never prints or stores `HF_TOKEN`. This smoke test is separate from unit tests and should be run only when live provider access is intended.

## Tests

Run the complete test suite with:

```powershell
python -m pytest -q
```

Tests mock hosted clients and local models. They do not call the real Hugging Face API or download local answer models. Coverage includes document loading, metadata chunking, retrieval and reranking, retry confidence, guardrails, claim extraction, NLI behavior, revision decisions, generator routing, per-model failure isolation, JSON serialization, and token non-disclosure.

## Evaluation utility

Saved single-model result files can be summarized into a CSV with:

```powershell
python scripts/evaluate_outputs.py
```

The evaluation module reads saved result metadata and aggregates fields such as generator mode/model, claim counts, faithfulness, refusal state, and retrieval confidence. The current evaluation utility primarily targets the single-model result structure.

## Repository structure

```text
main.py                              Interactive application and workflow wiring
streamlit_app.py                     Text-question Streamlit interface
data/                                Input PDFs
outputs/                             Generated JSON reports (Git-ignored)
scripts/evaluate_outputs.py          Result-summary utility
scripts/test_hf_model_routes.py      Live hosted-provider route smoke test
src/document_loader.py               PDF extraction and text cleaning
src/document_collection.py           Multi-PDF preparation
src/chunker.py                       Metadata-aware overlapping chunking
src/retriever.py                     Sentence embeddings and FAISS retrieval
src/retrieval_retry.py               Original/rewritten/keyword retry flow
src/retrieval_confidence.py          Evidence-confidence assessment
src/reranker.py                      Cross-encoder evidence reranking
src/safety_guardrails.py             Query and document-injection checks
src/claim_extractor.py               Sentence and atomic-claim extraction
src/nli_verifier.py                  NLI verification and evidence selection
src/scoring.py                       Faithfulness calculation
src/answer_revision_agent.py         Revision rules and final safety gate
src/crewai_orchestrator.py           Optional named-role workflow abstraction
src/result_saver.py                  Single/multi-model JSON persistence
src/pipeline.py                      Reusable one-question backend pipeline
src/generators/base.py               Shared generator interface
src/generators/hf_api_generator.py   Hosted Hugging Face adapter
src/generators/local_transformers_generator.py  Optional local causal-LM adapter
src/generators/factory.py            Model configuration and provider dispatch
src/generators/multi_model_runner.py Independent per-model guardrail pipelines
tests/                               Mocked unit and workflow tests
```

## Security and repository hygiene

- Never commit `.env` or any real Hugging Face token.
- Hosted-provider exceptions are converted to controlled messages before output.
- Local model/cache paths and generated outputs are ignored by Git.
- Unit tests use fake clients and models.
- Retrieved prompt-like instructions are sanitized before generation.
- The application does not execute instructions found inside documents.

## Current limitations

- This is an interactive prototype rather than a web service or production API.
- Models used for embeddings, reranking, NLI, and legacy FLAN modes may download on first use if they are not cached.
- Retrieval-confidence, revision, and final-safety decisions are heuristic and threshold-based.
- Claim extraction is deterministic and optimized for sentence/list patterns rather than full semantic decomposition.
- The application builds an in-memory FAISS index each time it starts.
- The evaluation script primarily understands the single-model output schema.

# Claim Grounded Agentic RAG for Technical Documents

This project is a prototype of a claim grounded Retrieval Augmented Generation system for technical documents.

The system allows a user to place a PDF document inside the `data` folder, ask questions from the terminal, and receive an answer grounded in retrieved document evidence. In addition to a standard RAG answer, the prototype extracts claims from the generated answer, verifies each claim against retrieved evidence, and calculates a faithfulness score.

## Project Goal

Large language models can generate useful summaries and answers, but they may also produce unsupported or incorrect statements. This project explores a lightweight agentic RAG workflow where the generated answer is not accepted directly. Instead, it is checked claim by claim against the retrieved document evidence.

The current prototype demonstrates around 30 percent of the final project idea. Some components are simplified at this stage using deterministic logic. Later versions can replace these parts with stronger open source models.

## Current Prototype Features

The current terminal based prototype supports:

1. PDF text extraction
2. Word based document chunking
3. Semantic embedding using a Hugging Face sentence transformer
4. FAISS based vector retrieval
5. Lightweight open source answer generation
6. Claim extraction from generated answer
7. Claim verification using semantic similarity
8. Faithfulness score calculation
9. Standard RAG baseline comparison
10. JSON result saving for later evaluation

## Models Used

The prototype uses open source Hugging Face models.

| Component | Model |
|---|---|
| Embedding and retrieval | `sentence-transformers/all-MiniLM-L6-v2` |
| Lightweight answer generation | `google/flan-t5-small` |

The current claim verification step uses semantic similarity with the embedding model. In the final version, this can be replaced or improved using a Natural Language Inference model.

## System Workflow

```text
PDF Document
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embedding Model
    ↓
FAISS Vector Search
    ↓
User Question
    ↓
Retrieved Evidence
    ↓
Answer Generation
    ↓
Claim Extraction
    ↓
Claim Verification
    ↓
Faithfulness Score
    ↓
Saved JSON Result
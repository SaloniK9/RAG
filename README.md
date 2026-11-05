# Engineering RAG

An integrated, local-first experimentation platform for retrieval-augmented
generation over engineering documents. This repository currently implements
**Phase 1** of [IMPLEMENTATION.md](./IMPLEMENTATION.md).

## Phase 1 capabilities

- One FastAPI backend and React/Tailwind dashboard.
- Upload and parse PDF, DOCX, and UTF-8 TXT documents.
- Preserve extracted page, section, table-row, file metadata, and exact
  engineering values/units in stored chunks.
- Fixed-size and recursive character chunking with configuration saved per
  document.
- Actual case-sensitive BM25 retrieval and cosine-similarity dense retrieval.
- Ollama adapters configured for `nomic-embed-text` and `qwen2.5:7b`.
- Naive RAG prompts that require evidence-grounded answers and citations.
- SQLite persistence for documents, chunks, and model-specific embeddings.
- Document management, configurable chunking/retrieval, source display, and
  request timing in the chat dashboard.

Dense vectors are generated and cached on the first dense query. BM25,
document ingestion, and the app itself do not require Ollama. Actual answer
generation always uses Ollama; the backend returns an explicit service error
if it is not running or the configured model is unavailable. No model output
or evaluation result is simulated.

## Prerequisites

- Python 3.13 (3.11+ supported by the selected dependencies).
- Node.js 20+ and npm.
- Ollama for model-backed dense retrieval and answer generation.

Install Ollama, then download the models:

```powershell
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

Ollama should be reachable at `http://localhost:11434`. Configure a different
address or model IDs with `OLLAMA_BASE_URL`, `RAG_EMBEDDING_MODEL`, and
`RAG_GENERATION_MODEL`.

## Run locally

In PowerShell, from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The API is at `http://localhost:8000` and its
interactive API reference is at `http://localhost:8000/docs`.

To store data outside the repository, set `RAG_DATA_DIR` to a local directory
before starting the backend. Uploaded files, SQLite records, chunks, and cached
embeddings are persisted there.

## Test

```powershell
python -m pytest backend\tests -q
```

Backend tests use a deterministic fake Ollama adapter only at the test boundary;
the application has no fake model path. The suite covers extraction, metadata
preservation, chunking, exact engineering token retrieval, cosine ranking,
document APIs, and explicit handling of unavailable Ollama.

## Phase 1 limitations

- Scanned PDFs need OCR; OCR is not included.
- Chunk size/overlap are character-based, not token-based.
- Dense search uses exact cosine scoring over persisted Ollama vectors; FAISS
  indexing is deferred.
- BM25 and dense are separately selectable; hybrid RRF and reranking are Phase 2.
- Generation requires a running Ollama instance and the configured Qwen model.
- Advanced/Corrective/Graph RAG, evaluation/experiments, and the later dashboard
  pages are intentionally deferred to their specified phases.

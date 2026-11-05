# IMPLEMENTATION.md
# Engineering RAG: A Modular and Configurable RAG Experimentation Platform

## 1. Project Overview

### Objective

Develop a modular Retrieval-Augmented Generation (RAG) platform specifically designed for engineering documents, technical manuals, standards, research papers, equipment datasheets and technical specifications.

The system should support multiple RAG configurations within one integrated application.

Users must be able to select different:
- Document chunking strategies.
- Embedding models.
- Retrieval algorithms.
- Reranking methods.
- RAG architectures.
- Evaluation methods.

The platform should allow users to compare configurations and determine which pipeline performs best for different types of engineering queries.

This is a research-oriented project focused on controlled experiments, measurable improvements, reproducibility and engineering-data accuracy.

**Core research question:**

Which RAG pipeline configuration performs best for engineering documents, under what conditions, and by how much compared with the baseline?

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Frontend | React + Tailwind CSS |
| Backend | Python + FastAPI |
| PDF Processing | PyMuPDF / Docling |
| LLM Runtime | Ollama |
| Generation Model | Qwen2.5 7B |
| Embedding Model | nomic-embed-text |
| Additional Embeddings | BGE-small, MiniLM |
| Vector Database | FAISS initially |
| Keyword Retrieval | BM25 / rank-bm25 |
| Dense Retrieval | Sentence Transformers / Ollama |
| Hybrid Retrieval | BM25 + Dense using RRF |
| Reranking | BGE Cross-Encoder |
| Evaluation | RAGAS + Custom Metrics |
| Experiment Tracking | CSV / JSON; MLflow optional |
| Deployment | Microsoft Azure |

Keep the implementation modular so individual technologies can be replaced without rewriting the complete application.

---

## 3. Overall System Architecture

Implement the following integrated pipeline:

1. Document Upload
2. Document Parsing and Extraction
3. Text Preprocessing
4. Configurable Chunking
5. Embedding Generation
6. Indexing and Storage
7. Configurable Retrieval
8. Optional Reranking
9. RAG Architecture Selection
10. LLM Answer Generation
11. Engineering-Aware Validation
12. Evaluation and Experiment Tracking

The frontend should provide configuration controls for the pipeline.

All selected configurations must connect to actual backend implementations. Do not implement UI-only toggles.

---

## 4. Document Ingestion and Processing

Support the following document formats:

- PDF
- DOCX
- TXT

The system should extract:
- Text and paragraphs.
- Headings and subheadings.
- Tables and tabular specifications.
- Page numbers.
- Section information.
- Document metadata.
- Numerical values and measurement units.

Use PyMuPDF or Docling for document extraction.

Preserve document structure wherever possible.

### Important engineering-specific requirement

Engineering documents frequently contain tables, specifications, equations and numerical measurements.

Avoid flattening tables in a way that separates values from their corresponding labels, units or headings.

Each extracted document should retain:
- Document ID.
- Document name.
- Page number.
- Section heading.
- Content.
- Source location.

---

## 5. Chunking Module

Implement five independently selectable chunking strategies.

### A. Fixed-size Chunking
- Divide documents into fixed token or character lengths.
- Configurable chunk size.
- Configurable overlap.

### B. Sentence-level Chunking
- Split text into individual sentences.
- Support grouping adjacent sentences.
- Preserve sentence boundaries.

### C. Recursive Chunking
- Use hierarchical separators.
- Preserve paragraphs and sentence boundaries wherever possible.
- Configurable chunk size and overlap.

### D. Semantic Chunking
- Group sentences according to semantic similarity.
- Use embeddings to identify topic boundaries.
- Configurable similarity threshold.

### E. Structure-aware Chunking
- Use headings, sections and subsections.
- Preserve relationships between headings and associated content.
- Keep tables and technical specifications together.
- Retain document hierarchy in metadata.

### Implementation requirements

- All chunkers must follow a common interface.
- Each chunker must be independently testable.
- Chunk metadata must be preserved.
- The same document should be processable through different chunking strategies.
- Save the selected chunking configuration for each experiment.

---

## 6. Embedding Module

Implement a configurable embedding interface.

Initial embedding models:

1. nomic-embed-text (Ollama).
2. BAAI/bge-small-en-v1.5.
3. sentence-transformers/all-MiniLM-L6-v2.

Requirements:

- Allow users to select an embedding model.
- Generate embeddings for document chunks.
- Generate query embeddings using the same compatible model.
- Handle embedding dimensions correctly.
- Record model name and version.
- Re-index documents when the embedding model changes.
- Maintain separate indexes for incompatible embedding spaces.

Embedding generation should be reusable across retrieval configurations when the chunking and embedding configuration remains unchanged.

---

## 7. Retrieval Module

Implement four retrieval configurations.

### A. BM25 Retrieval

- Implement keyword-based sparse retrieval.
- Use rank-bm25 or an equivalent library.
- Support configurable top-K.
- Preserve exact technical terms and numerical tokens.

### B. Dense / Semantic Retrieval

- Use vector similarity search.
- Implement cosine similarity or another documented similarity metric.
- Use FAISS initially.
- Retrieve semantically relevant chunks even when query wording differs from the document.

### C. Hybrid Retrieval

Combine BM25 and dense retrieval.

Implement Reciprocal Rank Fusion (RRF).

Requirements:
- Retrieve candidates from both retrieval methods.
- Combine rankings using RRF.
- Remove duplicate chunks.
- Support configurable retrieval parameters.
- Preserve source metadata.

### D. Hybrid Retrieval with Reranking

- Retrieve candidate chunks using hybrid retrieval.
- Pass candidates through a cross-encoder reranker.
- Reorder chunks according to query-document relevance.
- Return the final top-K chunks.

Use BAAI/bge-reranker-base or an appropriate cross-encoder model.

Every retrieval method must implement a common interface.

---

## 8. RAG Architecture Module

Implement the following RAG architectures.

### A. Naive RAG

1. Receive user query.
2. Retrieve relevant chunks.
3. Pass retrieved context to the LLM.
4. Generate an answer grounded in retrieved evidence.
5. Display source citations.

### B. Advanced RAG

Implement:
- Query preprocessing.
- Query transformation or expansion where appropriate.
- Improved retrieval.
- Optional reranking.
- Context filtering.
- Context preparation.
- Grounded answer generation.

### C. Corrective RAG

Implement a retrieval evaluation and correction mechanism.

Expected workflow:

1. Receive the query.
2. Retrieve relevant documents.
3. Evaluate document relevance.
4. Detect insufficient or irrelevant retrieval results.
5. Trigger alternative retrieval or query reformulation when needed.
6. Retrieve improved evidence.
7. Generate the final response using validated context.

The system must identify when the available evidence is insufficient.

### D. Graph RAG (Experimental)

Implement as a separate experimental architecture.

Expected functionality:
- Extract entities and relationships from engineering documents.
- Construct a knowledge graph.
- Identify related entities.
- Retrieve relevant graph information.
- Combine graph-derived evidence with document retrieval.

Do not make Graph RAG a mandatory dependency for other RAG architectures.

---

## 9. Engineering-Aware Validation Module

This is a critical research component.

Engineering documents contain numerical values, units, symbols and measurements. Small differences can significantly change technical meaning.

Implement validation for:

- Numerical values.
- Measurement units.
- Decimal precision.
- Scientific notation.
- Mathematical symbols.
- Positive and negative values.
- Unit conversions.
- Engineering specifications.

### Example test cases

| Input | Expected distinction |
|---|---|
| 10 MPa | 10 megapascals |
| 10 mPa | 10 millipascals |
| 100 °C | Temperature value |
| 1000 °C | Different temperature |
| 5 mm | Millimetres |
| 5 cm | Centimetres |

Requirements:

1. Preserve original numerical values and units.
2. Do not silently interchange units.
3. Implement a numerical consistency checker.
4. Compare generated answer values with retrieved source evidence.
5. Flag unsupported or conflicting numerical claims.
6. Preserve the source page and section for verification.

Do not assume that similar numerical tokens represent equivalent engineering measurements.

---

## 10. Benchmarking and Evaluation Module

Create a frozen evaluation dataset containing approximately 50–100 questions.

### Dataset categories

Include:

1. Basic factual questions.
2. Multi-section questions.
3. Numerical and unit-sensitive questions.
4. Table-based questions.
5. Comparison questions.
6. Unanswerable questions.
7. Technical specification questions.

Each question should contain:

- Question ID.
- Question text.
- Expected answer.
- Relevant document ID.
- Relevant page or section.
- Ground-truth relevant chunks.
- Answerability label.
- Numerical/unit sensitivity label.

### A. Retrieval Metrics

Implement:

- Recall@K.
- Mean Reciprocal Rank (MRR).
- nDCG@K.

### B. Generation Metrics

Implement:

- Answer correctness.
- Faithfulness.
- Context relevance.

### C. System Metrics

Measure:

- Total response latency.
- Retrieval latency.
- Generation latency.
- Token usage.
- Estimated execution cost.

### Experimental methodology

Establish a baseline configuration.

Change one factor at a time while keeping all other variables fixed.

For every experiment:
- Record the complete configuration.
- Record metric values.
- Calculate absolute improvement over baseline.
- Calculate relative improvement over baseline.
- Calculate confidence intervals where appropriate.
- Store experiment results.
- Maintain reproducibility.

Never fabricate evaluation results or claim improvements without experimental evidence.

---

## 11. Hallucination Measurement Module

Implement at least one automated hallucination evaluation method.

Investigate:

- RAGAS faithfulness.
- LLM-as-a-judge.
- SelfCheckGPT-style consistency checking.
- Numerical and unit consistency validation.

Compare automated evaluation results against manually labelled answers.

Record:
- Supported claims.
- Unsupported claims.
- Incorrect numerical values.
- Incorrect units.
- Missing citations.
- Answers generated without sufficient evidence.

Provide a comparison between automated and manual evaluations.

---

## 12. Frontend Design

Build a professional research dashboard using React and Tailwind CSS.

### Page 1: Document Management

Features:
- Upload engineering documents.
- Display document information.
- Show processing status.
- Display chunk count.
- Display extracted metadata.
- Allow document removal and reprocessing.

### Page 2: RAG Configuration

Provide interactive configuration controls.

**Chunking Strategy**
- Fixed-size.
- Sentence-level.
- Recursive.
- Semantic.
- Structure-aware.

**Embedding Model**
- nomic-embed-text.
- BGE-small.
- MiniLM.

**Retrieval Strategy**
- BM25.
- Dense / Semantic.
- Hybrid.
- Hybrid + Reranker.

**RAG Architecture**
- Naive RAG.
- Advanced RAG.
- Corrective RAG.
- Graph RAG.

**Additional Parameters**
- Chunk size.
- Chunk overlap.
- Top-K.
- Similarity threshold.
- Reranking toggle.
- Embedding selection.
- LLM selection.

### Page 3: Document Chat

Features:
- Engineering-related query input.
- Generated answers.
- Retrieved source chunks.
- Source document name.
- Page and section references.
- Numerical evidence.
- Insufficient-evidence warnings.
- Retrieval latency.
- Generation latency.

### Page 4: Experiment Comparison

Allow comparison of different configurations.

Display:
- Retrieval metrics.
- Generation metrics.
- Latency.
- Cost.
- Baseline versus experimental results.
- Performance charts.
- Configuration details.

Provide filters for chunking, embedding, retrieval and RAG architecture.

### Page 5: Evaluation Dashboard

Features:
- Run the frozen evaluation dataset.
- Display Recall@K, MRR and nDCG.
- Display faithfulness and correctness.
- Display latency and cost.
- Display numerical validation results.
- Export experiment results to CSV.

---

## 13. Backend Architecture

Use Python and FastAPI.

Maintain independent modules inside one integrated backend.

Suggested structure:

```text
backend/
│
├── app/
│   ├── main.py
│   ├── config.py
│   │
│   ├── ingestion/
│   │   ├── parser.py
│   │   ├── preprocessing.py
│   │   └── metadata.py
│   │
│   ├── chunking/
│   │   ├── base.py
│   │   ├── fixed.py
│   │   ├── sentence.py
│   │   ├── recursive.py
│   │   ├── semantic.py
│   │   └── structure_aware.py
│   │
│   ├── embeddings/
│   │   ├── base.py
│   │   ├── ollama_embeddings.py
│   │   └── sentence_transformer.py
│   │
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── rag/
│   │   ├── base.py
│   │   ├── naive_rag.py
│   │   ├── advanced_rag.py
│   │   ├── corrective_rag.py
│   │   └── graph_rag.py
│   │
│   ├── validation/
│   │   ├── numeric_checker.py
│   │   └── unit_checker.py
│   │
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   ├── hallucination.py
│   │   └── benchmark.py
│   │
│   ├── experiments/
│   │   ├── runner.py
│   │   └── logger.py
│   │
│   └── api/
│       ├── documents.py
│       ├── configuration.py
│       ├── chat.py
│       └── evaluation.py
│
├── tests/
├── evaluation_dataset/
├── requirements.txt
└── README.md
```

The frontend should be maintained in a separate directory within the same repository.

---

## 14. Testing Requirements

Implement unit tests for:

- All chunking strategies.
- Chunk metadata preservation.
- Embedding dimensions.
- BM25 ranking.
- Dense retrieval.
- Hybrid retrieval.
- RRF score fusion.
- Reranking.
- Retrieval metrics.
- Numerical validation.
- Unit validation.

Implement integration tests for:

- Document ingestion to indexing.
- Query to retrieval.
- Retrieval to generation.
- Corrective RAG fallback.
- End-to-end evaluation.

Include test cases for:
- Engineering tables.
- Missing units.
- Decimal values.
- Scientific notation.
- Unanswerable questions.
- Similar technical terms.
- Conflicting numerical specifications.

---

## 15. GitHub and Version Control

Maintain one integrated codebase and shared architecture.

Suggested branches:

- main: Stable integration.
- development: Ongoing integrated development.
- Feature branches: Temporary branches for individual changes when needed.

Requirements:
- No direct commits to main.
- Use pull requests.
- Review changes before merging.
- Maintain README documentation.
- Include tests with implementations.
- Avoid committing API keys, .env files, model weights and large datasets.
- Use .gitignore and environment variables.

Do not create separate disconnected implementations of the same RAG system.

---

## 16. Azure Integration

Prepare the application for Azure deployment.

Explore and practice:
- Azure compute resources.
- Azure Storage.
- Azure database or vector storage options.
- Application deployment.
- Environment variables and secrets.
- Budget alerts.
- Resource monitoring.
- Shutting down unused resources.

Keep Azure integration optional during initial local development.

The system must work locally before deployment.

---

## 17. Implementation Phases

Implement incrementally, but keep all phases integrated into the same application.

### Phase 1: Core Foundation

- Repository setup.
- Frontend and backend setup.
- Document ingestion.
- PDF parsing.
- Fixed-size and recursive chunking.
- BM25 retrieval.
- Dense retrieval.
- Qwen2.5 integration.
- Basic Naive RAG.
- Initial document chat.

### Phase 2: Modular Pipeline

- Sentence-level chunking.
- Semantic chunking.
- Structure-aware chunking.
- Multiple embedding models.
- Hybrid retrieval using RRF.
- Reranking.
- Configuration dashboard.

### Phase 3: Advanced RAG

- Advanced RAG.
- Corrective RAG.
- Evidence relevance checking.
- Numerical validation.
- Unit consistency checking.
- Source-grounded responses.

### Phase 4: Research Evaluation

- Frozen evaluation dataset v0.1.
- Recall@K.
- MRR.
- nDCG.
- Faithfulness.
- Answer correctness.
- Latency and cost.
- Experiment tracking.
- Configuration comparison dashboard.

### Phase 5: Research Extensions

- Graph RAG.
- Engineering knowledge graph.
- Additional hallucination detection methods.
- Automated versus manual evaluation comparison.
- Azure deployment.

---

## 18. Mandatory Development Rules

1. Build one integrated platform, not multiple disconnected applications.
2. Every configuration toggle must invoke a real backend implementation.
3. Implement actual algorithms, not UI placeholders.
4. Keep chunking, embedding, retrieval, reranking and generation independent.
5. Store configuration metadata for every experiment.
6. Ensure experiments are reproducible.
7. Change one experimental variable at a time for controlled comparisons.
8. Do not hardcode model responses or evaluation results.
9. Write understandable code with appropriate comments and documentation.
10. Add tests alongside every major module.
11. Handle unavailable models and insufficient evidence gracefully.
12. Preserve engineering numerical values and units.
13. Avoid unnecessary dependencies
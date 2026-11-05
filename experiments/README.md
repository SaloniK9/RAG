# Engineering RAG benchmark v0.1

This directory contains a reproducible baseline run against the frozen
`evaluation_dataset/questions_v0.1.json` dataset. The benchmark builds a fresh
index from the ten PDFs in `Engineering-RAG-Documents`; it does not read or
modify the application's live SQLite index. The PDFs themselves are not
included in this branch.

## Frozen baseline

The runner rejects model overrides that do not match this configuration:

| Setting | Baseline |
|---|---|
| Chunking | Recursive |
| Chunk size / overlap | 1,000 / 150 characters |
| Embeddings | `nomic-embed-text` |
| Retrieval | Dense cosine similarity |
| Retrieval depth | Top 5 |
| Generation | `qwen2.5:7b` through local Ollama |
| RAG architecture | Existing Naive RAG prompt and evidence format |

Ollama's generation-temperature default is not overridden. The effective model
tag, endpoint, dataset SHA-256, source-file hashes, index dimensions, and chunk
count are recorded in `baseline_results.json`.

## Dataset and relevance labels

Dataset v0.1 contains 50 questions: 10 basic factual, 10 numerical/unit-sensitive,
8 table-based, 8 multi-section, 8 comparison, and 6 unanswerable. Each answerable
question maps to one or more exact PDF pages. At run time, all generated chunks
from those pages in the named source PDF are treated as relevant. This makes the
retrieval metrics reproducible without hard-coding chunk IDs that depend on the
chunking implementation. Unanswerable questions have no relevant chunks and are
excluded from retrieval metric denominators.

Before indexing, the runner verifies every source PDF's SHA-256 and page count
against the frozen manifest, and rejects missing or additional PDFs. Dataset
references use 1-based PDF page numbers, the same page numbering carried by the
application parser.

## Metrics and interpretation

* **Recall@5**, **MRR**, and **nDCG@5** use binary relevance at the cited-page
  level. Their denominator contains the 44 answerable questions.
* **Answer correctness** is a transparent unigram lexical F1 against the
  reference answer. For the six unanswerable questions it is a binary
  abstention check. It is not a semantic or human correctness judgment.
* **Faithfulness** is the fraction of distinct answer content tokens present in
  retrieved evidence. It is a lexical-support proxy, not entailment. It is
  unavailable for unanswerable questions because those questions have no
  evidence label.
* **Numerical/unit correctness** checks that expected numeric values, signs,
  and recognized units appear in the generated answer. Hexadecimal IDs,
  common spelled-out counts, and one-/two-thirds fractions are normalized.
  It deliberately does not infer unit conversions, other equivalent notation,
  or tolerance. A missing measurement is `null`, not zero.
* **Latency** records per-question query embedding plus dense ranking,
  prompt-construction time, generation request wall time, and total request
  wall time. Index parsing/chunking/embedding time is reported separately.
  Ollama-reported model-load, prompt-evaluation, token-generation, and total
  durations are retained when present.
* **Cold/warm** is inferred from Ollama's `load_duration`: over 1 second is
  labeled cold, at most 1 second warm, and an unavailable field unknown. The
  runner does not evict or unload models to force a cold start; the first
  request can therefore be warm if Ollama already has the model loaded.
  Per-question labels and class-specific latency aggregates are retained; an
  empty class has `null` measurements rather than zeroes.
* **Token counts** come from Ollama's `prompt_eval_count` and `eval_count`.
  Local Ollama does not provide a monetary API charge, so `model_cost_usd` is
  `null`.
* **95% confidence intervals** use a deterministic, question-level percentile
  bootstrap with 2,000 resamples. These intervals describe variation across
  this fixed dataset's questions; they do not represent repeated-run,
  temperature, or hardware uncertainty.

The methods above are intentionally reported as proxies where a validated
semantic or human-judgment process is not available. Do not compare their
values to RAGAS, human annotation, or other semantic metrics as though they
were equivalent.

## Run

Requirements: the repository's existing Python environment, the ten exact
source PDFs, and a reachable local Ollama instance with `qwen2.5:7b` and
`nomic-embed-text` available.

From PowerShell at the repository root:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m benchmarks.runner `
  --dataset ..\evaluation_dataset\questions_v0.1.json `
  --source-dir ..\Engineering-RAG-Documents `
  --output ..\experiments\baseline_results.json `
  --csv ..\experiments\results_table.csv
```

The default command runs all 50 questions. `--limit N` is available only for a
smoke test and marks its output `partial`; partial output is not the baseline
result. The runner uses the existing document parser, recursive chunker, dense
retriever, and Naive RAG prompt. It does not update the live application
database.

Validate the deliverables and run the benchmark tests from the repository root:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest -q
```

## Artifacts

* `baseline_results.json` contains the exact configuration, dataset/corpus
  hashes, question-level retrieval/generation observations, latency and token
  profiles, metric summaries, and limitations.
* `results_table.csv` contains the baseline row and the requested aggregate
  metrics, with bootstrap interval columns where at least two observations
  exist.

Do not edit the frozen dataset after running a comparison. Create a new
versioned dataset for changes to questions, expected answers, labels, or source
references.

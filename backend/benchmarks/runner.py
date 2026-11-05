import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from app.chunking.recursive import RecursiveChunker
from app.config import EMBEDDING_MODEL, GENERATION_MODEL, OLLAMA_BASE_URL
from app.domain import Chunk
from app.embeddings.ollama import OllamaClient, OllamaUnavailableError
from app.ingestion.parser import DocumentParsingError, parse_document
from app.rag.naive import build_prompt
from app.retrieval.dense import DenseRetriever

from benchmarks.dataset import load_dataset
from benchmarks.metrics import (
    evaluate_generation,
    mean_reciprocal_rank,
    metric_summary,
    ndcg_at_k,
    recall_at_k,
)


TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 32
COLD_LOAD_THRESHOLD_NS = 1_000_000_000
BASELINE_CONFIG = {
    "chunking_strategy": "recursive",
    "chunk_size_characters": CHUNK_SIZE,
    "chunk_overlap_characters": CHUNK_OVERLAP,
    "embedding_model": "nomic-embed-text",
    "retrieval_strategy": "dense_cosine",
    "top_k": TOP_K,
    "generation_model": "qwen2.5:7b",
    "rag_architecture": "naive",
}


def run_benchmark(
    dataset_path: Path,
    source_dir: Path,
    output_path: Path,
    csv_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path, source_dir)
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero.")
    questions = dataset["questions"][:limit]
    if EMBEDDING_MODEL.split(":", 1)[0] != BASELINE_CONFIG["embedding_model"]:
        raise ValueError(
            "The baseline requires nomic-embed-text; check RAG_EMBEDDING_MODEL."
        )
    if GENERATION_MODEL != BASELINE_CONFIG["generation_model"]:
        raise ValueError(
            "The baseline requires qwen2.5:7b; check RAG_GENERATION_MODEL."
        )

    client = OllamaClient(
        OLLAMA_BASE_URL,
        EMBEDDING_MODEL,
        GENERATION_MODEL,
        timeout=600,
    )
    health = client.health()
    if health.get("available") is not True:
        raise OllamaUnavailableError(
            f"Ollama is unavailable: {health.get('error') or 'no health response'}"
        )
    _require_model(health.get("models", []), EMBEDDING_MODEL)
    _require_model(health.get("models", []), GENERATION_MODEL)

    indexing_started = time.perf_counter()
    chunks = _parse_and_chunk(source_dir, dataset["source_documents"])
    embeddings = _embed_chunks(client, chunks)
    indexing_latency_ms = (time.perf_counter() - indexing_started) * 1000
    if not chunks or len(chunks) != len(embeddings):
        raise RuntimeError("The baseline index has no chunks or incomplete embeddings.")

    print(
        f"Index ready: {len(chunks)} chunks, {len(embeddings[0])} dimensions; "
        f"build took {indexing_latency_ms / 1000:.1f}s.",
        flush=True,
    )
    query_results: list[dict[str, Any]] = []
    for index, question in enumerate(questions, 1):
        result = _evaluate_question(client, chunks, embeddings, question)
        query_results.append(result)
        print(
            f"[{index}/{len(questions)}] {question['id']} "
            f"retrieval={result['latency_ms']['retrieval']:.0f}ms "
            f"generation={result['latency_ms']['generation_wall']:.0f}ms "
            f"load={result['ollama_profile']['load_classification']}",
            flush=True,
        )

    results: dict[str, Any] = {
        "experiment_id": "baseline-dense-recursive-nomic-qwen25-v0.1",
        "status": "complete" if limit is None else "partial",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["version"],
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "source_corpus": {
            "directory": str(source_dir),
            "document_count": len(dataset["source_documents"]),
            "page_count": dataset["scope"]["page_count"],
            "documents": dataset["source_documents"],
        },
        "configuration": {
            **BASELINE_CONFIG,
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "ollama_base_url": OLLAMA_BASE_URL,
            "generation_temperature": "Ollama model default; not overridden",
            "cold_start_load_threshold_ns": COLD_LOAD_THRESHOLD_NS,
        },
        "index": {
            "chunk_count": len(chunks),
            "embedding_dimensions": len(embeddings[0]),
            "chunk_embedding_batch_size": EMBED_BATCH_SIZE,
            "index_build_latency_ms": round(indexing_latency_ms, 2),
        },
        "evaluation": {
            "question_count": len(questions),
            "answerable_question_count": sum(
                question["answerability"] == "answerable" for question in questions
            ),
            "unanswerable_question_count": sum(
                question["answerability"] == "unanswerable" for question in questions
            ),
            "metrics_method": {
                "retrieval": "Binary relevance by source document and cited PDF page.",
                "answer_correctness": (
                    "Unigram lexical F1 against the reference answer; unanswerable "
                    "questions score 1 only when a configured abstention phrase is used."
                ),
                "faithfulness": (
                    "Fraction of distinct answer content tokens present in retrieved "
                    "evidence; lexical support proxy, not semantic entailment."
                ),
                "numerical_unit_correctness": (
                    "Expected numeric values, signs, and recognized units must appear "
                    "in the generated answer; conversions and equivalent notation are "
                    "not inferred."
                ),
                "confidence_intervals": (
                    "Deterministic 95% non-parametric bootstrap over question-level "
                    "scores (2,000 resamples); describes dataset-question variation, "
                    "not repeated-run or hardware uncertainty."
                ),
            },
            "summary": _summarize(query_results),
            "questions": query_results,
        },
        "limitations": [
            "The correctness and faithfulness values are transparent lexical proxies, not human or independent-model judgments.",
            "Local Ollama reports token counts and generation timings but no monetary API cost; cost is null.",
            "Cold/warm labels reflect Ollama load_duration only; this runner does not evict models to force a cold start.",
            "Embedding/index-build time is reported separately from per-question retrieval and generation latency.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    _write_results_csv(csv_path, results)
    return results


def _parse_and_chunk(
    source_dir: Path,
    source_documents: list[dict[str, Any]],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunker = RecursiveChunker()
    for manifest in source_documents:
        path = source_dir / manifest["name"]
        print(f"Parsing and chunking {path.name}...", flush=True)
        content = path.read_bytes()
        document_id = hashlib.sha256(content).hexdigest()[:16]
        try:
            segments, _ = parse_document(document_id, path.name, content)
            document_chunks = chunker.chunk(segments, CHUNK_SIZE, CHUNK_OVERLAP)
            chunks.extend(document_chunks)
            print(
                f"  {len(segments)} segments, {len(document_chunks)} chunks.",
                flush=True,
            )
        except (DocumentParsingError, ValueError) as error:
            raise RuntimeError(f"Could not index {path.name}: {error}") from error
    return chunks


def _embed_chunks(client: OllamaClient, chunks: list[Chunk]) -> list[list[float]]:
    vectors: list[list[float]] = []
    batch_count = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        vectors.extend(client.embed([chunk.content for chunk in batch]))
        completed_batches = start // EMBED_BATCH_SIZE + 1
        if completed_batches % 10 == 0 or completed_batches == batch_count:
            print(
                f"Embedded {len(vectors)}/{len(chunks)} chunks "
                f"({completed_batches}/{batch_count} batches).",
                flush=True,
            )
    dimensions = {len(vector) for vector in vectors}
    if len(vectors) != len(chunks) or len(dimensions) != 1:
        raise RuntimeError("Ollama returned incomplete or inconsistent embeddings.")
    return vectors


def _evaluate_question(
    client: OllamaClient,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    question: dict[str, Any],
) -> dict[str, Any]:
    request_started = time.perf_counter()
    retrieval_started = time.perf_counter()
    query_vector = client.embed([question["question"]])[0]
    search_results = DenseRetriever.rank(
        chunks,
        TOP_K,
        [query_vector, *chunk_embeddings],
    )
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000

    relevant_ids = {
        chunk.id
        for chunk in chunks
        if chunk.document_name == question["source_document"]
        and chunk.page_number in question["source_pages"]
    }
    retrieved_ids = [result.chunk.id for result in search_results]
    evidence = "\n\n".join(result.chunk.content for result in search_results)
    prompt_construction_started = time.perf_counter()
    prompt = build_prompt(question["question"], search_results)
    prompt_construction_latency_ms = (time.perf_counter() - prompt_construction_started) * 1000

    generation_started = time.perf_counter()
    answer, ollama_metrics = _profiled_generate(client, prompt)
    generation_latency_ms = (time.perf_counter() - generation_started) * 1000
    total_latency_ms = (time.perf_counter() - request_started) * 1000

    generation_scores = evaluate_generation(
        answer,
        question["expected_answer"],
        question["answerability"],
        question["numerical_unit_sensitive"],
        evidence,
    )
    relevant_for_metrics = relevant_ids if question["answerability"] == "answerable" else set()
    return {
        "question_id": question["id"],
        "category": question["category"],
        "answerability": question["answerability"],
        "numerical_unit_sensitive": question["numerical_unit_sensitive"],
        "answer": answer,
        "retrieval": {
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": sorted(relevant_for_metrics),
            "recall_at_5": recall_at_k(retrieved_ids, relevant_for_metrics, TOP_K),
            "mrr": mean_reciprocal_rank(retrieved_ids, relevant_for_metrics),
            "ndcg_at_5": ndcg_at_k(retrieved_ids, relevant_for_metrics, TOP_K),
            "sources": [
                {
                    "chunk_id": result.chunk.id,
                    "document": result.chunk.document_name,
                    "page": result.chunk.page_number,
                    "score": result.score,
                }
                for result in search_results
            ],
        },
        "generation_metrics": generation_scores,
        "latency_ms": {
            "retrieval": retrieval_latency_ms,
            "prompt_construction": prompt_construction_latency_ms,
            "generation_wall": generation_latency_ms,
            "total_request": total_latency_ms,
        },
        "ollama_profile": ollama_metrics,
    }


def _profiled_generate(
    client: OllamaClient,
    prompt: str,
) -> tuple[str, dict[str, int | float | str | None]]:
    response = client._post(
        "/api/chat",
        {
            "model": client.generation_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )
    message = response.get("message")
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise OllamaUnavailableError("Ollama returned an empty answer.")

    raw_metrics = {
        "load_duration": response.get("load_duration"),
        "prompt_eval_duration": response.get("prompt_eval_duration"),
        "eval_duration": response.get("eval_duration"),
        "total_duration": response.get("total_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
    }
    durations = {
        key: _optional_number(raw_metrics[key])
        for key in ("load_duration", "prompt_eval_duration", "eval_duration", "total_duration")
    }
    token_counts = {
        key: _optional_number(raw_metrics[key])
        for key in ("prompt_eval_count", "eval_count")
    }
    load_duration = durations["load_duration"]
    if load_duration is None:
        load_classification = "unknown"
    elif load_duration > COLD_LOAD_THRESHOLD_NS:
        load_classification = "cold"
    else:
        load_classification = "warm"
    return answer.strip(), {
        "model_load_ms": _nanoseconds_to_ms(durations["load_duration"]),
        "prompt_evaluation_ms": _nanoseconds_to_ms(durations["prompt_eval_duration"]),
        "token_generation_ms": _nanoseconds_to_ms(durations["eval_duration"]),
        "ollama_total_ms": _nanoseconds_to_ms(durations["total_duration"]),
        "prompt_tokens": token_counts["prompt_eval_count"],
        "output_tokens": token_counts["eval_count"],
        "load_classification": load_classification,
    }


def _summarize(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval_values = {
        name: [
            result["retrieval"][name]
            for result in query_results
            if result["answerability"] == "answerable"
        ]
        for name in ("recall_at_5", "mrr", "ndcg_at_5")
    }
    generation_values = {
        name: [result["generation_metrics"][name] for result in query_results]
        for name in (
            "answer_correctness",
            "faithfulness",
            "numerical_unit_correctness",
        )
    }
    latency_names = (
        "retrieval",
        "prompt_construction",
        "generation_wall",
        "total_request",
    )
    latency_values = {
        name: [result["latency_ms"][name] for result in query_results]
        for name in latency_names
    }
    profile_fields = (
        "model_load_ms",
        "prompt_evaluation_ms",
        "token_generation_ms",
        "ollama_total_ms",
    )
    profile_values = {
        name: [result["ollama_profile"][name] for result in query_results]
        for name in profile_fields
    }
    prompt_counts = [result["ollama_profile"]["prompt_tokens"] for result in query_results]
    output_counts = [result["ollama_profile"]["output_tokens"] for result in query_results]
    classifications = [
        result["ollama_profile"]["load_classification"] for result in query_results
    ]
    latency_by_load = {
        classification: _summarize_load_class(
            [
                result
                for result in query_results
                if result["ollama_profile"]["load_classification"] == classification
            ]
        )
        for classification in ("cold", "warm", "unknown")
    }

    return {
        "retrieval": {
            name: metric_summary(values) for name, values in retrieval_values.items()
        },
        "generation": {
            name: metric_summary(values) for name, values in generation_values.items()
        },
        "latency_ms": {
            name: metric_summary(values) for name, values in latency_values.items()
        },
        "ollama_reported_latency_ms": {
            name: metric_summary(values) for name, values in profile_values.items()
        },
        "token_usage": {
            "prompt_tokens_total": _sum_if_complete(prompt_counts),
            "output_tokens_total": _sum_if_complete(output_counts),
            "prompt_tokens_measured_requests": sum(value is not None for value in prompt_counts),
            "output_tokens_measured_requests": sum(value is not None for value in output_counts),
            "request_count": len(query_results),
        },
        "load_classification": {
            "cold_requests": classifications.count("cold"),
            "warm_requests": classifications.count("warm"),
            "unknown_requests": classifications.count("unknown"),
            "cold_load_threshold_ns": COLD_LOAD_THRESHOLD_NS,
            "latency_by_classification": latency_by_load,
        },
        "model_cost_usd": None,
    }


def _summarize_load_class(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "request_count": len(query_results),
        "retrieval_latency_ms": metric_summary(
            [result["latency_ms"]["retrieval"] for result in query_results]
        ),
        "generation_latency_ms": metric_summary(
            [result["latency_ms"]["generation_wall"] for result in query_results]
        ),
        "total_latency_ms": metric_summary(
            [result["latency_ms"]["total_request"] for result in query_results]
        ),
        "model_load_duration_ms": metric_summary(
            [result["ollama_profile"]["model_load_ms"] for result in query_results]
        ),
    }


def _write_results_csv(path: Path, results: dict[str, Any]) -> None:
    summary = results["evaluation"]["summary"]
    retrieval = summary["retrieval"]
    generation = summary["generation"]
    latency = summary["latency_ms"]
    row = {
        "experiment_id": results["experiment_id"],
        "dataset_version": results["dataset_version"],
        "status": results["status"],
        "configuration": json.dumps(results["configuration"], sort_keys=True),
        "recall_at_5": _metric_mean(retrieval["recall_at_5"]),
        "recall_at_5_ci95": _json_ci(retrieval["recall_at_5"]),
        "mrr": _metric_mean(retrieval["mrr"]),
        "mrr_ci95": _json_ci(retrieval["mrr"]),
        "ndcg_at_5": _metric_mean(retrieval["ndcg_at_5"]),
        "ndcg_at_5_ci95": _json_ci(retrieval["ndcg_at_5"]),
        "answer_correctness": _metric_mean(generation["answer_correctness"]),
        "answer_correctness_ci95": _json_ci(generation["answer_correctness"]),
        "faithfulness": _metric_mean(generation["faithfulness"]),
        "faithfulness_ci95": _json_ci(generation["faithfulness"]),
        "numerical_unit_correctness": _metric_mean(
            generation["numerical_unit_correctness"]
        ),
        "retrieval_latency_ms": _metric_mean(latency["retrieval"]),
        "generation_latency_ms": _metric_mean(latency["generation_wall"]),
        "total_latency_ms": _metric_mean(latency["total_request"]),
        "prompt_tokens_total": summary["token_usage"]["prompt_tokens_total"],
        "output_tokens_total": summary["token_usage"]["output_tokens_total"],
        "model_cost_usd": summary["model_cost_usd"],
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _require_model(models: object, requested: str) -> None:
    if not isinstance(models, list):
        raise OllamaUnavailableError("Ollama health response contains no model list.")
    available = {model for model in models if isinstance(model, str)}
    compatible = requested in available or (
        ":" not in requested and f"{requested}:latest" in available
    )
    if not compatible:
        raise OllamaUnavailableError(
            f"Required baseline model '{requested}' is not available in Ollama."
        )


def _optional_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _nanoseconds_to_ms(value: int | float | None) -> float | None:
    return value / 1_000_000 if value is not None else None


def _sum_if_complete(values: list[int | float | None]) -> int | float | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _metric_mean(metric: dict[str, Any]) -> float | None:
    return metric["mean"]


def _json_ci(metric: dict[str, Any]) -> str | None:
    interval = metric["ci95"]
    return json.dumps(interval) if interval is not None else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Engineering RAG v0.1 baseline benchmark."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../evaluation_dataset/questions_v0.1.json"),
    )
    parser.add_argument("--source-dir", type=Path, default=Path("../Engineering-RAG-Documents"))
    parser.add_argument("--output", type=Path, default=Path("../experiments/baseline_results.json"))
    parser.add_argument("--csv", type=Path, default=Path("../experiments/results_table.csv"))
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N questions for a clearly marked partial smoke test.",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    results = run_benchmark(
        arguments.dataset,
        arguments.source_dir,
        arguments.output,
        arguments.csv,
        arguments.limit,
    )
    print(
        f"{results['status']} benchmark: "
        f"{results['evaluation']['question_count']} questions; "
        f"results written to {arguments.output} and {arguments.csv}"
    )


if __name__ == "__main__":
    main()

import csv
import math

from benchmarks.runner import (
    _profiled_generate,
    _require_model,
    _summarize,
    _sum_if_complete,
    _write_results_csv,
)
from app.embeddings.ollama import OllamaUnavailableError


def _result(
    question_id: str,
    answerability: str,
    recall: float | None,
    answer_correctness: float | None,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "answerability": answerability,
        "retrieval": {
            "recall_at_5": recall,
            "mrr": recall,
            "ndcg_at_5": recall,
        },
        "generation_metrics": {
            "answer_correctness": answer_correctness,
            "faithfulness": answer_correctness,
            "numerical_unit_correctness": None,
        },
        "latency_ms": {
            "retrieval": 10.0,
            "prompt_construction": 0.1,
            "generation_wall": 200.0,
            "total_request": 210.0,
        },
        "ollama_profile": {
            "model_load_ms": None,
            "prompt_evaluation_ms": 100.0,
            "token_generation_ms": 80.0,
            "ollama_total_ms": 190.0,
            "prompt_tokens": 25,
            "output_tokens": 12,
            "load_classification": "unknown",
        },
    }


def test_summary_and_csv_generation_preserve_null_metrics(tmp_path) -> None:
    query_results = [
        _result("ERAG-001", "answerable", 1.0, 0.8),
        _result("ERAG-002", "unanswerable", None, 1.0),
    ]
    summary = _summarize(query_results)  # type: ignore[arg-type]
    output = tmp_path / "results_table.csv"
    results = {
        "experiment_id": "test-baseline",
        "dataset_version": "0.1.0",
        "status": "complete",
        "configuration": {"retrieval_strategy": "dense_cosine", "top_k": 5},
        "evaluation": {"summary": summary},
    }

    _write_results_csv(output, results)
    row = next(csv.DictReader(output.open(encoding="utf-8", newline="")))

    assert math.isclose(summary["retrieval"]["recall_at_5"]["mean"], 1.0)
    assert summary["retrieval"]["recall_at_5"]["n"] == 1
    assert summary["retrieval"]["recall_at_5"]["ci95"] is None
    assert row["recall_at_5"] == "1.0"
    assert row["answer_correctness"] == "0.9"
    assert row["model_cost_usd"] == ""
    assert summary["load_classification"]["latency_by_classification"]["cold"][
        "request_count"
    ] == 0
    assert summary["load_classification"]["latency_by_classification"]["warm"][
        "generation_latency_ms"
    ]["mean"] is None


def test_profiled_generation_converts_ollama_timings_and_nulls() -> None:
    class FakeClient:
        generation_model = "qwen2.5:7b"

        def _post(self, path, payload):
            assert path == "/api/chat"
            assert payload["model"] == "qwen2.5:7b"
            return {
                "message": {"content": "Answer"},
                "load_duration": 2_000_000_000,
                "prompt_eval_duration": 3_000_000_000,
                "eval_duration": None,
                "total_duration": 5_000_000_000,
                "prompt_eval_count": 40,
            }

    answer, profile = _profiled_generate(FakeClient(), "prompt")  # type: ignore[arg-type]

    assert answer == "Answer"
    assert profile["model_load_ms"] == 2000
    assert profile["prompt_evaluation_ms"] == 3000
    assert profile["token_generation_ms"] is None
    assert profile["output_tokens"] is None
    assert profile["load_classification"] == "cold"


def test_baseline_requires_the_exact_generation_tag() -> None:
    _require_model(["qwen2.5:7b", "nomic-embed-text:latest"], "qwen2.5:7b")
    _require_model(["nomic-embed-text:latest"], "nomic-embed-text")

    try:
        _require_model(["qwen2.5:14b"], "qwen2.5:7b")
    except OllamaUnavailableError as error:
        assert "qwen2.5:7b" in str(error)
    else:
        raise AssertionError("A different Qwen tag must not satisfy the baseline.")


def test_unavailable_token_counts_make_aggregate_total_null() -> None:
    assert _sum_if_complete([20, 30]) == 50
    assert _sum_if_complete([20, None]) is None
    assert _sum_if_complete([]) is None

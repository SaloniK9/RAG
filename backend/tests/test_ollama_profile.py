from app.embeddings.ollama import GenerationResult, OllamaClient


def test_generation_profile_converts_ollama_timing_and_token_fields() -> None:
    result = GenerationResult(
        answer="OK",
        model="qwen2.5:7b",
        total_duration_ns=2_000_000_000,
        load_duration_ns=2_000_000_000,
        prompt_eval_count=30,
        prompt_eval_duration_ns=600_000_000,
        eval_count=12,
        eval_duration_ns=400_000_000,
    )

    assert result.profile == {
        "model": "qwen2.5:7b",
        "model_load_ms": 2000.0,
        "prompt_eval_ms": 600.0,
        "token_generation_ms": 400.0,
        "ollama_total_ms": 2000.0,
        "prompt_tokens": 30,
        "output_tokens": 12,
        "generation_tokens_per_second": 30.0,
        "cold_start": True,
    }


def test_generation_profile_keeps_unavailable_values_null() -> None:
    result = GenerationResult(
        answer="OK",
        model="qwen2.5:7b",
        total_duration_ns=None,
        load_duration_ns=None,
        prompt_eval_count=None,
        prompt_eval_duration_ns=None,
        eval_count=None,
        eval_duration_ns=None,
    )

    assert result.profile["model_load_ms"] is None
    assert result.profile["prompt_tokens"] is None
    assert result.profile["generation_tokens_per_second"] is None
    assert result.profile["cold_start"] is None


def test_subsecond_model_load_is_classified_as_warm() -> None:
    result = GenerationResult(
        answer="OK",
        model="qwen2.5:7b",
        total_duration_ns=1_000_000_000,
        load_duration_ns=70_000_000,
        prompt_eval_count=10,
        prompt_eval_duration_ns=100_000_000,
        eval_count=2,
        eval_duration_ns=200_000_000,
    )

    assert result.profile["model_load_ms"] == 70.0
    assert result.profile["cold_start"] is False


def test_generate_profiled_reads_real_ollama_metrics() -> None:
    client = OllamaClient(
        "http://localhost:11434",
        "nomic-embed-text",
        "qwen2.5:7b",
    )
    client._post = lambda path, payload: {
        "model": "qwen2.5:7b",
        "message": {"content": "OK."},
        "total_duration": 2_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 30,
        "prompt_eval_duration": 600_000_000,
        "eval_count": 12,
        "eval_duration": 400_000_000,
    }

    result = client.generate_profiled("Reply exactly OK.")

    assert result.answer == "OK."
    assert result.profile["prompt_tokens"] == 30
    assert result.profile["output_tokens"] == 12

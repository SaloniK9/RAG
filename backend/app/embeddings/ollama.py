import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_COLD_MODEL_LOAD_THRESHOLD_NS = 1_000_000_000


class OllamaUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_count: int | None
    prompt_eval_duration_ns: int | None
    eval_count: int | None
    eval_duration_ns: int | None

    @property
    def profile(self) -> dict[str, int | float | str | None]:
        load_seconds = _duration_seconds(self.load_duration_ns)
        prompt_seconds = _duration_seconds(self.prompt_eval_duration_ns)
        generation_seconds = _duration_seconds(self.eval_duration_ns)
        total_seconds = _duration_seconds(self.total_duration_ns)
        return {
            "model": self.model,
            "model_load_ms": _milliseconds(load_seconds),
            "prompt_eval_ms": _milliseconds(prompt_seconds),
            "token_generation_ms": _milliseconds(generation_seconds),
            "ollama_total_ms": _milliseconds(total_seconds),
            "prompt_tokens": self.prompt_eval_count,
            "output_tokens": self.eval_count,
            "generation_tokens_per_second": (
                round(self.eval_count / generation_seconds, 3)
                if self.eval_count is not None
                and generation_seconds is not None
                and generation_seconds > 0
                else None
            ),
            "cold_start": (
                self.load_duration_ns >= _COLD_MODEL_LOAD_THRESHOLD_NS
                if self.load_duration_ns is not None
                else None
            ),
        }


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        embedding_model: str,
        generation_model: str,
        timeout: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": texts},
        )
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaUnavailableError(
                "Ollama returned an invalid embedding response."
            )
        normalized: list[list[float]] = []
        expected_dimension: int | None = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise OllamaUnavailableError(
                    "Ollama returned an empty or invalid embedding vector."
                )
            values = [float(value) for value in vector]
            if expected_dimension is None:
                expected_dimension = len(values)
            elif len(values) != expected_dimension:
                raise OllamaUnavailableError(
                    "Ollama returned embeddings with inconsistent dimensions."
                )
            normalized.append(values)
        return normalized

    def generate(self, prompt: str) -> str:
        return self.generate_profiled(prompt).answer

    def generate_profiled(self, prompt: str) -> GenerationResult:
        result = self._post(
            "/api/chat",
            {
                "model": self.generation_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        message = result.get("message")
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise OllamaUnavailableError("Ollama returned an empty answer.")
        return GenerationResult(
            answer=answer.strip(),
            model=str(result.get("model", self.generation_model)),
            total_duration_ns=_optional_int(result.get("total_duration")),
            load_duration_ns=_optional_int(result.get("load_duration")),
            prompt_eval_count=_optional_int(result.get("prompt_eval_count")),
            prompt_eval_duration_ns=_optional_int(
                result.get("prompt_eval_duration")
            ),
            eval_count=_optional_int(result.get("eval_count")),
            eval_duration_ns=_optional_int(result.get("eval_duration")),
        )

    def health(self) -> dict[str, object]:
        try:
            result = self._get("/api/tags")
        except OllamaUnavailableError as error:
            return {
                "available": False,
                "models": [],
                "error": str(error),
            }
        models = result.get("models", [])
        names = [
            model.get("name", "")
            for model in models
            if isinstance(model, dict) and model.get("name")
        ]
        return {"available": True, "models": names, "error": None}

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(request)

    def _get(self, path: str) -> dict[str, object]:
        return self._request(Request(f"{self.base_url}{path}"))

    def _request(self, request: Request) -> dict[str, object]:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise OllamaUnavailableError(
                f"Ollama request failed ({error.code}): {detail or error.reason}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise OllamaUnavailableError(
                f"Could not connect to Ollama at {self.base_url}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise OllamaUnavailableError("Ollama returned invalid JSON.") from error
        if not isinstance(result, dict):
            raise OllamaUnavailableError("Ollama returned an invalid response.")
        return result


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _duration_seconds(value: int | None) -> float | None:
    return value / 1_000_000_000 if value is not None else None


def _milliseconds(value: float | None) -> float | None:
    return round(value * 1000, 3) if value is not None else None

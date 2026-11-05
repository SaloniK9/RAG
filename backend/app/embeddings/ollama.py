import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaUnavailableError(RuntimeError):
    pass


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
        return answer.strip()

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

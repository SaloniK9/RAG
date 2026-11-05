from pathlib import Path

from fastapi.testclient import TestClient

from app.embeddings.ollama import GenerationResult, OllamaUnavailableError
from app.main import create_app


class FakeOllama:
    embedding_model = "nomic-embed-text"
    generation_model = "qwen2.5:7b"

    def health(self):
        return {"available": False, "models": [], "error": "Ollama is offline"}

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def generate(self, prompt):
        assert "10 MPa" in prompt
        return "The specified pressure is 10 MPa. [S1]"

    def generate_profiled(self, prompt):
        return GenerationResult(
            answer=self.generate(prompt),
            model=self.generation_model,
            total_duration_ns=None,
            load_duration_ns=None,
            prompt_eval_count=None,
            prompt_eval_duration_ns=None,
            eval_count=None,
            eval_duration_ns=None,
        )


def test_upload_bm25_chat_and_document_deletion(tmp_path: Path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, ollama=FakeOllama()))
    upload = client.post(
        "/api/documents",
        files={"file": ("pressure.txt", b"Maximum pressure is 10 MPa.", "text/plain")},
        data={"chunking_strategy": "fixed", "chunk_size": "100", "chunk_overlap": "10"},
    )

    assert upload.status_code == 201
    document_id = upload.json()["document"]["id"]
    assert upload.json()["document"]["chunk_count"] == 1

    chunks = client.get(f"/api/documents/{document_id}/chunks")
    assert chunks.status_code == 200
    assert chunks.json()[0]["source_location"] == "document body"

    answer = client.post(
        "/api/chat",
        json={"query": "What is the maximum pressure?", "retrieval_strategy": "bm25"},
    )
    assert answer.status_code == 200
    assert answer.json()["answer"].startswith("The specified pressure")
    assert answer.json()["sources"][0]["content"] == "Maximum pressure is 10 MPa."

    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 204
    assert client.get("/api/documents").json() == []


def test_dense_chat_reports_missing_ollama(tmp_path: Path) -> None:
    class OfflineOllama(FakeOllama):
        def embed(self, texts):
            raise OllamaUnavailableError("Ollama is offline")

    client = TestClient(create_app(data_dir=tmp_path, ollama=OfflineOllama()))
    client.post(
        "/api/documents",
        files={"file": ("manual.txt", b"Rated load: 5 kN.", "text/plain")},
    )

    response = client.post(
        "/api/chat",
        json={"query": "Rated load?", "retrieval_strategy": "dense"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama is offline"


def test_dense_chat_generates_and_caches_model_embeddings(tmp_path: Path) -> None:
    class DenseOllama(FakeOllama):
        def __init__(self):
            self.embed_calls = []

        def embed(self, texts):
            self.embed_calls.append(texts)
            return [[1.0, 0.0] for _ in texts]

        def generate(self, prompt):
            return "The rated load is 5 kN. [S1]"

        def generate_profiled(self, prompt):
            return GenerationResult(
                answer=self.generate(prompt),
                model=self.generation_model,
                total_duration_ns=1_000_000_000,
                load_duration_ns=1_100_000_000,
                prompt_eval_count=20,
                prompt_eval_duration_ns=200_000_000,
                eval_count=8,
                eval_duration_ns=400_000_000,
            )

    ollama = DenseOllama()
    client = TestClient(create_app(data_dir=tmp_path, ollama=ollama))
    client.post(
        "/api/documents",
        files={"file": ("load.txt", b"Rated load: 5 kN.", "text/plain")},
    )

    first = client.post(
        "/api/chat",
        json={"query": "Rated load?", "retrieval_strategy": "dense"},
    )
    second = client.post(
        "/api/chat",
        json={"query": "Rated load?", "retrieval_strategy": "dense"},
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["sources"][0]["content"] == "Rated load: 5 kN."
    assert first.json()["models"]["embedding"] == "nomic-embed-text"
    assert first.json()["generation_profile"]["prompt_tokens"] == 20
    assert first.json()["generation_profile"]["output_tokens"] == 8
    assert first.json()["generation_profile"]["cold_start"] is True
    assert len(ollama.embed_calls) == 3

import hashlib
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.chunking.fixed import FixedSizeChunker
from app.chunking.recursive import RecursiveChunker
from app.config import (
    DATABASE_PATH,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    MAX_UPLOAD_BYTES,
    OLLAMA_BASE_URL,
    UPLOADS_DIR,
)
from app.domain import Chunk, SearchResult
from app.embeddings.ollama import OllamaClient, OllamaUnavailableError
from app.ingestion.parser import DocumentParsingError, parse_document
from app.rag.naive import build_prompt
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.storage.repository import SQLiteRepository


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    retrieval_strategy: Literal["bm25", "dense"] = "bm25"
    top_k: int = Field(default=5, ge=1, le=20)


def create_app(
    data_dir: Path | None = None,
    repository: SQLiteRepository | None = None,
    ollama: OllamaClient | None = None,
) -> FastAPI:
    resolved_data_dir = data_dir or DATABASE_PATH.parent
    repo = repository or SQLiteRepository(
        resolved_data_dir / "rag.sqlite3",
        resolved_data_dir / "uploads",
    )
    model_client = ollama or OllamaClient(
        OLLAMA_BASE_URL,
        EMBEDDING_MODEL,
        GENERATION_MODEL,
    )
    application = FastAPI(
        title="Engineering RAG",
        version="0.1.0",
        description="Phase 1 local engineering document RAG experimentation platform.",
    )
    application.state.repository = repo
    application.state.ollama = model_client
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "ollama": model_client.health(),
            "configured_models": {
                "embedding": model_client.embedding_model,
                "generation": model_client.generation_model,
            },
        }

    @application.post("/api/documents", status_code=201)
    async def upload_document(
        file: UploadFile = File(...),
        chunking_strategy: Literal["fixed", "recursive"] = Form("fixed"),
        chunk_size: int = Form(default=1000, ge=1, le=10000),
        chunk_overlap: int = Form(default=150, ge=0, le=9999),
    ) -> dict[str, object]:
        if chunk_overlap >= chunk_size:
            raise HTTPException(
                status_code=422,
                detail="chunk_overlap must be less than chunk_size.",
            )
        name = Path(file.filename or "").name
        file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
        document_id = str(uuid.uuid4())
        try:
            segments, metadata = parse_document(document_id, name, file_bytes)
            chunker = (
                FixedSizeChunker()
                if chunking_strategy == "fixed"
                else RecursiveChunker()
            )
            chunks = chunker.chunk(segments, chunk_size, chunk_overlap)
        except DocumentParsingError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not chunks:
            raise HTTPException(status_code=422, detail="No document chunks were created.")

        content_hash = hashlib.sha256(file_bytes).hexdigest()
        file_path = repo.uploads_dir / f"{document_id}{Path(name).suffix.lower()}"
        file_path.write_bytes(file_bytes)
        try:
            document = repo.add_document(
                document_id=document_id,
                name=name,
                file_path=file_path,
                metadata={**metadata, "sha256": content_hash},
                segments=segments,
                chunks=chunks,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except (OSError, sqlite3.Error):
            file_path.unlink(missing_ok=True)
            raise
        return {"document": document}

    @application.get("/api/documents")
    def list_documents() -> list[dict[str, object]]:
        return repo.list_documents()

    @application.get("/api/documents/{document_id}/chunks")
    def get_document_chunks(document_id: str) -> list[dict[str, object]]:
        if repo.get_document(document_id) is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return [
            {
                "id": chunk.id,
                "content": chunk.content,
                "page_number": chunk.page_number,
                "section_heading": chunk.section_heading,
                "source_location": chunk.source_location,
                "metadata": chunk.metadata,
            }
            for chunk in repo.get_chunks()
            if chunk.document_id == document_id
        ]

    @application.delete("/api/documents/{document_id}", status_code=204)
    def delete_document(document_id: str) -> Response:
        file_path = repo.delete_document(document_id)
        if file_path is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        Path(file_path).unlink(missing_ok=True)
        return Response(status_code=204)

    @application.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="Query cannot be blank.")
        chunks = repo.get_chunks()
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Upload an engineering document before starting a chat.",
            )

        retrieval_started = time.perf_counter()
        if request.retrieval_strategy == "bm25":
            results = BM25Retriever().retrieve(query, chunks, request.top_k)
        else:
            results = _dense_retrieve(repo, model_client, query, chunks, request.top_k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000

        generation_started = time.perf_counter()
        try:
            generation = model_client.generate_profiled(
                build_prompt(query, results)
            )
        except OllamaUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000

        return {
            "answer": generation.answer,
            "retrieval_strategy": request.retrieval_strategy,
            "models": {
                "embedding": model_client.embedding_model
                if request.retrieval_strategy == "dense"
                else None,
                "generation": model_client.generation_model,
            },
            "sources": [
                {
                    "source_id": f"S{index}",
                    "chunk_id": result.chunk.id,
                    "document_id": result.chunk.document_id,
                    "document_name": result.chunk.document_name,
                    "page_number": result.chunk.page_number,
                    "section_heading": result.chunk.section_heading,
                    "source_location": result.chunk.source_location,
                    "content": result.chunk.content,
                    "score": result.score,
                }
                for index, result in enumerate(results, start=1)
            ],
            "timings_ms": {
                "retrieval": round(retrieval_latency_ms, 2),
                "generation": round(generation_latency_ms, 2),
                "total": round(retrieval_latency_ms + generation_latency_ms, 2),
            },
            "generation_profile": generation.profile,
        }

    return application


def _dense_retrieve(
    repository: SQLiteRepository,
    ollama: OllamaClient,
    query: str,
    chunks: list[Chunk],
    top_k: int,
) -> list[SearchResult]:
    try:
        query_vector = ollama.embed([query])[0]
        cached = repository.get_chunk_embeddings(
            ollama.embedding_model,
            [chunk.id for chunk in chunks],
        )
        missing_chunks = [chunk for chunk in chunks if chunk.id not in cached]
        if missing_chunks:
            vectors = ollama.embed([chunk.content for chunk in missing_chunks])
            cached.update(
                {
                    chunk.id: vector
                    for chunk, vector in zip(missing_chunks, vectors)
                }
            )
            repository.save_chunk_embeddings(ollama.embedding_model, cached)
        embeddings = [query_vector, *(cached[chunk.id] for chunk in chunks)]
        return DenseRetriever(ollama.embed).rank(chunks, top_k, embeddings)
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"Dense retrieval failed: {error}",
        ) from error


app = create_app()

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domain import Chunk, DocumentSegment


class SQLiteRepository:
    def __init__(self, database_path: Path, uploads_dir: Path) -> None:
        self.database_path = database_path
        self.uploads_dir = uploads_dir
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    segment_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    chunking_strategy TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    document_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    section_heading TEXT,
                    source_location TEXT NOT NULL,
                    chunking_strategy TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    PRIMARY KEY (chunk_id, model)
                );
                CREATE INDEX IF NOT EXISTS chunks_document_id_idx
                    ON chunks(document_id);
                """
            )

    def add_document(
        self,
        document_id: str,
        name: str,
        file_path: Path,
        metadata: dict[str, str],
        segments: list[DocumentSegment],
        chunks: list[Chunk],
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, name, file_path, created_at, metadata_json,
                    segment_count, chunk_count, chunking_strategy,
                    chunk_size, chunk_overlap
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    name,
                    str(file_path),
                    created_at,
                    json.dumps(metadata),
                    len(segments),
                    len(chunks),
                    chunking_strategy,
                    chunk_size,
                    chunk_overlap,
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, document_name, content, page_number,
                    section_heading, source_location, chunking_strategy, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.document_name,
                        chunk.content,
                        chunk.page_number,
                        chunk.section_heading,
                        chunk.source_location,
                        chunk.chunking_strategy,
                        json.dumps(chunk.metadata),
                    )
                    for chunk in chunks
                ],
            )
        return {
            "id": document_id,
            "name": name,
            "created_at": created_at,
            "metadata": metadata,
            "segment_count": len(segments),
            "chunk_count": len(chunks),
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def get_chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks ORDER BY document_id, id"
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def get_chunk_embeddings(
        self,
        model: str,
        chunk_ids: list[str],
    ) -> dict[str, list[float]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT chunk_id, vector_json FROM chunk_embeddings "
                f"WHERE model = ? AND chunk_id IN ({placeholders})",
                [model, *chunk_ids],
            ).fetchall()
        return {row["chunk_id"]: json.loads(row["vector_json"]) for row in rows}

    def save_chunk_embeddings(
        self,
        model: str,
        embeddings: dict[str, list[float]],
    ) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings (
                    chunk_id, model, vector_json, dimensions
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (chunk_id, model, json.dumps(vector), len(vector))
                    for chunk_id, vector in embeddings.items()
                ],
            )

    def delete_document(self, document_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT file_path FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                return None
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return str(row["file_path"])

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"]),
            "segment_count": row["segment_count"],
            "chunk_count": row["chunk_count"],
            "chunking_strategy": row["chunking_strategy"],
            "chunk_size": row["chunk_size"],
            "chunk_overlap": row["chunk_overlap"],
        }

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            document_name=row["document_name"],
            content=row["content"],
            page_number=row["page_number"],
            section_heading=row["section_heading"],
            source_location=row["source_location"],
            chunking_strategy=row["chunking_strategy"],
            metadata=json.loads(row["metadata_json"]),
        )

from typing import Protocol

from app.domain import Chunk, SearchResult


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int,
    ) -> list[SearchResult]: ...

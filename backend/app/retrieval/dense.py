import math
from collections.abc import Callable

from app.domain import Chunk, SearchResult

EmbedFunction = Callable[[list[str]], list[list[float]]]


class DenseRetriever:
    def __init__(self, embed: EmbedFunction) -> None:
        self.embed = embed

    def retrieve(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")
        if not chunks:
            return []
        vectors = self.embed([query, *(chunk.content for chunk in chunks)])
        return self.rank(chunks, top_k, vectors)

    @staticmethod
    def rank(
        chunks: list[Chunk],
        top_k: int,
        vectors: list[list[float]],
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")
        if not chunks:
            return []
        if len(vectors) != len(chunks) + 1:
            raise ValueError("Embedding provider returned an unexpected number of vectors.")
        query_vector = vectors[0]
        if not query_vector or any(len(vector) != len(query_vector) for vector in vectors):
            raise ValueError("Embedding dimensions do not match.")
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            raise ValueError("Query embedding has zero magnitude.")

        scored: list[tuple[int, float]] = []
        for index, vector in enumerate(vectors[1:]):
            norm = math.sqrt(sum(value * value for value in vector))
            score = (
                sum(left * right for left, right in zip(query_vector, vector))
                / (query_norm * norm)
                if norm
                else 0.0
            )
            scored.append((index, score))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
        return [
            SearchResult(chunk=chunks[index], score=score, rank=rank)
            for rank, (index, score) in enumerate(ranked[:top_k], start=1)
        ]

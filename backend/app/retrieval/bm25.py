import re

from rank_bm25 import BM25Okapi

from app.domain import Chunk, SearchResult

_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[._/%+-][A-Za-z0-9]+)*|[^\w\s]",
    flags=re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


class BM25Retriever:
    def retrieve(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")
        query_tokens = tokenize(query)
        if not query_tokens or not chunks:
            return []
        corpus = [tokenize(chunk.content) for chunk in chunks]
        model = BM25Okapi(corpus)
        scores = model.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [
            SearchResult(chunk=chunks[index], score=float(score), rank=rank)
            for rank, (index, score) in enumerate(ranked[:top_k], start=1)
        ]

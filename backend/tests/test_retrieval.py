import math

import pytest

from app.domain import Chunk
from app.retrieval.bm25 import BM25Retriever, tokenize
from app.retrieval.dense import DenseRetriever


def _chunk(identifier: str, content: str) -> Chunk:
    return Chunk(
        id=identifier,
        document_id="doc",
        document_name="manual.pdf",
        content=content,
        source_location="page 2",
    )


def test_bm25_ranks_exact_engineering_terms_and_case_sensitive_units() -> None:
    chunks = [
        _chunk("wrong-unit", "The limit is 10 mPa."),
        _chunk("right-unit", "The limit is 10 MPa."),
        _chunk("other", "The operating temperature is 80 C."),
    ]

    results = BM25Retriever().retrieve("10 MPa", chunks, 2)

    assert results[0].chunk.id == "right-unit"
    assert "10" in tokenize("10 MPa")
    assert "MPa" in tokenize("10 MPa")


def test_dense_retrieval_ranks_by_cosine_similarity() -> None:
    vectors = {
        "query": [1.0, 0.0],
        "steel pressure rating": [0.9, 0.1],
        "pump curve": [0.0, 1.0],
    }
    retriever = DenseRetriever(lambda texts: [vectors[text] for text in texts])
    chunks = [_chunk("steel", "steel pressure rating"), _chunk("pump", "pump curve")]

    results = retriever.retrieve("query", chunks, 2)

    assert results[0].chunk.id == "steel"
    assert math.isclose(results[0].score, 0.9 / math.sqrt(0.82))


def test_dense_retrieval_rejects_mismatched_dimensions() -> None:
    retriever = DenseRetriever(lambda _: [[1.0, 0.0], [1.0]])
    with pytest.raises(ValueError, match="dimensions"):
        retriever.retrieve("query", [_chunk("a", "text")], 1)

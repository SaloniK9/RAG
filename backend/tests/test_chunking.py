from app.chunking.fixed import FixedSizeChunker
from app.chunking.recursive import RecursiveChunker
from app.domain import DocumentSegment


def _segment(text: str) -> DocumentSegment:
    return DocumentSegment(
        document_id="doc-1",
        document_name="spec.txt",
        content=text,
        page_number=4,
        section_heading="3.2 Pressure",
        source_location="page 4",
        metadata={"revision": "A"},
    )


def test_fixed_chunking_keeps_metadata_and_overlaps() -> None:
    chunks = FixedSizeChunker().chunk([_segment("abcdefghij")], 6, 2)

    assert [chunk.content for chunk in chunks] == ["abcdef", "efghij"]
    assert chunks[0].page_number == 4
    assert chunks[0].section_heading == "3.2 Pressure"
    assert chunks[0].metadata == {"revision": "A"}
    assert chunks[0].id != chunks[1].id


def test_fixed_chunking_rejects_invalid_overlap() -> None:
    try:
        FixedSizeChunker().chunk([_segment("abc")], 5, 5)
    except ValueError as error:
        assert "chunk_overlap" in str(error)
    else:
        raise AssertionError("Invalid overlap should be rejected.")


def test_recursive_chunking_prefers_paragraph_boundaries() -> None:
    chunks = RecursiveChunker().chunk(
        [_segment("First paragraph.\n\nSecond paragraph.")],
        20,
        0,
    )

    assert [chunk.content for chunk in chunks] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert all(chunk.chunking_strategy == "recursive" for chunk in chunks)


def test_recursive_chunking_applies_overlap_within_size_limit() -> None:
    chunks = RecursiveChunker().chunk(
        [_segment("First paragraph.\n\nSecond paragraph.")],
        24,
        5,
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 24 for chunk in chunks)
    assert chunks[1].content.startswith(chunks[0].content[-5:])


def test_fixed_chunking_keeps_table_rows_with_headers() -> None:
    table = DocumentSegment(
        document_id="doc-1",
        document_name="spec.txt",
        content=(
            "Table:\nParameter | Limit\n"
            "Pressure | 10 MPa\nTemperature | 80 C"
        ),
        page_number=2,
        section_heading="Limits",
        source_location="page 2, table",
        metadata={"content_type": "table"},
    )

    chunks = FixedSizeChunker().chunk([table], 45, 0)

    assert len(chunks) == 2
    assert all("Parameter | Limit" in chunk.content for chunk in chunks)
    assert "Pressure | 10 MPa" in chunks[0].content
    assert "Temperature | 80 C" in chunks[1].content
    assert chunks[1].source_location == "page 2, table"

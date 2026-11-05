from typing import Protocol

from app.domain import Chunk, DocumentSegment


class Chunker(Protocol):
    strategy: str

    def chunk(
        self,
        segments: list[DocumentSegment],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Chunk]: ...


def create_chunk(
    segment: DocumentSegment,
    text: str,
    strategy: str,
    ordinal: int,
) -> Chunk:
    import hashlib

    stable_source = "|".join(
        (
            segment.document_id,
            str(segment.page_number or ""),
            segment.section_heading or "",
            strategy,
            str(ordinal),
            text,
        )
    )
    chunk_id = hashlib.sha256(stable_source.encode("utf-8")).hexdigest()[:20]
    return Chunk(
        id=chunk_id,
        document_id=segment.document_id,
        document_name=segment.document_name,
        content=text,
        page_number=segment.page_number,
        section_heading=segment.section_heading,
        source_location=segment.source_location,
        chunking_strategy=strategy,
        metadata=segment.metadata,
    )


def chunk_table_rows(
    segment: DocumentSegment,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
    ordinal_start: int,
) -> list[Chunk]:
    rows = [line.strip() for line in segment.content.splitlines() if line.strip()]
    if not rows:
        return []

    header_count = (
        2 if rows[0].casefold().startswith("table") and len(rows) > 1 else 1
    )
    header = rows[:header_count]
    data_rows = rows[header_count:]
    header_text = "\n".join(header)
    capacity = max(1, chunk_size - len(header_text) - 1)
    groups: list[list[str]] = []
    current: list[str] = []
    for row in data_rows:
        current_size = sum(len(item) + 1 for item in current)
        if current and current_size + len(row) > capacity:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    if not groups:
        groups = [[]]

    chunks: list[Chunk] = []
    previous_rows: list[str] = []
    for index, group in enumerate(groups):
        prefix: list[str] = []
        if index and chunk_overlap:
            overlap_size = 0
            for row in reversed(previous_rows):
                if overlap_size + len(row) + 1 > chunk_overlap:
                    break
                prefix.insert(0, row)
                overlap_size += len(row) + 1
        while prefix and len("\n".join([*header, *prefix, *group])) > chunk_size:
            prefix.pop(0)
        text = "\n".join([*header, *prefix, *group])
        chunks.append(
            create_chunk(
                segment,
                text,
                strategy,
                ordinal_start + len(chunks),
            )
        )
        previous_rows = [*prefix, *group]
    return chunks

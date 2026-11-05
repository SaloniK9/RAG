from app.chunking.base import chunk_table_rows, create_chunk
from app.domain import Chunk, DocumentSegment


class FixedSizeChunker:
    strategy = "fixed"

    def chunk(
        self,
        segments: list[DocumentSegment],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Chunk]:
        _validate_sizes(chunk_size, chunk_overlap)
        chunks: list[Chunk] = []
        ordinal = 0
        step = chunk_size - chunk_overlap
        for segment in segments:
            if segment.metadata.get("content_type") == "table":
                table_chunks = chunk_table_rows(
                    segment,
                    chunk_size,
                    chunk_overlap,
                    self.strategy,
                    ordinal,
                )
                chunks.extend(table_chunks)
                ordinal += len(table_chunks)
                continue
            start = 0
            while start < len(segment.content):
                text = segment.content[start : start + chunk_size].strip()
                if text:
                    chunks.append(create_chunk(segment, text, self.strategy, ordinal))
                    ordinal += 1
                if start + chunk_size >= len(segment.content):
                    break
                start += step
        return chunks


def _validate_sizes(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1.")

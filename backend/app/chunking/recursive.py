from app.chunking.base import chunk_table_rows, create_chunk
from app.chunking.fixed import _validate_sizes
from app.domain import Chunk, DocumentSegment


class RecursiveChunker:
    strategy = "recursive"
    separators = ("\n\n", "\n", ". ", " ", "")

    def chunk(
        self,
        segments: list[DocumentSegment],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Chunk]:
        _validate_sizes(chunk_size, chunk_overlap)
        chunks: list[Chunk] = []
        ordinal = 0
        content_limit = chunk_size - chunk_overlap
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
            parts = _split_recursively(
                segment.content,
                content_limit,
                self.separators,
            )
            previous = ""
            for part in parts:
                text = part if not previous or chunk_overlap == 0 else (
                    previous[-chunk_overlap:] + part
                )
                chunks.append(create_chunk(segment, text, self.strategy, ordinal))
                ordinal += 1
                previous = text
        return chunks


def _split_recursively(text: str, limit: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= limit:
        return [text.strip()] if text.strip() else []
    if not separators:
        return [text[start : start + limit].strip() for start in range(0, len(text), limit)]

    separator = separators[0]
    if not separator:
        parts = [text[index : index + limit] for index in range(0, len(text), limit)]
    else:
        parts = text.split(separator)
        if len(parts) == 1:
            return _split_recursively(text, limit, separators[1:])
        parts = [
            part + (separator if index < len(parts) - 1 else "")
            for index, part in enumerate(parts)
        ]

    merged: list[str] = []
    current = ""
    for part in parts:
        if len(part) > limit:
            if current.strip():
                merged.append(current.strip())
                current = ""
            merged.extend(_split_recursively(part, limit, separators[1:]))
            continue
        if current and len(current) + len(part) > limit:
            merged.append(current.strip())
            current = ""
        current += part
    if current.strip():
        merged.append(current.strip())
    return merged

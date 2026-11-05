from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentSegment:
    document_id: str
    document_name: str
    content: str
    page_number: int | None = None
    section_heading: str | None = None
    source_location: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_name: str
    content: str
    page_number: int | None = None
    section_heading: str | None = None
    source_location: str = ""
    chunking_strategy: str = "fixed"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

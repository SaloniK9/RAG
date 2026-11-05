import re
from pathlib import Path
from zipfile import BadZipFile

import pymupdf
from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from app.domain import DocumentSegment


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S")


class DocumentParsingError(ValueError):
    pass


def parse_document(
    document_id: str,
    document_name: str,
    content: bytes,
) -> tuple[list[DocumentSegment], dict[str, str]]:
    extension = Path(document_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParsingError(
            f"Unsupported file type '{extension or '(none)'}'. Use PDF, DOCX, or TXT."
        )
    if not content:
        raise DocumentParsingError("The uploaded document is empty.")

    if extension == ".pdf":
        return _parse_pdf(document_id, document_name, content)
    if extension == ".docx":
        return _parse_docx(document_id, document_name, content)
    return _parse_text(document_id, document_name, content)


def _parse_pdf(
    document_id: str,
    document_name: str,
    content: bytes,
) -> tuple[list[DocumentSegment], dict[str, str]]:
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise DocumentParsingError(f"Could not read the PDF: {error}") from error

    segments: list[DocumentSegment] = []
    current_section: str | None = None
    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            tables = page.find_tables().tables
            table_bounds = [table.bbox for table in tables]
            page_items: list[tuple[float, float, str]] = []

            for block in page.get_text("blocks", sort=True):
                if len(block) < 5 or not str(block[4]).strip():
                    continue
                x0, y0, x1, y1 = block[:4]
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                if any(
                    left <= center_x <= right and top <= center_y <= bottom
                    for left, top, right, bottom in table_bounds
                ):
                    continue
                page_items.append((y0, x0, str(block[4]).strip()))

            for table in tables:
                rows = table.extract()
                table_text = "\n".join(
                    " | ".join(str(cell or "").strip() for cell in row)
                    for row in rows
                    if any(str(cell or "").strip() for cell in row)
                )
                if table_text:
                    left, top, _, _ = table.bbox
                    page_items.append((top, left, f"Table:\n{table_text}"))

            page_items.sort(key=lambda item: (item[0], item[1]))
            section_lines: list[str] = []

            def flush_section() -> None:
                text = "\n".join(section_lines).strip()
                if text:
                    segments.append(
                        DocumentSegment(
                            document_id=document_id,
                            document_name=document_name,
                            content=text,
                            page_number=page_number,
                            section_heading=current_section,
                            source_location=f"page {page_number}",
                        )
                    )
                section_lines.clear()

            for _, _, item_text in page_items:
                if item_text.startswith("Table:\n"):
                    flush_section()
                    segments.append(
                        DocumentSegment(
                            document_id=document_id,
                            document_name=document_name,
                            content=item_text,
                            page_number=page_number,
                            section_heading=current_section,
                            source_location=f"page {page_number}, table",
                            metadata={"content_type": "table"},
                        )
                    )
                    continue
                for line in item_text.splitlines():
                    normalized = line.strip()
                    if not normalized:
                        continue
                    if _is_heading(normalized):
                        flush_section()
                        current_section = normalized
                    section_lines.append(normalized)
            flush_section()
    finally:
        metadata = {
            str(key): str(value)
            for key, value in (document.metadata or {}).items()
            if value is not None
        }
        document.close()

    if not segments:
        raise DocumentParsingError(
            "No extractable text was found in the PDF. Scanned PDFs require OCR, "
            "which is not included in Phase 1."
        )
    return segments, metadata


def _parse_docx(
    document_id: str,
    document_name: str,
    content: bytes,
) -> tuple[list[DocumentSegment], dict[str, str]]:
    try:
        from io import BytesIO

        document = Document(BytesIO(content))
    except (BadZipFile, PackageNotFoundError, KeyError) as error:
        raise DocumentParsingError(f"Could not read the DOCX: {error}") from error

    segments: list[DocumentSegment] = []
    current_section: str | None = None
    paragraph_lines: list[str] = []

    def flush_paragraphs() -> None:
        text = "\n".join(paragraph_lines).strip()
        if text:
            segments.append(
                DocumentSegment(
                    document_id=document_id,
                    document_name=document_name,
                    content=text,
                    section_heading=current_section,
                    source_location="document body",
                )
            )
        paragraph_lines.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style and paragraph.style.name.startswith("Heading"):
            flush_paragraphs()
            current_section = text
        paragraph_lines.append(text)
    flush_paragraphs()

    for table_index, table in enumerate(document.tables, start=1):
        table_text = "\n".join(
            " | ".join(cell.text.strip() for cell in row.cells)
            for row in table.rows
        ).strip()
        if table_text:
            segments.append(
                DocumentSegment(
                    document_id=document_id,
                    document_name=document_name,
                    content=f"Table {table_index}:\n{table_text}",
                    section_heading=current_section,
                    source_location=f"table {table_index}",
                    metadata={"content_type": "table"},
                )
            )

    if not segments:
        raise DocumentParsingError("No extractable text was found in the DOCX.")
    properties = document.core_properties
    metadata = {
        key: str(value)
        for key, value in {
            "title": properties.title,
            "author": properties.author,
            "subject": properties.subject,
        }.items()
        if value
    }
    return segments, metadata


def _parse_text(
    document_id: str,
    document_name: str,
    content: bytes,
) -> tuple[list[DocumentSegment], dict[str, str]]:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise DocumentParsingError("TXT documents must use UTF-8 encoding.") from error
    if not text:
        raise DocumentParsingError("No extractable text was found in the TXT file.")
    return [
        DocumentSegment(
            document_id=document_id,
            document_name=document_name,
            content=text,
            source_location="document body",
        )
    ], {}


def _is_heading(text: str) -> bool:
    if len(text) > 120 or len(text.split()) > 16:
        return False
    if _NUMBERED_HEADING.match(text):
        return True
    letters = [character for character in text if character.isalpha()]
    return bool(letters) and len(letters) >= 3 and all(
        character.isupper() for character in letters
    )

from io import BytesIO

import pymupdf
from docx import Document

from app.ingestion.parser import DocumentParsingError, parse_document


def test_txt_parser_preserves_engineering_values() -> None:
    segments, metadata = parse_document(
        "d1",
        "spec.txt",
        b"Maximum pressure: 10 MPa; operating temperature: -20.5 C.",
    )

    assert len(segments) == 1
    assert "10 MPa" in segments[0].content
    assert "-20.5 C" in segments[0].content
    assert segments[0].source_location == "document body"
    assert metadata == {}


def test_docx_parser_keeps_tables_and_heading_metadata() -> None:
    document = Document()
    document.add_heading("Pressure Limits", level=1)
    document.add_paragraph("Verify the vessel rating.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Parameter"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Pressure"
    table.cell(1, 1).text = "10 MPa"
    output = BytesIO()
    document.save(output)

    segments, _ = parse_document("d2", "spec.docx", output.getvalue())

    assert segments[0].section_heading == "Pressure Limits"
    assert any("Pressure | 10 MPa" in segment.content for segment in segments)


def test_pdf_parser_preserves_page_number_and_table_rows() -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "1. Design Data")
    page.insert_text((72, 100), "Allowable pressure: 10 MPa")
    pdf_bytes = document.tobytes()
    document.close()

    segments, _ = parse_document("d3", "spec.pdf", pdf_bytes)

    assert segments
    assert all(segment.page_number == 1 for segment in segments)
    assert any("10 MPa" in segment.content for segment in segments)
    assert any(segment.section_heading == "1. Design Data" for segment in segments)


def test_parser_rejects_unsupported_and_invalid_files() -> None:
    for name, body in (("drawing.png", b"data"), ("broken.pdf", b"not a pdf")):
        try:
            parse_document("d4", name, body)
        except DocumentParsingError:
            continue
        raise AssertionError(f"{name} should not have parsed.")

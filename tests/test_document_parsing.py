from io import BytesIO

from docx import Document as DocxDocument
from openpyxl import Workbook
from pptx import Presentation
import pytest

from isthmus.document_parsing import parse_document_bytes
from isthmus.models import RelationshipType


def _bytes(document):
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_markdown_preserves_headings_and_parent_relationships():
    _, chunks, relationships = parse_document_bytes(
        b"# Policy\n\nTravel expenses are reimbursed.\n\n## Exceptions\n\nUrgent trips qualify.",
        "policy.md", "Finance",
    )
    assert chunks[-1].heading_path == ("Policy", "Exceptions")
    assert chunks[-1].source_location == "line 7"
    assert any(
        relation.source_chunk_id == chunks[2].id
        and relation.target_chunk_id == chunks[3].id
        and relation.relationship_type == RelationshipType.CHILD
        for relation in relationships
    )


def test_docx_preserves_heading_and_table_rows():
    document = DocxDocument()
    document.add_heading("Policy", level=1)
    document.add_paragraph("Approved travel only.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Limit"
    table.cell(0, 1).text = "100"
    _, chunks, _ = parse_document_bytes(_bytes(document), "policy.docx", "Finance")
    assert chunks[1].heading_path == ("Policy",)
    assert chunks[2].chunk_type == "table_row"
    assert '"Limit", "100"' in chunks[2].content


def test_presentation_and_spreadsheet_locations():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Overview"
    slide.placeholders[1].text = "Revenue increased"
    _, slides, _ = parse_document_bytes(_bytes(presentation), "report.pptx", "Finance")
    assert any("Revenue increased" in chunk.content and "slide 1" in chunk.source_location for chunk in slides)

    workbook = Workbook()
    workbook.active.title = "Budget"
    workbook.active["B2"] = 100
    _, rows, _ = parse_document_bytes(_bytes(workbook), "budget.xlsx", "Finance")
    assert rows[0].source_location == "sheet Budget, row 2"
    assert '"B2": "100"' in rows[0].content


def test_text_and_unsupported_formats():
    _, chunks, _ = parse_document_bytes(b"First paragraph.\n\nSecond paragraph.", "notes.txt", "General")
    assert len(chunks) == 2
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document_bytes(b"x", "unknown.bin", "General")

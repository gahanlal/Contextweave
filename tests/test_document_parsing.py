from io import BytesIO

from docx import Document as DocxDocument
from openpyxl import Workbook
from pptx import Presentation
import pytest

from isthmus.document_parsing import parse_document_bytes
from isthmus.index import SQLiteIndex
from isthmus.models import Chunk, Document, RelationshipType
from conftest import FakeEmbeddingProvider


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


def test_markdown_extracts_qualifies_edges_for_exception_sections():
    _, chunks, relationships = parse_document_bytes(
        b"# Policy\n\nTravel costs require manager approval.\n\n"
        b"## Exceptions\n\nExcept urgent travel, approval is unnecessary.",
        "policy.md",
        "Finance",
        max_words_per_chunk=4,
    )

    assert any(
        relation.source_chunk_id == chunks[2].id
        and relation.target_chunk_id == chunks[4].id
        and relation.relationship_type == RelationshipType.QUALIFIES
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


def test_token_aware_text_chunks_keep_provenance():
    class Provider:
        max_input_tokens = 12

        def count_tokens(self, text):
            return len(text) + 2

        def embed(self, text):
            return (1.0, 0.0)

    provider = Provider()
    _, chunks, relationships = parse_document_bytes(
        b"alpha beta gamma delta", "notes.txt", "General", embedding_provider=provider,
    )
    assert len(chunks) > 1
    assert all(provider.count_tokens(chunk.content) <= provider.max_input_tokens for chunk in chunks)
    assert " ".join(chunk.content for chunk in chunks) == "alpha beta gamma delta"
    assert all(chunk.source_location == "line 1" for chunk in chunks)
    assert len(relationships) == 2 * (len(chunks) - 1)


def test_content_revisions_coexist_and_legacy_index_is_preserved(tmp_path):
    database = tmp_path / "index.db"
    provider = FakeEmbeddingProvider()
    first = parse_document_bytes(b"Original text.", "report.txt", "Finance")
    renamed = parse_document_bytes(b"Original text.", "renamed.txt", "Finance")
    changed = parse_document_bytes(b"Updated text.", "report.txt", "Finance")
    assert first[0].id == renamed[0].id
    assert first[1][0].id == renamed[1][0].id
    assert first[0].revision_id == renamed[0].revision_id
    assert first[0].id != changed[0].id
    assert first[0].revision_id != changed[0].revision_id
    with SQLiteIndex(provider, database) as index:
        index.add_document(Document("legacy", "Old", "Finance"), [Chunk("legacy:c1", "legacy", "Finance", "old", 1)])
        for parsed in (first, renamed, changed):
            index.add_document(*parsed)
    with SQLiteIndex(provider, database) as index:
        assert set(index.documents) == {"legacy", first[0].id, changed[0].id}
        assert {index.chunks[chunk_id].content for chunk_id in index.chunks} == {
            "old", "Original text.", "Updated text."
        }
        assert index.documents[first[0].id].revision_id == first[0].revision_id
        assert index.documents[changed[0].id].revision_id == changed[0].revision_id


@pytest.mark.parametrize("filename", ["report.txt", "report.md", "report.docx", "report.pptx", "report.xlsx"])
def test_each_format_uses_bytes_for_identity(filename):
    if filename.endswith(".docx"):
        source = DocxDocument()
        source.add_paragraph("Text")
        content = _bytes(source)
    elif filename.endswith(".pptx"):
        source = Presentation()
        source.slides.add_slide(source.slide_layouts[1]).shapes.title.text = "Text"
        content = _bytes(source)
    elif filename.endswith(".xlsx"):
        source = Workbook()
        source.active["A1"] = "Text"
        content = _bytes(source)
    else:
        content = b"Text"
    first = parse_document_bytes(content, filename, "Finance")[0]
    same = parse_document_bytes(content, "renamed" + filename[6:], "Finance")[0]
    changed = parse_document_bytes(content + b"\n", filename, "Finance")[0]
    assert first.id == same.id
    assert first.revision_id == same.revision_id
    assert first.id != changed.id
    assert first.revision_id != changed.revision_id


@pytest.mark.parametrize("sector", ["", "  ", None])
def test_document_requires_nonempty_sector(sector):
    with pytest.raises(ValueError, match="sector"):
        parse_document_bytes(b"text", "Cyber-law.txt", sector)

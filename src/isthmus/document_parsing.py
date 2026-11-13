from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Iterable

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook
from pptx import Presentation

from isthmus.embeddings import EmbeddingProvider, token_limit
from isthmus.ingestion import NoExtractableTextError, _require_sector, _source_identity, _split_text, parse_pdf_bytes
from isthmus.models import Chunk, Document, Relationship, RelationshipType


@dataclass(frozen=True, slots=True)
class _Block:
    content: str
    location: str
    kind: str
    headings: tuple[str, ...] = ()
    heading: bool = False


def parse_document_bytes(
    content: bytes,
    filename: str,
    sector_id: str,
    max_words_per_chunk: int = 300,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> tuple[Document, tuple[Chunk, ...], tuple[Relationship, ...]]:
    path = Path(filename)
    suffix = path.suffix.casefold()
    if max_words_per_chunk < 1:
        raise ValueError("max_words_per_chunk must be positive")
    _require_sector(sector_id)
    if suffix == ".pdf":
        return parse_pdf_bytes(content, filename, sector_id, max_words_per_chunk, embedding_provider=embedding_provider)
    if suffix in (".txt", ".md"):
        text = content.decode("utf-8-sig")
        blocks = _text_blocks(text, markdown=suffix == ".md")
    elif suffix == ".docx":
        blocks = _docx_blocks(content)
    elif suffix == ".pptx":
        blocks = _pptx_blocks(content)
    elif suffix == ".xlsx":
        blocks = _xlsx_blocks(content)
    else:
        raise ValueError(f"Unsupported document format: {suffix or '(none)'}")
    document_id, revision_id = _source_identity(BytesIO(content))
    document = Document(document_id, path.stem, sector_id, revision_id)
    chunks: list[Chunk] = []
    relationships: list[Relationship] = []
    headings: dict[tuple[str, ...], str] = {}
    budget = token_limit(embedding_provider) if embedding_provider is not None else None
    for block in blocks:
        for part in _split_text(block.content, max_words_per_chunk, budget):
            number = len(chunks) + 1
            chunk = Chunk(
                id=f"{document_id}:c{number}",
                document_id=document_id,
                sector_id=sector_id,
                content=part,
                sequence_number=number,
                section_id=" / ".join(block.headings),
                heading_path=block.headings,
                source_location=block.location,
                chunk_type=block.kind,
                content_hash=hashlib.sha256(part.encode("utf-8")).hexdigest(),
            )
            parent_path = block.headings[:-1] if block.heading else block.headings
            parent_id = headings.get(parent_path)
            if parent_id:
                relationships.extend((
                    Relationship(parent_id, chunk.id, RelationshipType.CHILD),
                    Relationship(chunk.id, parent_id, RelationshipType.PARENT),
                ))
            if block.heading:
                headings[block.headings] = chunk.id
            if chunks:
                relationships.extend((
                    Relationship(chunks[-1].id, chunk.id, RelationshipType.NEXT),
                    Relationship(chunk.id, chunks[-1].id, RelationshipType.PREVIOUS),
                ))
            chunks.append(chunk)
    if not chunks:
        raise NoExtractableTextError(f"{path.name} contains no extractable text.")
    return document, tuple(chunks), tuple(relationships)


def _text_blocks(text: str, *, markdown: bool) -> Iterable[_Block]:
    headings: list[str] = []
    paragraph: list[str] = []
    start = 1
    for number, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if markdown else None
        if match or not line.strip():
            if paragraph:
                yield _Block("\n".join(paragraph), f"line {start}", "paragraph", tuple(headings))
                paragraph = []
            if match:
                level, title = len(match.group(1)), match.group(2)
                headings = headings[:level - 1]
                headings.append(title)
                yield _Block(title, f"line {number}", "heading", tuple(headings), True)
        else:
            if not paragraph:
                start = number
            paragraph.append(line)
    if paragraph:
        yield _Block("\n".join(paragraph), f"line {start}", "paragraph", tuple(headings))


def _docx_blocks(content: bytes) -> Iterable[_Block]:
    document = DocxDocument(BytesIO(content))
    headings: list[str] = []
    for position, item in enumerate(document.iter_inner_content(), start=1):
        location = f"document block {position}"
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            match = re.match(r"Heading (\d+)$", item.style.name or "", re.IGNORECASE)
            if match:
                level = int(match.group(1))
                headings = headings[:level - 1]
                headings.append(text)
                yield _Block(text, location, "heading", tuple(headings), True)
            else:
                yield _Block(text, location, "paragraph", tuple(headings))
        elif isinstance(item, Table):
            for row_number, row in enumerate(item.rows, start=1):
                values = [cell.text for cell in row.cells]
                yield _Block(json.dumps(values, ensure_ascii=False), f"{location}, row {row_number}", "table_row", tuple(headings))


def _pptx_blocks(content: bytes) -> Iterable[_Block]:
    presentation = Presentation(BytesIO(content))
    for slide_number, slide in enumerate(presentation.slides, start=1):
        headings = (slide.shapes.title.text.strip(),) if slide.shapes.title and slide.shapes.title.text.strip() else ()
        for shape_number, shape in enumerate(slide.shapes, start=1):
            location = f"slide {slide_number}, shape {shape_number}"
            if shape.has_table:
                for row_number, row in enumerate(shape.table.rows, start=1):
                    yield _Block(json.dumps([cell.text for cell in row.cells], ensure_ascii=False), f"{location}, row {row_number}", "table_row", headings)
            elif shape.has_text_frame and shape.text.strip():
                yield _Block(shape.text, location, "slide", headings)


def _xlsx_blocks(content: bytes) -> Iterable[_Block]:
    workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            for row_number, row in enumerate(sheet.iter_rows(), start=1):
                values = {cell.coordinate: str(cell.value) for cell in row if cell.value is not None}
                if values:
                    yield _Block(json.dumps(values, ensure_ascii=False), f"sheet {sheet.title}, row {row_number}", "spreadsheet_row", (sheet.title,))
    finally:
        workbook.close()

from __future__ import annotations

import hashlib
from io import BytesIO
import re
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from contextweave.models import (
    Chunk,
    Document,
    Relationship,
    RelationshipType,
)


class NoExtractableTextError(ValueError):
    pass


def sector_for_pdf(path: str | Path) -> str:
    name = Path(path).stem.lower()
    if any(term in name for term in ("cyber", "crime", "law")):
        return "Cybersecurity & Law"
    if any(term in name for term in ("finance", "financial", "economic", "agriculture")):
        return "Finance"
    if "sustainable" in name:
        return "Sustainability"
    return "General"


def parse_pdf(
    path: str | Path,
    sector_id: str | None = None,
    max_words_per_chunk: int = 300,
) -> tuple[Document, tuple[Chunk, ...], tuple[Relationship, ...]]:
    pdf_path = Path(path)
    with pdf_path.open("rb") as stream:
        return _parse_pdf_stream(stream, pdf_path, sector_id, max_words_per_chunk)


def parse_pdf_bytes(
    content: bytes,
    filename: str,
    sector_id: str,
    max_words_per_chunk: int = 300,
) -> tuple[Document, tuple[Chunk, ...], tuple[Relationship, ...]]:
    return _parse_pdf_stream(
        BytesIO(content),
        Path(filename),
        sector_id,
        max_words_per_chunk,
    )


def _parse_pdf_stream(
    stream: BinaryIO,
    pdf_path: Path,
    sector_id: str | None,
    max_words_per_chunk: int,
) -> tuple[Document, tuple[Chunk, ...], tuple[Relationship, ...]]:
    if max_words_per_chunk < 1:
        raise ValueError("max_words_per_chunk must be positive")

    document_id = hashlib.sha256(pdf_path.name.casefold().encode("utf-8")).hexdigest()[:20]
    document = Document(
        id=document_id,
        title=pdf_path.stem,
        sector_id=sector_id or sector_for_pdf(pdf_path),
    )
    chunks: list[Chunk] = []
    sequence_number = 0
    reader = PdfReader(stream)
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for content in _split_text(page_text, max_words_per_chunk):
            sequence_number += 1
            chunks.append(
                Chunk(
                    id=f"{document_id}:p{page_number}:c{sequence_number}",
                    document_id=document.id,
                    sector_id=document.sector_id,
                    content=content,
                    sequence_number=sequence_number,
                    section_id=f"page-{page_number}",
                    heading_path=(pdf_path.stem,),
                    source_location=f"page {page_number}",
                    chunk_type="pdf_page",
                )
            )

    if not chunks:
        raise NoExtractableTextError(
            f"{pdf_path.name} contains no extractable text; scanned PDFs need OCR."
        )

    relationships: list[Relationship] = []
    for previous, following in zip(chunks, chunks[1:]):
        relationships.extend(
            (
                Relationship(
                    previous.id,
                    following.id,
                    RelationshipType.NEXT,
                ),
                Relationship(
                    following.id,
                    previous.id,
                    RelationshipType.PREVIOUS,
                ),
            )
        )
    return document, tuple(chunks), tuple(relationships)


def _split_text(text: str, max_words: int) -> tuple[str, ...]:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    result: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            result.append(" ".join(current))
            current = []
            current_words = 0

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            words = sentence.split()
            while words:
                available = max_words - current_words
                if available == 0:
                    flush()
                    available = max_words
                part = words[:available]
                words = words[available:]
                current.append(" ".join(part))
                current_words += len(part)
                if words:
                    flush()
        if current_words >= max_words:
            flush()
    flush()
    return tuple(result)

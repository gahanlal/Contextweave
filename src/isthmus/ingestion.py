from __future__ import annotations

import hashlib
from io import BytesIO
import re
from pathlib import Path
from collections.abc import Callable
from typing import BinaryIO

from pypdf import PdfReader

from isthmus.embeddings import EmbeddingProvider, count_tokens, token_limit
from isthmus.models import (
    Chunk,
    Document,
    Relationship,
    RelationshipType,
)


class NoExtractableTextError(ValueError):
    pass


def _require_sector(sector_id: str) -> None:
    if not isinstance(sector_id, str) or not sector_id.strip():
        raise ValueError("An explicit non-empty sector is required.")


def _source_identity(stream: BinaryIO) -> tuple[str, str]:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    revision = digest.hexdigest()
    return f"sha256-{revision}", revision


def parse_pdf(
    path: str | Path,
    sector_id: str,
    max_words_per_chunk: int = 300,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> tuple[Document, tuple[Chunk, ...], tuple[Relationship, ...]]:
    pdf_path = Path(path)
    with pdf_path.open("rb") as stream:
        return _parse_pdf_stream(stream, pdf_path, sector_id, max_words_per_chunk, embedding_provider)


def parse_pdf_bytes(
    content: bytes,
    filename: str,
    sector_id: str,
    max_words_per_chunk: int = 300,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> tuple[Document, tuple[Chunk, ...], tuple[Relationship, ...]]:
    return _parse_pdf_stream(
        BytesIO(content),
        Path(filename),
        sector_id,
        max_words_per_chunk,
        embedding_provider,
    )


def _parse_pdf_stream(
    stream: BinaryIO,
    pdf_path: Path,
    sector_id: str,
    max_words_per_chunk: int,
    embedding_provider: EmbeddingProvider | None,
) -> tuple[Document, tuple[Chunk, ...], tuple[Relationship, ...]]:
    if max_words_per_chunk < 1:
        raise ValueError("max_words_per_chunk must be positive")
    _require_sector(sector_id)

    document_id, revision_id = _source_identity(stream)
    document = Document(
        id=document_id,
        title=pdf_path.stem,
        sector_id=sector_id,
        revision_id=revision_id,
    )
    chunks: list[Chunk] = []
    sequence_number = 0
    budget = token_limit(embedding_provider) if embedding_provider is not None else None
    reader = PdfReader(stream)
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for content in _split_text(page_text, max_words_per_chunk, budget):
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


def _split_text(
    text: str,
    max_words: int,
    budget: tuple[Callable[[str], int], int] | None = None,
) -> tuple[str, ...]:
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
    if budget is None:
        return tuple(result)
    counter, maximum = budget
    return tuple(
        piece
        for part in result
        for piece in _split_to_token_limit(part, counter, maximum)
    )


def _split_to_token_limit(text: str, counter: Callable[[str], int], maximum: int) -> tuple[str, ...]:
    result = []
    remaining = text
    while remaining:
        if count_tokens(counter, remaining) <= maximum:
            result.append(remaining)
            break
        ends = [match.end() for match in re.finditer(r"\S+", remaining)]
        low, high, end = 0, len(ends) - 1, 0
        while low <= high:
            middle = (low + high) // 2
            candidate = ends[middle]
            if count_tokens(counter, remaining[:candidate]) <= maximum:
                end = candidate
                low = middle + 1
            else:
                high = middle - 1
        if end == 0:
            low, high = 1, ends[0]
            while low <= high:
                middle = (low + high) // 2
                if count_tokens(counter, remaining[:middle]) <= maximum:
                    end = middle
                    low = middle + 1
                else:
                    high = middle - 1
        if end == 0:
            raise ValueError("The embedding token limit is too small to encode a character.")
        result.append(remaining[:end].strip())
        remaining = remaining[end:].lstrip()
    return tuple(result)

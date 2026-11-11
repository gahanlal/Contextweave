from __future__ import annotations

from collections.abc import Iterable

from contextweave.embeddings import EmbeddingProvider, cosine_similarity
from contextweave.models import Chunk, Document, Relationship, SearchHit


class InMemoryIndex:
    """Small vector index used by the local POC and unit tests."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.relationships: list[Relationship] = []

    def add_document(self, document: Document, chunks: Iterable[Chunk]) -> None:
        self.documents[document.id] = document
        for chunk in chunks:
            if chunk.document_id != document.id:
                raise ValueError(f"chunk {chunk.id} belongs to a different document")
            if chunk.sector_id != document.sector_id:
                raise ValueError(f"chunk {chunk.id} belongs to a different sector")
            if not chunk.embedding:
                chunk = Chunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    sector_id=chunk.sector_id,
                    content=chunk.content,
                    sequence_number=chunk.sequence_number,
                    embedding=self.embedding_provider.embed(chunk.content),
                    section_id=chunk.section_id,
                    heading_path=chunk.heading_path,
                    source_location=chunk.source_location,
                    chunk_type=chunk.chunk_type,
                    token_count=chunk.token_count,
                    content_hash=chunk.content_hash,
                )
            self.chunks[chunk.id] = chunk

    def add_relationship(self, relationship: Relationship) -> None:
        source = self.chunks.get(relationship.source_chunk_id)
        target = self.chunks.get(relationship.target_chunk_id)
        if source is None or target is None:
            raise ValueError("relationship endpoints must be indexed chunks")
        if source.document_id != target.document_id:
            raise ValueError("relationships cannot cross document boundaries")
        self.relationships.append(relationship)

    def search(self, query: str, limit: int = 30) -> tuple[SearchHit, ...]:
        query_vector = self.embedding_provider.embed(query)
        ranked = sorted(
            (
                (cosine_similarity(query_vector, chunk.embedding), chunk)
                for chunk in self.chunks.values()
            ),
            key=lambda item: (-item[0], item[1].id),
        )[:limit]
        return tuple(
            SearchHit(chunk=chunk, similarity=score, rank=rank)
            for rank, (score, chunk) in enumerate(ranked, start=1)
        )

    def similarity(self, query: str, chunk: Chunk) -> float:
        return cosine_similarity(
            self.embedding_provider.embed(query),
            chunk.embedding,
        )

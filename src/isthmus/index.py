from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from heapq import nsmallest
import json
from pathlib import Path
import sqlite3
import struct
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from isthmus.embeddings import (
    EmbeddingProvider,
    embed_many,
    normalize_vector,
    normalized_cosine_similarity,
)
from isthmus.models import (
    Chunk,
    Document,
    Relationship,
    RelationshipType,
    SearchHit,
)


class InMemoryIndex:
    """Small vector index used by the local POC and unit tests."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.relationships: list[Relationship] = []
        self.sectors: set[str] = set()
        self.version = 0

    def add_document(
        self,
        document: Document,
        chunks: Iterable[Chunk],
        relationships: Iterable[Relationship] = (),
    ) -> None:
        chunk_list = list(chunks)
        edges = tuple(relationships)
        for chunk in chunk_list:
            if chunk.document_id != document.id:
                raise ValueError(f"chunk {chunk.id} belongs to a different document")
            if chunk.sector_id != document.sector_id:
                raise ValueError(f"chunk {chunk.id} belongs to a different sector")
        chunk_ids = {chunk.id for chunk in chunk_list}
        if any(edge.source_chunk_id not in chunk_ids or edge.target_chunk_id not in chunk_ids for edge in edges):
            raise ValueError("relationship endpoints must belong to the indexed document")
        missing_embeddings = [chunk for chunk in chunk_list if not chunk.embedding]
        generated_embeddings = iter(
            embed_many(
                self.embedding_provider,
                [chunk.content for chunk in missing_embeddings],
            )
        )
        prepared_chunks = []
        for chunk in chunk_list:
            embedding = chunk.embedding or next(generated_embeddings)
            normalized = normalize_vector(embedding)
            if normalized != chunk.embedding:
                chunk = replace(chunk, embedding=normalized)
            prepared_chunks.append(chunk)
        old_ids = {
            chunk_id for chunk_id, chunk in self.chunks.items()
            if chunk.document_id == document.id
        }
        for chunk_id in old_ids:
            del self.chunks[chunk_id]
        if old_ids:
            self.relationships = [
                edge for edge in self.relationships
                if edge.source_chunk_id not in old_ids and edge.target_chunk_id not in old_ids
            ]
        self.documents[document.id] = document
        self.sectors.add(document.sector_id)
        for chunk in prepared_chunks:
            self.chunks[chunk.id] = chunk
        self.relationships.extend(edges)
        self.version += 1

    def add_relationship(self, relationship: Relationship) -> None:
        self.add_relationships((relationship,))

    def add_relationships(self, relationships: Iterable[Relationship]) -> None:
        edges = tuple(relationships)
        for relationship in edges:
            source = self.chunks.get(relationship.source_chunk_id)
            target = self.chunks.get(relationship.target_chunk_id)
            if source is None or target is None:
                raise ValueError("relationship endpoints must be indexed chunks")
            if source.document_id != target.document_id:
                raise ValueError("relationships cannot cross document boundaries")
        self.relationships.extend(edges)
        if edges:
            self.version += 1

    def search(self, query: str, limit: int = 30) -> tuple[SearchHit, ...]:
        return self.search_by_vector(self.embed_query(query), limit)

    def embed_query(self, query: str) -> tuple[float, ...]:
        return normalize_vector(self.embedding_provider.embed(query))

    def search_by_vector(
        self,
        query_vector: tuple[float, ...],
        limit: int = 30,
    ) -> tuple[SearchHit, ...]:
        if limit <= 0:
            return ()
        unit_query_vector = normalize_vector(query_vector)
        scored = (
            (normalized_cosine_similarity(unit_query_vector, chunk.embedding), chunk)
            for chunk in self.chunks.values()
        )
        ranked = nsmallest(limit, scored, key=lambda item: (-item[0], item[1].id))
        return tuple(
            SearchHit(chunk=chunk, similarity=score, rank=rank)
            for rank, (score, chunk) in enumerate(ranked, start=1)
        )

    def similarity(self, query: str, chunk: Chunk) -> float:
        return self.similarity_to_vector(self.embed_query(query), chunk)

    @staticmethod
    def similarity_to_vector(
        query_vector: tuple[float, ...],
        chunk: Chunk,
    ) -> float:
        return normalized_cosine_similarity(query_vector, chunk.embedding)


class BackendIndex(InMemoryIndex):
    """Base interface for a managed, replaceable index backend."""

    provider_id: str

    def add_sector(self, sector_id: str) -> None:
        raise NotImplementedError

    def remove_sector(self, sector_id: str) -> None:
        raise NotImplementedError

    def remove_document(self, document_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> BackendIndex:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


class SQLiteIndex(BackendIndex):
    """Persist indexed content in SQLite and keep vectors cached for local search."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        database_path: str | Path = "isthmus.db",
    ) -> None:
        super().__init__(embedding_provider)
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self._vector_client = None
        self._vector_collection = None
        self._vector_index_version = -1
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.provider_id = getattr(
            embedding_provider,
            "provider_id",
            f"{type(embedding_provider).__module__}.{type(embedding_provider).__qualname__}",
        )
        self._create_schema()
        try:
            self._validate_provider()
        except ValueError:
            self.connection.close()
            raise
        self._load_cache()

    def search_by_vector(
        self,
        query_vector: tuple[float, ...],
        limit: int = 30,
    ) -> tuple[SearchHit, ...]:
        if limit <= 0 or not self.chunks:
            return ()
        unit_query_vector = normalize_vector(query_vector)
        if not unit_query_vector:
            raise ValueError("Query embedding cannot be empty.")
        if self._vector_index_version != self.version:
            if self._vector_client is None:
                self._vector_client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False)
                )
            if self._vector_collection is not None:
                self._vector_client.delete_collection(self._vector_collection.name)
            self._vector_collection = self._vector_client.create_collection(
                name=f"isthmus-{uuid4().hex}",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
            chunks = list(self.chunks.values())
            batch_size = self._vector_client.get_max_batch_size()
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                self._vector_collection.add(
                    ids=[chunk.id for chunk in batch],
                    embeddings=[list(chunk.embedding) for chunk in batch],
                )
            self._vector_index_version = self.version
        result = self._vector_collection.query(
            query_embeddings=[list(unit_query_vector)],
            n_results=min(limit, len(self.chunks)),
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"]
        if distances is None:
            raise RuntimeError("ChromaDB returned no similarity distances.")
        ranked = sorted(
            ((max(0.0, min(1.0, 1.0 - distance / 2.0)), self.chunks[chunk_id])
             for chunk_id, distance in zip(ids, distances[0])),
            key=lambda item: (-item[0], item[1].id),
        )
        return tuple(
            SearchHit(chunk=chunk, similarity=score, rank=rank)
            for rank, (score, chunk) in enumerate(ranked, start=1)
        )

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sectors (
                id TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                sector_id TEXT NOT NULL REFERENCES sectors(id),
                revision_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                sector_id TEXT NOT NULL,
                content TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                section_id TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                source_location TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                content_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relationships (
                source_chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                target_chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                relationship_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT NOT NULL,
                PRIMARY KEY (source_chunk_id, target_chunk_id, relationship_type)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document_sequence
                ON chunks(document_id, sequence_number);
            """
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO sectors (id) "
            "SELECT DISTINCT sector_id FROM documents"
        )

    def _validate_provider(self) -> None:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key = 'embedding_provider'"
        ).fetchone()
        if row is not None:
            if row["value"] != self.provider_id:
                has_vectors = self.connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM chunks LIMIT 1)"
                ).fetchone()[0]
                if not has_vectors:
                    with self.connection:
                        self.connection.execute(
                            "UPDATE index_metadata SET value = ? WHERE key = 'embedding_provider'",
                            (self.provider_id,),
                        )
                    return
                self.connection.close()
                raise ValueError(
                    f"Database embeddings use {row['value']}; selected provider is "
                    f"{self.provider_id}. Use a separate database or the original provider."
                )
            return
        has_vectors = self.connection.execute(
            "SELECT EXISTS(SELECT 1 FROM chunks LIMIT 1)"
        ).fetchone()[0]
        if has_vectors:
            self.connection.close()
            raise ValueError(
                "This database has embeddings but no provider metadata. Re-index it "
                "with the intended embedding provider before querying."
            )
        with self.connection:
            self.connection.execute(
                "INSERT INTO index_metadata (key, value) "
                "VALUES ('embedding_provider', ?)",
                (self.provider_id,),
            )

    def _load_cache(self) -> None:
        self.sectors.update(
            row["id"] for row in self.connection.execute("SELECT id FROM sectors")
        )
        document_rows = self.connection.execute(
            "SELECT id, title, sector_id, revision_id FROM documents ORDER BY id"
        )
        chunk_rows = iter(
            self.connection.execute(
                "SELECT * FROM chunks ORDER BY document_id, sequence_number"
            )
        )
        pending_chunk = next(chunk_rows, None)
        for row in document_rows:
            document = Document(
                id=row["id"],
                title=row["title"],
                sector_id=row["sector_id"],
                revision_id=row["revision_id"],
            )
            chunks = []
            while (
                pending_chunk is not None
                and pending_chunk["document_id"] == document.id
            ):
                chunks.append(self._chunk_from_row(pending_chunk))
                pending_chunk = next(chunk_rows, None)
            super().add_document(document, chunks)

        super().add_relationships(
            (
                Relationship(
                    source_chunk_id=row["source_chunk_id"],
                    target_chunk_id=row["target_chunk_id"],
                    relationship_type=RelationshipType(row["relationship_type"]),
                    confidence=row["confidence"],
                    metadata=json.loads(row["metadata"]),
                )
                for row in self.connection.execute("SELECT * FROM relationships")
            )
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> Chunk:
        stored_embedding = row["embedding"]
        if isinstance(stored_embedding, str):
            embedding = tuple(json.loads(stored_embedding))
        else:
            if len(stored_embedding) % 8:
                raise ValueError(f"stored chunk {row['id']} has an invalid embedding")
            dimensions = len(stored_embedding) // 8
            embedding = struct.unpack(f"!{dimensions}d", stored_embedding)
        if not embedding:
            raise ValueError(f"stored chunk {row['id']} has no embedding")
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            sector_id=row["sector_id"],
            content=row["content"],
            sequence_number=row["sequence_number"],
            embedding=embedding,
            section_id=row["section_id"],
            heading_path=tuple(json.loads(row["heading_path"])),
            source_location=row["source_location"],
            chunk_type=row["chunk_type"],
            token_count=row["token_count"],
            content_hash=row["content_hash"],
        )

    def add_document(
        self,
        document: Document,
        chunks: Iterable[Chunk],
        relationships: Iterable[Relationship] = (),
    ) -> None:
        chunk_list = list(chunks)
        edges = tuple(relationships)
        for chunk in chunk_list:
            if chunk.document_id != document.id:
                raise ValueError(f"chunk {chunk.id} belongs to a different document")
            if chunk.sector_id != document.sector_id:
                raise ValueError(f"chunk {chunk.id} belongs to a different sector")
        chunk_ids = {chunk.id for chunk in chunk_list}
        if any(edge.source_chunk_id not in chunk_ids or edge.target_chunk_id not in chunk_ids for edge in edges):
            raise ValueError("relationship endpoints must belong to the indexed document")
        missing = [chunk for chunk in chunk_list if not chunk.embedding]
        generated = iter(
            embed_many(self.embedding_provider, [chunk.content for chunk in missing])
        )
        stored_chunks = []
        for chunk in chunk_list:
            normalized = normalize_vector(chunk.embedding or next(generated))
            stored_chunks.append(
                replace(chunk, embedding=normalized)
                if normalized != chunk.embedding
                else chunk
            )

        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO sectors (id) VALUES (?)",
                (document.sector_id,),
            )
            self.connection.execute(
                "DELETE FROM documents WHERE id = ?", (document.id,)
            )
            self.connection.execute(
                "INSERT INTO documents (id, title, sector_id, revision_id) "
                "VALUES (?, ?, ?, ?)",
                (document.id, document.title, document.sector_id, document.revision_id),
            )
            self.connection.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, sector_id, content, sequence_number, embedding,
                    section_id, heading_path, source_location, chunk_type, token_count,
                    content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.sector_id,
                        chunk.content,
                        chunk.sequence_number,
                        struct.pack(f"!{len(chunk.embedding)}d", *chunk.embedding),
                        chunk.section_id,
                        json.dumps(chunk.heading_path),
                        chunk.source_location,
                        chunk.chunk_type,
                        chunk.token_count,
                        chunk.content_hash,
                    )
                    for chunk in stored_chunks
                ],
            )
            self.connection.executemany(
                "INSERT INTO relationships (source_chunk_id, target_chunk_id, "
                "relationship_type, confidence, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    (edge.source_chunk_id, edge.target_chunk_id, edge.relationship_type.value,
                     edge.confidence, json.dumps(edge.metadata))
                    for edge in edges
                ),
            )

        old_ids = {
            chunk_id
            for chunk_id, chunk in self.chunks.items()
            if chunk.document_id == document.id
        }
        for chunk_id in old_ids:
            del self.chunks[chunk_id]
        if old_ids:
            self.relationships = [
                relationship
                for relationship in self.relationships
                if relationship.source_chunk_id not in old_ids
                and relationship.target_chunk_id not in old_ids
            ]
        self.documents[document.id] = document
        self.sectors.add(document.sector_id)
        self.chunks.update((chunk.id, chunk) for chunk in stored_chunks)
        self.relationships.extend(edges)
        self.version += 1

    def add_relationship(self, relationship: Relationship) -> None:
        self.add_relationships((relationship,))

    def add_relationships(self, relationships: Iterable[Relationship]) -> None:
        edges = tuple(relationships)
        for relationship in edges:
            source = self.chunks.get(relationship.source_chunk_id)
            target = self.chunks.get(relationship.target_chunk_id)
            if source is None or target is None:
                raise ValueError("relationship endpoints must be indexed chunks")
            if source.document_id != target.document_id:
                raise ValueError("relationships cannot cross document boundaries")
        if not edges:
            return
        with self.connection:
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO relationships (
                    source_chunk_id, target_chunk_id, relationship_type, confidence,
                    metadata
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        relationship.source_chunk_id,
                        relationship.target_chunk_id,
                        relationship.relationship_type.value,
                        relationship.confidence,
                        json.dumps(relationship.metadata),
                    )
                    for relationship in edges
                ),
            )
        self.relationships.extend(edges)
        self.version += 1

    def add_sector(self, sector_id: str) -> None:
        normalized_id = sector_id.strip()
        if not normalized_id:
            raise ValueError("sector name cannot be empty")
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO sectors (id) VALUES (?)",
                (normalized_id,),
            )
        self.sectors.add(normalized_id)

    def remove_document(self, document_id: str) -> None:
        if document_id not in self.documents:
            raise KeyError(f"unknown document: {document_id}")
        with self.connection:
            self.connection.execute(
                "DELETE FROM documents WHERE id = ?", (document_id,)
            )
        self.documents.pop(document_id)
        removed_ids = {
            chunk_id
            for chunk_id, chunk in self.chunks.items()
            if chunk.document_id == document_id
        }
        for chunk_id in removed_ids:
            del self.chunks[chunk_id]
        self.relationships = [
            edge
            for edge in self.relationships
            if edge.source_chunk_id not in removed_ids
            and edge.target_chunk_id not in removed_ids
        ]
        self.version += 1

    def remove_sector(self, sector_id: str) -> None:
        if any(document.sector_id == sector_id for document in self.documents.values()):
            raise ValueError(f"sector {sector_id} still contains documents")
        if sector_id not in self.sectors:
            raise KeyError(f"unknown sector: {sector_id}")
        with self.connection:
            self.connection.execute("DELETE FROM sectors WHERE id = ?", (sector_id,))
        self.sectors.remove(sector_id)
        self.version += 1

    def close(self) -> None:
        try:
            if self._vector_client is not None and self._vector_collection is not None:
                self._vector_client.delete_collection(self._vector_collection.name)
                self._vector_collection = None
        finally:
            self.connection.close()

    def __enter__(self) -> SQLiteIndex:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

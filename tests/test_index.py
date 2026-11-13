import sqlite3

import pytest

from isthmus.index import InMemoryIndex, SQLiteIndex
from isthmus.models import (
    Chunk,
    Document,
    Relationship,
    RelationshipType,
)


class FixedEmbeddingProvider:
    def embed(self, _text):
        return (1.0, 0.0)


def test_search_returns_top_k_with_deterministic_ties():
    index = InMemoryIndex(FixedEmbeddingProvider())
    document = Document("doc", "Document", "sector")
    index.add_document(
        document,
        [
            Chunk("z", "doc", "sector", "top z", 1, embedding=(1.0, 0.0)),
            Chunk("a", "doc", "sector", "top a", 2, embedding=(1.0, 0.0)),
            Chunk("lower", "doc", "sector", "lower", 3, embedding=(0.0, 1.0)),
        ],
    )

    hits = index.search("query", limit=2)

    assert [hit.chunk.id for hit in hits] == ["a", "z"]
    assert [hit.rank for hit in hits] == [1, 2]


def test_sqlite_index_persists_content_relationships_and_vectors(tmp_path):
    database_path = tmp_path / "isthmus.db"
    document = Document("doc", "Document", "sector")
    chunks = [
        Chunk("one", "doc", "sector", "first chunk", 1),
        Chunk("two", "doc", "sector", "second chunk", 2),
    ]

    with SQLiteIndex(FixedEmbeddingProvider(), database_path) as index:
        index.add_document(document, chunks)
        index.add_relationships(
            [Relationship("one", "two", RelationshipType.NEXT)]
        )

    with SQLiteIndex(FixedEmbeddingProvider(), database_path) as reopened:
        assert reopened.documents["doc"] == document
        assert [hit.chunk.id for hit in reopened.search("query")] == ["one", "two"]
        assert reopened.relationships == [
            Relationship("one", "two", RelationshipType.NEXT)
        ]

    with sqlite3.connect(database_path) as connection:
        embedding = connection.execute(
            "SELECT embedding FROM chunks WHERE id = 'one'"
        ).fetchone()[0]
    assert isinstance(embedding, bytes)
    assert len(embedding) == 16


def test_sqlite_sector_and_document_management(tmp_path):
    database_path = tmp_path / "isthmus.db"
    with SQLiteIndex(FixedEmbeddingProvider(), database_path) as index:
        index.add_sector("Empty")
        document = Document("doc", "Document", "sector")
        index.add_document(
            document,
            [Chunk("chunk", "doc", "sector", "content", 1)],
        )
        index.add_relationship(
            Relationship("chunk", "chunk", RelationshipType.REFERENCES)
        )
        assert index.sectors == {"Empty", "sector"}
        try:
            index.remove_sector("sector")
        except ValueError:
            pass
        else:
            raise AssertionError("a sector containing a document must not be deleted")
        index.remove_document("doc")
        index.remove_sector("sector")
        assert index.chunks == {}
        assert index.relationships == []
        assert index.sectors == {"Empty"}


def test_sqlite_rejects_a_different_embedding_provider(tmp_path):
    class OtherEmbeddingProvider:
        def embed(self, _text):
            return (0.0, 1.0)

    database_path = tmp_path / "isthmus.db"
    with SQLiteIndex(FixedEmbeddingProvider(), database_path) as index:
        index.add_document(
            Document("doc", "Document", "sector"),
            [Chunk("chunk", "doc", "sector", "content", 1)],
        )

    try:
        SQLiteIndex(OtherEmbeddingProvider(), database_path)
    except ValueError as error:
        assert "selected provider" in str(error)
    else:
        raise AssertionError("an incompatible embedding provider must be rejected")


def test_reindex_rolls_back_document_if_relationship_storage_fails(tmp_path):
    database_path = tmp_path / "isthmus.db"
    document = Document("doc", "Document", "sector")
    edge = Relationship("new", "new", RelationshipType.REFERENCES)
    with SQLiteIndex(FixedEmbeddingProvider(), database_path) as index:
        index.add_document(document, [Chunk("old", "doc", "sector", "old", 1)])
        with pytest.raises(sqlite3.IntegrityError):
            index.add_document(
                document, [Chunk("new", "doc", "sector", "new", 1)], [edge, edge]
            )
        assert set(index.chunks) == {"old"}
    with SQLiteIndex(FixedEmbeddingProvider(), database_path) as reopened:
        assert set(reopened.chunks) == {"old"}


def test_loading_relationships_does_not_write_to_database(tmp_path, monkeypatch):
    database_path = tmp_path / "isthmus.db"
    with SQLiteIndex(FixedEmbeddingProvider(), database_path) as index:
        index.add_document(
            Document("doc", "Document", "sector"),
            [Chunk("one", "doc", "sector", "one", 1), Chunk("two", "doc", "sector", "two", 2)],
            [Relationship("one", "two", RelationshipType.NEXT)],
        )

    def reject_writes(self, relationships):
        raise AssertionError("loading must not write relationships")

    monkeypatch.setattr(SQLiteIndex, "add_relationships", reject_writes)
    with SQLiteIndex(FixedEmbeddingProvider(), database_path) as reopened:
        assert len(reopened.relationships) == 1


def test_chroma_search_updates_when_document_changes(tmp_path):
    with SQLiteIndex(FixedEmbeddingProvider(), tmp_path / "vectors.db") as index:
        document = Document("doc", "Document", "sector")
        index.add_document(document, [
            Chunk("first", "doc", "sector", "first", 1, embedding=(1.0, 0.0)),
            Chunk("second", "doc", "sector", "second", 2, embedding=(0.0, 1.0)),
        ])
        hits = index.search_by_vector((1.0, 0.0), limit=2)
        assert [hit.chunk.id for hit in hits] == ["first", "second"]
        assert hits[0].similarity == pytest.approx(1.0)
        assert hits[1].similarity == pytest.approx(0.5)

        index.add_document(document, [
            Chunk("replacement", "doc", "sector", "new", 1, embedding=(1.0, 0.0))
        ])
        assert [hit.chunk.id for hit in index.search_by_vector((1.0, 0.0))] == ["replacement"]

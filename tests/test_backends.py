import sys
from types import ModuleType

import pytest

from conftest import FakeEmbeddingProvider
from isthmus.backends import open_index
from isthmus.index import BackendIndex, SQLiteIndex
from isthmus.models import Chunk, Document
from isthmus.pipeline import Isthmus


class CustomBackend(BackendIndex):
    def __init__(self, provider):
        super().__init__(provider)
        self.provider_id = "custom"
        self.closed = False

    def add_sector(self, sector_id):
        self.sectors.add(sector_id)

    def remove_sector(self, sector_id):
        self.sectors.remove(sector_id)

    def remove_document(self, document_id):
        del self.documents[document_id]

    def close(self):
        self.closed = True


def test_custom_backend_factory_supports_retrieval(monkeypatch):
    module = ModuleType("test_custom_backend")
    seen = []

    def create(provider, database):
        seen.append(database)
        return CustomBackend(provider)

    module.create = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("ISTHMUS_INDEX_FACTORY", "test_custom_backend:create")
    with open_index(FakeEmbeddingProvider(), "my-index") as index:
        index.add_document(Document("doc", "Policy", "sector"), [
            Chunk("chunk", "doc", "sector", "travel policy", 1)
        ])
        assert Isthmus(index).query("travel").context[0].id == "chunk"
    assert index.closed
    assert seen == ["my-index"]


def test_invalid_backend_factory_fails_explicitly(monkeypatch):
    monkeypatch.setenv("ISTHMUS_INDEX_FACTORY", "invalid")
    with pytest.raises(ValueError, match="module:function"):
        open_index(FakeEmbeddingProvider(), "unused")


def test_default_backend_remains_sqlite(monkeypatch, tmp_path):
    monkeypatch.delenv("ISTHMUS_INDEX_FACTORY", raising=False)
    with open_index(FakeEmbeddingProvider(), tmp_path / "test.db") as index:
        assert isinstance(index, SQLiteIndex)

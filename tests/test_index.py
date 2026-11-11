from contextweave.index import InMemoryIndex
from contextweave.models import Chunk, Document


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

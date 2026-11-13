from contextweave.embeddings import HashingEmbeddingProvider
from contextweave.index import InMemoryIndex
from contextweave.models import Chunk, Document, Relationship, RelationshipType, SearchHit
from contextweave.pipeline import ContextWeave, PipelineConfig


class CountingEmbeddingProvider:
    def __init__(self):
        self.calls = 0

    def embed(self, _text):
        self.calls += 1
        return (1.0, 0.0)


def test_pipeline_expands_every_candidate_before_returning_document_scores():
    embeddings = HashingEmbeddingProvider()
    index = InMemoryIndex(embeddings)
    for doc_id, sector, content in [
        ("a", "finance", "travel reimbursement and exceptions policy"),
        ("b", "legal", "travel reimbursement rules with legal exceptions"),
    ]:
        chunks = [
            Chunk(f"{doc_id}-1", doc_id, sector, content, 1, section_id="rules"),
            Chunk(f"{doc_id}-2", doc_id, sector, "exceptions require manager approval", 2, section_id="exceptions"),
        ]
        index.add_document(Document(doc_id, doc_id, sector), chunks)
        index.add_relationship(Relationship(f"{doc_id}-1", f"{doc_id}-2", RelationshipType.NEXT))
    result = ContextWeave(
        index,
        PipelineConfig(sector_threshold=0.0, global_top_k=10, target_documents=1),
    ).query("What are the travel reimbursement rules and exceptions?")
    assert result.expansion_order == result.candidate_document_ids
    assert {package.document_id for package in result.evidence_packages} == set(result.candidate_document_ids)
    assert len(result.document_assessments) == len(result.evidence_packages)


def test_pipeline_embeds_query_once_for_search_and_graph_expansion():
    embeddings = CountingEmbeddingProvider()
    index = InMemoryIndex(embeddings)
    document = Document("doc", "Document", "sector")
    chunks = [
        Chunk("one", "doc", "sector", "first policy chunk", 1),
        Chunk("two", "doc", "sector", "second policy chunk", 2),
    ]
    index.add_document(document, chunks)
    index.add_relationship(
        Relationship("one", "two", RelationshipType.NEXT)
    )
    embeddings.calls = 0

    ContextWeave(
        index,
        PipelineConfig(sector_threshold=0.0, global_top_k=10),
    ).query("policy")

    assert embeddings.calls == 1


def test_pipeline_reuses_graph_structure_until_index_changes():
    index = InMemoryIndex(CountingEmbeddingProvider())
    pipeline = ContextWeave(index)

    original = pipeline._get_graph_retriever()
    assert pipeline._get_graph_retriever() is original

    document = Document("new", "New document", "sector")
    index.add_document(
        document,
        [Chunk("new-chunk", "new", "sector", "new content", 1)],
    )

    assert pipeline._get_graph_retriever() is not original


def test_pipeline_expands_bounded_fallback_candidates_when_evidence_is_incomplete():
    index = InMemoryIndex(CountingEmbeddingProvider())
    main = Chunk("main", "a", "sector", "travel reimbursement policy", 1)
    exception = Chunk("exception", "b", "sector", "exceptions require approval", 1)
    index.add_document(Document("a", "Policy", "sector"), [main])
    index.add_document(Document("b", "Exceptions", "sector"), [exception])
    hits = (
        SearchHit(index.chunks["main"], 0.9, 1),
        SearchHit(index.chunks["exception"], 0.8, 2),
    )
    result = ContextWeave(index, PipelineConfig(
        sector_threshold=0.0, candidate_document_pool=1, fallback_document_pool=2,
    )).query("travel reimbursement and exceptions", query_vector=(1.0, 0.0), global_hits=hits)

    assert result.fallback_used
    assert result.expansion_order == ("a", "b")
    assert result.evidence_coverage["exceptions"] == "sufficient"

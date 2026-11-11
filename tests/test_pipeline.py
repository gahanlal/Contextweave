from contextweave.embeddings import HashingEmbeddingProvider
from contextweave.index import InMemoryIndex
from contextweave.models import Chunk, Document, Relationship, RelationshipType
from contextweave.pipeline import ContextWeave, PipelineConfig


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

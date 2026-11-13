from __future__ import annotations

import json

from isthmus.embeddings import HashingEmbeddingProvider
from isthmus.index import InMemoryIndex
from isthmus.models import Chunk, Document, Relationship, RelationshipType
from isthmus.pipeline import Isthmus, PipelineConfig


def build_demo() -> Isthmus:
    embeddings = HashingEmbeddingProvider()
    index = InMemoryIndex(embeddings)
    document = Document("policy-001", "Travel Reimbursement Policy", "Finance")
    contents = [
        "Employees may claim reimbursement for approved business travel expenses.",
        "Air travel should be booked in economy class at least fourteen days in advance.",
        "Exceptions to advance booking are allowed when urgent business needs prevent it.",
        "A manager must approve any exception before the trip whenever practical.",
        "Submit itemized receipts within thirty days after returning from travel.",
    ]
    chunks = [
        Chunk(
            id=f"policy-{number}",
            document_id=document.id,
            sector_id=document.sector_id,
            content=content,
            sequence_number=number,
            section_id="travel",
            heading_path=("Travel", "Reimbursement"),
            source_location=f"page {number}",
        )
        for number, content in enumerate(contents, start=1)
    ]
    index.add_document(document, chunks)
    for left, right in zip(chunks, chunks[1:]):
        index.add_relationship(
            Relationship(left.id, right.id, RelationshipType.NEXT)
        )
        index.add_relationship(
            Relationship(right.id, left.id, RelationshipType.PREVIOUS)
        )
    index.add_relationship(
        Relationship(chunks[1].id, chunks[2].id, RelationshipType.QUALIFIES)
    )
    return Isthmus(
        index,
        PipelineConfig(sector_threshold=0.0),
    )


def main() -> None:
    result = build_demo().query(
        "What are the travel reimbursement rules and what exceptions apply?"
    )
    print(json.dumps({
        "query": result.query_plan.original_query,
        "selected_sectors": result.selected_sector_ids,
        "candidate_documents": result.candidate_document_ids,
        "expanded_documents": result.expansion_order,
        "selected_documents": result.selected_document_ids,
        "evidence_coverage": result.evidence_coverage,
        "trace": [
            {
                "document_id": package.document_id,
                "seeds": package.seed_chunk_ids,
                "visited": package.visited_chunk_ids,
                "included": package.included_chunk_ids,
                "paths": [
                    {
                        "chunks": path.chunk_ids,
                        "relationships": [relationship.value for relationship in path.relationship_types],
                    }
                    for path in package.paths
                ],
            }
            for package in result.evidence_packages
        ],
        "context": [
            {
                "document_id": chunk.document_id,
                "location": chunk.source_location,
                "content": chunk.content,
            }
            for chunk in result.context
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

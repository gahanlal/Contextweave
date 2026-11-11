from contextweave.graph import AdaptiveGraphRetriever, TraversalConfig
from contextweave.models import (
    Chunk,
    QueryPlan,
    QueryRequirement,
    Relationship,
    RelationshipType,
    SearchHit,
)


def setup_graph(scores, edges, document_id="doc"):
    chunks = [
        Chunk(
            id=chunk_id,
            document_id=document_id,
            sector_id="sector",
            content=f"chunk {chunk_id}",
            sequence_number=number,
            token_count=1,
        )
        for number, (chunk_id, _score) in enumerate(scores, start=1)
    ]
    score_map = dict(scores)
    retriever = AdaptiveGraphRetriever(
        chunks,
        edges,
        similarity=lambda _query, chunk: score_map[chunk.id],
        config=TraversalConfig(
            max_nodes=20,
            max_depth=10,
            max_low_similarity_bridges=1,
            max_backward=20,
            max_forward=20,
            include_threshold=0.6,
        ),
    )
    plan = QueryPlan(
        "policy exception",
        "factual",
        (
            QueryRequirement("main", "policy", frozenset({"policy"})),
            QueryRequirement("exceptions", "exceptions", frozenset({"exception"})),
        ),
        True,
        True,
    )
    hits = [SearchHit(next(chunk for chunk in chunks if chunk.id == chunk_id), score, rank)
            for rank, (chunk_id, score) in enumerate(scores, start=1)
            if chunk_id == "seed"]
    return retriever, plan, hits


def sequential_edges(ids):
    edges = []
    for left, right in zip(ids, ids[1:]):
        edges.extend([
            Relationship(left, right, RelationshipType.NEXT),
            Relationship(right, left, RelationshipType.PREVIOUS),
        ])
    return edges


def test_bidirectional_expansion_and_low_similarity_bridge():
    scores = [("a", 0.9), ("seed", 0.95), ("bridge", 0.2), ("answer", 0.85)]
    retriever, plan, hits = setup_graph(scores, sequential_edges(["a", "seed", "bridge", "answer"]))
    evidence = retriever.retrieve("doc", plan, hits)
    assert set(evidence.visited_chunk_ids) == {"a", "seed", "bridge", "answer"}
    assert "bridge" not in evidence.included_chunk_ids
    assert "answer" in evidence.included_chunk_ids
    assert any(path.chunk_ids[-1] == "answer" for path in evidence.paths)


def test_multiple_seeds_are_deduplicated_and_cycles_terminate():
    scores = [("seed", 0.95), ("other", 0.91)]
    edges = [
        Relationship("seed", "other", RelationshipType.NEXT),
        Relationship("other", "seed", RelationshipType.NEXT),
    ]
    retriever, plan, hits = setup_graph(scores, edges)
    hits.append(SearchHit(hits[0].chunk, 0.8, 2))
    evidence = retriever.retrieve("doc", plan, hits)
    assert evidence.seed_chunk_ids == ("seed",)
    assert len(evidence.visited_chunk_ids) == 2


def test_traversal_never_crosses_document_boundary():
    chunks = [
        Chunk("seed", "doc", "sector", "seed policy", 1, token_count=1),
        Chunk("foreign", "other-doc", "sector", "foreign policy", 2, token_count=1),
    ]
    retriever = AdaptiveGraphRetriever(
        chunks,
        [Relationship("seed", "foreign", RelationshipType.NEXT)],
        similarity=lambda _query, chunk: 0.95 if chunk.id == "seed" else 0.99,
    )
    plan = QueryPlan(
        "policy",
        "factual",
        (QueryRequirement("main", "policy", frozenset({"policy"})),),
        False,
        False,
    )
    hits = [SearchHit(chunks[0], 0.95, 1)]
    evidence = retriever.retrieve("doc", plan, hits)
    assert evidence.visited_chunk_ids == ("seed",)


def test_qualifies_relationship_is_followed_for_exception_queries():
    scores = [("seed", 0.9), ("exception", 0.8)]
    retriever, plan, hits = setup_graph(
        scores,
        [Relationship("seed", "exception", RelationshipType.QUALIFIES)],
    )
    evidence = retriever.retrieve("doc", plan, hits)
    assert "exception" in evidence.included_chunk_ids
    assert RelationshipType.QUALIFIES in evidence.relationship_types_followed


def test_node_limit_is_respected():
    scores = [(f"c{number}", 0.9) for number in range(8)]
    retriever, plan, hits = setup_graph(scores, sequential_edges([item[0] for item in scores]))
    retriever.config = TraversalConfig(max_nodes=3, max_depth=10, max_forward=20, max_backward=20)
    evidence = retriever.retrieve("doc", plan, hits)
    assert len(evidence.visited_chunk_ids) <= 3

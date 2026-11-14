from conftest import FakeEmbeddingProvider
from isthmus.index import InMemoryIndex
from isthmus.models import Chunk, Document, EvidencePackage, QueryPlan, QueryRequirement, Relationship, RelationshipType, SearchHit
from isthmus.pipeline import Isthmus, PipelineConfig


class CountingEmbeddingProvider:
    def __init__(self):
        self.calls = 0

    def embed(self, _text):
        self.calls += 1
        return (1.0, 0.0)


def test_pipeline_expands_every_candidate_before_returning_document_scores():
    embeddings = FakeEmbeddingProvider()
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
    result = Isthmus(
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

    Isthmus(
        index,
        PipelineConfig(sector_threshold=0.0, global_top_k=10),
    ).query("policy")

    assert embeddings.calls == 1


def test_pipeline_uses_llm_sector_choice_alongside_semantic_scores():
    index = InMemoryIndex(CountingEmbeddingProvider())
    index.add_document(
        Document("finance-doc", "Finance policy", "finance"),
        [Chunk("finance-chunk", "finance-doc", "finance", "expense policy", 1)],
    )
    index.add_document(
        Document("legal-doc", "Legal policy", "legal"),
        [Chunk("legal-chunk", "legal-doc", "legal", "contract policy", 1)],
    )
    observed = {}

    class Selector:
        def select(self, query, scores, heuristic_sectors):
            observed.update(
                query=query,
                scores=scores,
                heuristic_sectors=heuristic_sectors,
            )
            return ("legal",)

    result = Isthmus(
        index,
        PipelineConfig(sector_threshold=0.0, global_top_k=10),
        sector_selector=Selector(),
    ).query("contract policy")

    assert result.selected_sector_ids == ("legal",)
    assert result.candidate_document_ids == ("legal-doc",)
    assert observed["query"] == "contract policy"
    assert set(observed["scores"]) == {"finance", "legal"}
    assert set(observed["heuristic_sectors"]) == {"finance", "legal"}


def test_pipeline_reuses_graph_structure_until_index_changes():
    index = InMemoryIndex(CountingEmbeddingProvider())
    pipeline = Isthmus(index)

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
    exception = Chunk("exception", "b", "sector", "travel reimbursement exceptions require approval", 1)
    index.add_document(Document("a", "Policy", "sector"), [main])
    index.add_document(Document("b", "Exceptions", "sector"), [exception])
    hits = (
        SearchHit(index.chunks["main"], 0.9, 1),
        SearchHit(index.chunks["exception"], 0.8, 2),
    )
    result = Isthmus(index, PipelineConfig(
        sector_threshold=0.0, candidate_document_pool=1, fallback_document_pool=2,
    )).query("travel reimbursement and exceptions", query_vector=(1.0, 0.0), global_hits=hits)

    assert result.fallback_used
    assert result.expansion_order == ("a", "b")
    assert result.evidence_coverage["exceptions"] == "sufficient"


def test_low_similarity_qualifier_survives_through_final_context():
    index = InMemoryIndex(FakeEmbeddingProvider())
    query = "What does this policy cover?"
    rule = Chunk(
        "rule",
        "policy",
        "sector",
        "The policy covers ordinary travel.",
        1,
        embedding=(1.0, 0.0),
    )
    qualifier = Chunk(
        "qualifier",
        "policy",
        "sector",
        "An exception to policy applies to urgent travel.",
        2,
        embedding=(0.0, 1.0),
    )
    index.add_document(
        Document("policy", "Policy", "sector"),
        [rule, qualifier],
        [Relationship("rule", "qualifier", RelationshipType.QUALIFIES)],
    )

    class Planner:
        def plan(self, _query):
            return QueryPlan(
                query,
                "factual",
                (
                    QueryRequirement("main", query, frozenset({"policy", "travel"})),
                    QueryRequirement("exceptions", query, frozenset({"exception"})),
                ),
                True,
                True,
            )

    result = Isthmus(index, query_planner=Planner()).query(
        query,
        query_vector=(1.0, 0.0),
        global_hits=(SearchHit(rule, 1.0, 1),),
    )

    assert result.evidence_packages[0].similarities["qualifier"] == 0.5
    assert result.evidence_packages[0].requirement_scores["exceptions"] < 0.72
    assert "exceptions" not in result.evidence_packages[0].supported_requirements
    assert "qualifier" in result.evidence_packages[0].included_chunk_ids
    assert "qualifier" in {chunk.id for chunk in result.context}


def test_pipeline_does_not_treat_one_term_as_complete_evidence():
    index = InMemoryIndex(CountingEmbeddingProvider())
    index.add_document(
        Document("doc", "Policy", "sector"),
        [Chunk("contractor", "doc", "sector", "The contractor attended.", 1)],
    )
    result = Isthmus(index).query("contractor reimbursement")
    assert result.evidence_coverage["main"] == "partial"
    assert "main" not in result.evidence_packages[0].supported_requirements


def test_context_budget_prioritizes_unique_requirement_evidence():
    index = InMemoryIndex(CountingEmbeddingProvider())
    chunks = [
        Chunk("main", "a", "sector", "main", 1, token_count=5),
        Chunk("redundant", "a", "sector", "redundant", 2, token_count=5),
        Chunk("exception", "b", "sector", "exception", 1, token_count=5),
    ]
    index.add_document(Document("a", "A", "sector"), chunks[:2])
    index.add_document(Document("b", "B", "sector"), chunks[2:])
    def package(doc_id, ids):
        return EvidencePackage(doc_id, (), (), ids, {}, (), (), frozenset(), 0, "")
    plan = QueryPlan("question", "factual", (
        QueryRequirement("main", "main", frozenset({"main"})),
        QueryRequirement("exceptions", "exceptions", frozenset({"exceptions"})),
    ), True, True)
    scores = {
        "main": {"main": 0.95},
        "redundant": {"main": 0.9},
        "exception": {"exceptions": 0.8},
    }
    pipeline = Isthmus(index, PipelineConfig(global_token_budget=10))
    context = pipeline._assemble_context(
        ("a", "b"), [package("a", ("main", "redundant")), package("b", ("exception",))],
        plan, lambda chunk: scores[chunk.id],
    )
    assert tuple(chunk.id for chunk in context) == ("main", "exception")

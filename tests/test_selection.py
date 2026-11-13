from isthmus.models import EvidencePackage, QueryPlan, QueryRequirement
from isthmus.selection import rerank_documents


def package(document_id, score, supported, sections=()):
    return EvidencePackage(
        document_id=document_id,
        seed_chunk_ids=(),
        visited_chunk_ids=(),
        included_chunk_ids=(),
        similarities={f"{document_id}-chunk": score},
        paths=(),
        relationship_types_followed=(),
        supported_requirements=frozenset(supported),
        token_count=0,
        explanation="",
    )


def plan(requirement_ids):
    return QueryPlan(
        "question",
        "factual",
        tuple(
            QueryRequirement(item, item, frozenset({item}))
            for item in requirement_ids
        ),
        False,
        False,
    )


def test_complete_single_document_is_sufficient():
    selected, assessments = rerank_documents(
        [package("complete", 0.9, {"main", "exception"}), package("redundant", 0.95, {"main"})],
        plan(("main", "exception")),
        target_documents=3,
    )
    assert selected == ("complete",)
    assert [item.document_id for item in assessments if item.selected] == ["complete"]


def test_unique_exception_evidence_outranks_a_redundant_document():
    selected, _ = rerank_documents(
        [
            package("policy", 0.92, {"main"}),
            package("duplicate", 0.95, {"main"}),
            package("qualification", 0.70, {"exception"}),
        ],
        plan(("main", "exception")),
        target_documents=1,
        max_documents=3,
    )
    assert selected == ("policy", "qualification")


def test_more_than_target_documents_can_cover_distinct_requirements():
    evidence = [
        package(f"doc-{number}", score, {f"r{number}"})
        for number, score in enumerate((0.95, 0.8, 0.6, 0.3), start=1)
    ]
    selected, _ = rerank_documents(
        evidence,
        plan(("r1", "r2", "r3", "r4")),
        target_documents=3,
        max_documents=5,
    )
    assert selected == ("doc-1", "doc-2", "doc-3", "doc-4")

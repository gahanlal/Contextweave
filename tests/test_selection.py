import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from isthmus.models import EvidencePackage, QueryPlan, QueryRequirement
from isthmus.query import sector_selector_from_environment
from isthmus.selection import LLMSectorSelector, rerank_documents


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


def test_llm_sector_selector_uses_query_scores_and_heuristic_shortlist():
    prompts = []
    selector = LLMSectorSelector(lambda prompt: prompts.append(prompt) or '{"selected_sectors":["legal"]}')

    selected = selector.select(
        "contract termination",
        {"finance": 0.62, "legal": 0.81},
        ("legal",),
    )

    payload = json.loads(prompts[0].split("\n", 1)[1])
    assert selected == ("legal",)
    assert payload["query"] == "contract termination"
    assert payload["sectors"] == [
        {"id": "finance", "semantic_score": 0.62, "heuristic_selected": False},
        {"id": "legal", "semantic_score": 0.81, "heuristic_selected": True},
    ]


def test_llm_sector_selector_rejects_unknown_sectors():
    selector = LLMSectorSelector(lambda _prompt: '{"selected_sectors":["unknown"]}')

    with pytest.raises(ValueError, match="invalid sector IDs"):
        selector.select("question", {"legal": 0.8}, ("legal",))


def test_llm_sector_selector_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ISTHMUS_LLM_SECTOR_SELECTION", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert sector_selector_from_environment() is None


@pytest.mark.parametrize(
    ("setting", "enabled"),
    [("true", None), ("false", True)],
)
def test_llm_sector_selector_can_be_opted_in(monkeypatch, setting, enabled):
    requests = []

    class Client:
        def __init__(self, **_kwargs):
            self.responses = self

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(output_text='{"selected_sectors":["finance"]}')

    monkeypatch.setattr("isthmus.query.OpenAI", Client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ISTHMUS_LLM_SECTOR_SELECTION", setting)
    selector = sector_selector_from_environment(enabled=enabled)

    assert selector is not None
    assert selector.select("travel costs", {"finance": 0.8}, ("finance",)) == ("finance",)
    assert requests[0]["model"] == "gpt-4o-mini"


def test_llm_sector_selector_rejects_invalid_opt_in_value(monkeypatch):
    monkeypatch.setenv("ISTHMUS_LLM_SECTOR_SELECTION", "sometimes")

    with pytest.raises(RuntimeError, match="must be true or false"):
        sector_selector_from_environment()


def test_complete_single_document_is_sufficient():
    selected, assessments = rerank_documents(
        [package("complete", 0.9, {"main", "exception"}), package("redundant", 0.95, {"main"})],
        plan(("main", "exception")),
        target_documents=1,
    )
    assert selected == ("complete",)
    assert [item.document_id for item in assessments if item.selected] == ["complete"]


def test_marginal_policy_stops_redundant_documents_before_the_target():
    selected, _ = rerank_documents(
        [package("a", 0.95, {"main"}), package("b", 0.9, {"main"}), package("c", 0.1, {"main"})],
        plan(("main",)),
        target_documents=3,
    )
    assert selected == ("a",)


def test_fixed_target_policy_can_be_compared_with_marginal_policy():
    evidence = [
        package("a", 0.95, {"main"}),
        package("b", 0.9, {"main"}),
        package("c", 0.1, {"main"}),
    ]

    selected, _ = rerank_documents(
        evidence,
        plan(("main",)),
        target_documents=3,
        selection_policy="fixed_target",
    )

    assert selected == ("a", "b")


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
    assert selected == ("duplicate", "qualification")


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


def test_marginal_policy_stops_redundant_documents_when_coverage_is_incomplete():
    evidence = [
        package("a", 0.9, {"main"}),
        package("b", 0.8, {"main"}),
        package("c", 0.7, {"main"}),
    ]
    selected, _ = rerank_documents(evidence, plan(("main", "missing")), target_documents=2)
    assert selected == ("a",)


def test_near_ties_use_score_then_id_not_input_order():
    evidence = [package("z", 0.9, {"main"}), package("a", 0.9, {"main"})]
    assert rerank_documents(evidence, plan(("main",)), target_documents=1)[0] == ("a",)
    assert rerank_documents(list(reversed(evidence)), plan(("main",)), target_documents=1)[0] == ("a",)


def test_marginal_partial_coverage_is_selected_before_complete_evidence():
    evidence = [
        replace(package("first", 0.9, {"r1"}), requirement_scores={"r1": 1.0}),
        replace(package("partial", 1.0, set()), requirement_scores={"r2": 0.71}),
        replace(package("complete", 0.0, {"r2"}), requirement_scores={"r2": 0.72}),
    ]
    selected, _ = rerank_documents(evidence, plan(("r1", "r2")), target_documents=1)
    assert selected == ("first", "partial", "complete")

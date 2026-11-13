import json

from isthmus.coverage import aggregate_coverage, coverage_status, requirement_score, semantic_scores
from isthmus.models import Chunk, QueryPlan, QueryRequirement
from isthmus.query import LLMQueryPlanner, understand_query


def test_one_shared_term_is_not_full_requirement_coverage():
    requirement = QueryRequirement(
        "main", "contractor reimbursement exceptions",
        frozenset({"contractor", "reimbursement", "exceptions"}),
    )
    chunk = Chunk("one", "doc", "sector", "A contractor attended.", 1, embedding=(1.0, 0.0))
    assert coverage_status(requirement_score(requirement, chunk, 1.0)) == "partial"
    two_terms = QueryRequirement("main", "contractor reimbursement", frozenset({"contractor", "reimbursement"}))
    assert coverage_status(requirement_score(two_terms, chunk, 1.0)) == "partial"


def test_coverage_aggregates_per_chunk_not_disconnected_terms():
    requirement = QueryRequirement("main", "contractor reimbursement policy", frozenset({"contractor", "reimbursement", "policy"}))
    plan = QueryPlan("question", "factual", (requirement,), False, False)
    chunks = [
        Chunk("a", "doc", "sector", "contractor", 1, embedding=(1.0, 0.0)),
        Chunk("b", "doc", "sector", "reimbursement", 2, embedding=(1.0, 0.0)),
    ]
    scores = aggregate_coverage(plan, chunks, lambda chunk: semantic_scores(plan, chunk, {"main": (1.0, 0.0)}))
    assert coverage_status(scores["main"]) == "partial"


def test_strong_semantic_match_can_support_paraphrased_evidence():
    requirement = QueryRequirement("main", "deadline for expense receipts", frozenset({"deadline", "expense", "receipts"}))
    chunk = Chunk("one", "doc", "sector", "Submit itemized claims within thirty days.", 1)
    assert coverage_status(requirement_score(requirement, chunk, 0.95)) == "sufficient"


def test_deterministic_planner_recognizes_waivers():
    assert understand_query("Can the booking deadline be waived?").asks_exceptions


def test_exception_coverage_needs_topic_as_well_as_exception_language():
    plan = understand_query("contractor reimbursement exceptions")
    vectors = {requirement.id: (1.0, 0.0) for requirement in plan.requirements}
    unrelated = Chunk("unrelated", "doc", "sector", "Employee waiver is available.", 1, embedding=(1.0, 0.0))
    partial_topic = Chunk("partial", "doc", "sector", "Contractor waiver is available.", 2, embedding=(1.0, 0.0))
    relevant = Chunk("relevant", "doc", "sector", "Contractor reimbursement waiver is available.", 3, embedding=(1.0, 0.0))
    assert coverage_status(semantic_scores(plan, unrelated, vectors)["exceptions"]) == "partial"
    assert coverage_status(semantic_scores(plan, partial_topic, vectors)["exceptions"]) == "partial"
    assert coverage_status(semantic_scores(plan, relevant, vectors)["exceptions"]) == "sufficient"


def test_deterministic_planner_separates_comparisons_and_explicit_questions():
    comparison = understand_query("Compare travel and reimbursement policies")
    assert [requirement.description for requirement in comparison.requirements] == ["travel", "reimbursement policies"]
    multipart = understand_query("What are the booking rules and who approves them?")
    assert [requirement.id for requirement in multipart.requirements] == ["main", "part-2"]
    assert len(understand_query("travel reimbursement and exceptions").requirements) == 2


def test_optional_model_planner_preserves_query_and_validates_output():
    query = "Can the booking deadline ever be waived?"
    planner = LLMQueryPlanner(lambda _: json.dumps({
        "intent": "factual",
        "asks_exceptions": True,
        "requirements": [{"id": "main", "description": "booking deadline"}],
    }))
    plan = planner.plan(query)
    assert plan.original_query == query
    assert plan.asks_exceptions
    assert {requirement.id for requirement in plan.requirements} == {"main", "exceptions"}


def test_optional_model_planner_falls_back_on_invalid_response(caplog):
    plan = LLMQueryPlanner(lambda _: "not json").plan("What are the rules?")
    assert plan.original_query == "What are the rules?"
    assert "using deterministic planner" in caplog.text


def test_optional_model_planner_uses_configured_model(monkeypatch):
    from types import SimpleNamespace
    import isthmus.query as query_module

    calls = []

    class FakeOpenAIProvider:
        def __init__(self):
            self.model = "chosen-planner"
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=self))

        def create(self, **kwargs):
            calls.append(kwargs)
            content = json.dumps({
                "intent": "factual", "asks_exceptions": False,
                "requirements": [{"id": "main", "description": "booking rules"}],
            })
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setenv("ISTHMUS_QUERY_PLANNER_MODEL", "chosen-planner")
    monkeypatch.delenv("ISTHMUS_QUERY_PLANNER_PROTOCOL", raising=False)
    monkeypatch.setattr(query_module, "OpenAIAnswerProvider", FakeOpenAIProvider)
    monkeypatch.setattr(query_module, "answer_provider", lambda protocol, model, endpoint: FakeOpenAIProvider())
    planner = query_module.planner_from_environment()
    assert planner is not None
    assert planner.plan("What are the rules?").requirements[0].description == "booking rules"
    assert calls[0]["model"] == "chosen-planner"

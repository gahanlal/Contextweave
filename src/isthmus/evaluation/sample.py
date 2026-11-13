from __future__ import annotations

from isthmus.demo import build_demo
from isthmus.embeddings import EmbeddingProvider
from isthmus.evaluation.dataset import EvaluationCase
from isthmus.index import InMemoryIndex


def build_sample_evaluation(provider: EmbeddingProvider) -> tuple[InMemoryIndex, tuple[EvaluationCase, ...]]:
    index = build_demo(provider).index
    cases = (
        EvaluationCase(
            case_id="booking-window",
            query="How far in advance should air travel be booked?",
            evidence_chunk_ids=frozenset({"policy-2"}),
            supporting_document_ids=frozenset({"policy-001"}),
            requirement_chunk_ids={"main": frozenset({"policy-2"})},
        ),
        EvaluationCase(
            case_id="booking-exception",
            query="What exceptions to advance booking are allowed and who approves them?",
            evidence_chunk_ids=frozenset({"policy-3", "policy-4"}),
            exception_chunk_ids=frozenset({"policy-3", "policy-4"}),
            supporting_document_ids=frozenset({"policy-001"}),
            requirement_chunk_ids={
                "exception": frozenset({"policy-3"}),
                "approval": frozenset({"policy-4"}),
            },
        ),
        EvaluationCase(
            case_id="receipt-deadline",
            query="When should travel receipts be submitted?",
            evidence_chunk_ids=frozenset({"policy-5"}),
            supporting_document_ids=frozenset({"policy-001"}),
            requirement_chunk_ids={"main": frozenset({"policy-5"})},
        ),
    )
    return index, cases

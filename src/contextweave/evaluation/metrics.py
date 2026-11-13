from __future__ import annotations

from dataclasses import dataclass

from contextweave.evaluation.baselines import BaselineResult
from contextweave.evaluation.dataset import EvaluationCase
from contextweave.index import InMemoryIndex


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    case_id: str
    baseline: str
    evidence_recall: float
    exception_recall: float | None
    context_completeness: float
    document_precision: float
    document_recall: float
    document_f1: float
    chunks_retrieved: int
    context_tokens: int
    latency_ms: float


def score_case(
    index: InMemoryIndex,
    case: EvaluationCase,
    result: BaselineResult,
) -> CaseMetrics:
    retrieved_chunks = {chunk.id for chunk in result.chunks}
    retrieved_documents = set(result.document_ids)
    evidence_recall = _recall(case.evidence_chunk_ids, retrieved_chunks)
    exception_recall = (
        _recall(case.exception_chunk_ids, retrieved_chunks)
        if case.exception_chunk_ids
        else None
    )
    requirements = case.requirement_chunk_ids or {
        "all": case.evidence_chunk_ids
    }
    context_completeness = sum(
        bool(required_chunks & retrieved_chunks)
        for required_chunks in requirements.values()
    ) / len(requirements)
    gold_documents = set(case.supporting_document_ids)
    if not gold_documents:
        gold_documents = {
            index.chunks[chunk_id].document_id
            for chunk_id in case.evidence_chunk_ids
            if chunk_id in index.chunks
        }
    if not gold_documents:
        raise ValueError(
            f"case {case.case_id} needs supporting_document_ids or indexed evidence IDs"
        )
    overlap = len(gold_documents & retrieved_documents)
    precision = overlap / len(retrieved_documents) if retrieved_documents else 0.0
    recall = overlap / len(gold_documents)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CaseMetrics(
        case_id=case.case_id,
        baseline=result.name,
        evidence_recall=evidence_recall,
        exception_recall=exception_recall,
        context_completeness=context_completeness,
        document_precision=precision,
        document_recall=recall,
        document_f1=f1,
        chunks_retrieved=len(retrieved_chunks),
        context_tokens=sum(chunk.token_count for chunk in result.chunks),
        latency_ms=result.latency_ms,
    )


def _recall(expected: frozenset[str], actual: set[str]) -> float:
    return len(expected & actual) / len(expected) if expected else 0.0

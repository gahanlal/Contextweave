from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import mean

from contextweave.evaluation.baselines import run_baselines
from contextweave.evaluation.dataset import EvaluationCase
from contextweave.evaluation.metrics import CaseMetrics, score_case
from contextweave.index import InMemoryIndex
from contextweave.pipeline import ContextWeave, PipelineConfig


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    case_count: int
    per_case: tuple[CaseMetrics, ...]
    summary: dict[str, dict[str, float | None]]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "summary": self.summary,
            "per_case": [asdict(item) for item in self.per_case],
        }


def evaluate_dataset(
    index: InMemoryIndex,
    cases: tuple[EvaluationCase, ...],
    config: PipelineConfig | None = None,
) -> EvaluationReport:
    if not cases:
        raise ValueError("evaluation dataset contains no cases")
    for case in cases:
        labeled_chunks = set(case.evidence_chunk_ids) | set(case.exception_chunk_ids)
        for chunk_ids in case.requirement_chunk_ids.values():
            labeled_chunks.update(chunk_ids)
        unknown_chunks = labeled_chunks - index.chunks.keys()
        unknown_documents = case.supporting_document_ids - index.documents.keys()
        if unknown_chunks or unknown_documents:
            raise ValueError(
                f"case {case.case_id} refers to unknown chunk IDs "
                f"{sorted(unknown_chunks)} or document IDs {sorted(unknown_documents)}"
            )
    pipeline = ContextWeave(index, config)
    scored_cases = []
    for case in cases:
        results = run_baselines(index, case.query, config, pipeline)
        scored_cases.extend(score_case(index, case, result) for result in results)
    return EvaluationReport(
        case_count=len(cases),
        per_case=tuple(scored_cases),
        summary=_summarize(scored_cases),
    )


def _summarize(
    rows: list[CaseMetrics],
) -> dict[str, dict[str, float | None]]:
    by_baseline: dict[str, list[CaseMetrics]] = defaultdict(list)
    for row in rows:
        by_baseline[row.baseline].append(row)
    fields = (
        "evidence_recall",
        "exception_recall",
        "context_completeness",
        "document_precision",
        "document_recall",
        "document_f1",
        "chunks_retrieved",
        "context_tokens",
        "latency_ms",
    )
    summary = {}
    for baseline, baseline_rows in by_baseline.items():
        summary[baseline] = {
            field: (
                mean(values)
                if (values := [
                    getattr(row, field)
                    for row in baseline_rows
                    if getattr(row, field) is not None
                ])
                else None
            )
            for field in fields
        }
    return summary

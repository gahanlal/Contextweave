from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    query: str
    evidence_chunk_ids: frozenset[str]
    exception_chunk_ids: frozenset[str] = frozenset()
    supporting_document_ids: frozenset[str] = frozenset()
    requirement_chunk_ids: dict[str, frozenset[str]] = field(default_factory=dict)


def load_dataset(path: str | Path) -> tuple[EvaluationCase, ...]:
    with Path(path).open(encoding="utf-8") as stream:
        return load_dataset_stream(stream)


def load_dataset_stream(stream: TextIO) -> tuple[EvaluationCase, ...]:
    cases = []
    seen_ids = set()
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            case = _case_from_mapping(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid evaluation dataset entry on line {line_number}: {error}"
            ) from error
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("evaluation dataset contains no cases")
    return tuple(cases)


def _case_from_mapping(data: object) -> EvaluationCase:
    if not isinstance(data, dict):
        raise TypeError("each JSONL line must be an object")
    case_id = data["id"]
    query = data["query"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("id must be a non-empty string")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    evidence = _string_set(data["evidence_chunk_ids"], "evidence_chunk_ids")
    if not evidence:
        raise ValueError("evidence_chunk_ids must not be empty")
    exceptions = _string_set(
        data.get("exception_chunk_ids", []),
        "exception_chunk_ids",
    )
    documents = _string_set(
        data.get("supporting_document_ids", []),
        "supporting_document_ids",
    )
    raw_requirements = data.get("requirement_chunk_ids", {})
    if not isinstance(raw_requirements, dict):
        raise TypeError("requirement_chunk_ids must be an object")
    requirements = {
        requirement_id: _string_set(chunk_ids, f"requirement {requirement_id}")
        for requirement_id, chunk_ids in raw_requirements.items()
    }
    return EvaluationCase(
        case_id=case_id.strip(),
        query=query.strip(),
        evidence_chunk_ids=evidence,
        exception_chunk_ids=exceptions,
        supporting_document_ids=documents,
        requirement_chunk_ids=requirements,
    )


def _string_set(value: object, field_name: str) -> frozenset[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise TypeError(f"{field_name} must be a list of non-empty strings")
    return frozenset(item.strip() for item in value)

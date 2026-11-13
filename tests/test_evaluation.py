from io import StringIO
import json

import pytest

from contextweave.evaluation.dataset import load_dataset_stream
from contextweave.evaluation.runner import evaluate_dataset
from contextweave.evaluation.sample import build_sample_evaluation


def test_dataset_loader_parses_evidence_and_requirement_labels():
    dataset = load_dataset_stream(
        StringIO(
            json.dumps(
                {
                    "id": "q1",
                    "query": "What exception applies?",
                    "evidence_chunk_ids": ["chunk-a"],
                    "exception_chunk_ids": ["chunk-a"],
                    "supporting_document_ids": ["doc-a"],
                    "requirement_chunk_ids": {"exception": ["chunk-a"]},
                }
            )
        )
    )

    assert dataset[0].evidence_chunk_ids == frozenset({"chunk-a"})
    assert dataset[0].exception_chunk_ids == frozenset({"chunk-a"})
    assert dataset[0].requirement_chunk_ids == {
        "exception": frozenset({"chunk-a"})
    }


def test_evaluation_runs_all_five_retrieval_baselines():
    index, cases = build_sample_evaluation()

    report = evaluate_dataset(index, cases)

    assert report.case_count == 3
    assert len(report.per_case) == 15
    assert set(report.summary) == {
        "A: top-k",
        "B: fixed window",
        "C: adaptive graph",
        "D: sector + graph",
        "E: document reranking",
    }
    assert report.summary["E: document reranking"]["evidence_recall"] == 1.0


def test_dataset_loader_rejects_missing_gold_evidence():
    with pytest.raises(ValueError, match="evidence_chunk_ids"):
        load_dataset_stream(
            StringIO(
                json.dumps(
                    {
                        "id": "q1",
                        "query": "What happened?",
                        "evidence_chunk_ids": [],
                    }
                )
            )
        )


def test_evaluation_rejects_labels_not_in_index():
    index, cases = build_sample_evaluation()
    from dataclasses import replace

    invalid_case = replace(cases[0], evidence_chunk_ids=frozenset({"missing"}))
    with pytest.raises(ValueError, match="unknown chunk IDs"):
        evaluate_dataset(index, (invalid_case,))

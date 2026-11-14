import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from isthmus.models import Chunk, SearchHit
from isthmus.demo import build_demo
from conftest import FakeEmbeddingProvider


path = Path(__file__).resolve().parents[1] / "evaluation" / "docugami_benchmark.py"
spec = importlib.util.spec_from_file_location("docugami_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
comparator_spec = importlib.util.spec_from_file_location(
    "docugami_comparators", path.with_name("docugami_comparators.py")
)
comparators = importlib.util.module_from_spec(comparator_spec)
comparator_spec.loader.exec_module(comparators)
sys.path.insert(0, str(path.parent))
answer_qa_path = path.with_name("docugami_answer_qa.py")
answer_qa_spec = importlib.util.spec_from_file_location("docugami_answer_qa", answer_qa_path)
answer_qa = importlib.util.module_from_spec(answer_qa_spec)
answer_qa_spec.loader.exec_module(answer_qa)


def test_document_metrics_include_missing_and_extra_sources():
    scores = benchmark.document_metrics({"a", "b"}, {"a", "c"})
    assert scores == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "any_hit": 1.0,
        "all_sources": 0.0,
    }
    with pytest.raises(ValueError):
        benchmark.document_metrics(set(), {"a"})


def test_comparison_applies_the_same_word_budget_to_baselines():
    chunks = [
        Chunk("a", "a", "sector", "one", 1, token_count=4),
        Chunk("b", "b", "sector", "two", 2, token_count=5),
        Chunk("c", "c", "sector", "three", 3, token_count=2),
    ]
    assert [chunk.id for chunk in benchmark.budget_chunks(chunks, 6)] == ["a", "c"]


def test_size_matching_truncates_all_contexts_to_equal_word_counts():
    top_k = [
        Chunk("a", "doc-a", "sector", "one two three four", 1, token_count=4),
    ]
    reranked = [
        Chunk("b", "doc-b", "sector", "five six", 1, token_count=2),
    ]
    third = [
        Chunk("c", "doc-c", "sector", "seven eight nine", 1, token_count=3),
    ]

    matched = benchmark.match_context_words_many(
        {"top_k": top_k, "reranked": reranked, "third": third},
        4,
    )

    assert {
        arm: sum(chunk.token_count for chunk in chunks)
        for arm, chunks in matched.items()
    } == {"top_k": 2, "reranked": 2, "third": 2}
    assert matched["top_k"][0].content == "one two"
    assert matched["reranked"][0].content == "five six"


def test_passage_labels_include_page_level_facts():
    labels_path = path.with_name("docugami_passage_labels.json")
    labels = answer_qa.load_passage_labels(labels_path)

    assert len(labels) == 7
    assert all(case["required_facts"] and case["evidence_pages"] for case in labels)
    assert all(
        isinstance(reference["page"], int) and reference["path"].endswith(".pdf")
        for case in labels
        for reference in case["evidence_pages"]
    )


def test_answer_qa_summary_averages_blinded_judge_scores_by_arm():
    cases = [
        {
            "arms": {
                "top_k": {"error": None, "citation_format_valid": True},
                "fixed_target": {"error": None, "citation_format_valid": True},
                "marginal_coverage": {"error": None, "citation_format_valid": True},
            },
            "judge": {
                "scores": {
                    "top_k": {"correctness": 2, "completeness": 1, "citation_support": 3},
                    "fixed_target": {"correctness": 3, "completeness": 3, "citation_support": 3},
                    "marginal_coverage": {"correctness": 4, "completeness": 3, "citation_support": 4},
                },
            },
        }
    ]

    summary = answer_qa._summary(cases)

    assert summary["marginal_coverage"] == {
        "correctness": 4,
        "completeness": 3,
        "citation_support": 4,
        "citation_format_failures": 0,
    }
    assert summary["pairwise_score_outcomes"]["fixed_target_vs_top_k"] == {
        "fixed_target_wins": 1,
        "top_k_wins": 0,
        "ties": 0,
    }
    assert summary["pairwise_score_outcomes"]["fixed_target_vs_marginal_coverage"] == {
        "fixed_target_wins": 0,
        "marginal_coverage_wins": 1,
        "ties": 0,
    }


def test_answer_qa_keeps_uncited_drafts_for_grounding_judgment():
    class FakeProvider:
        def generate(self, prompt):
            assert json.loads(prompt)["evidence"]
            return '{"answer":"The policy reimburses the expense."}'

    pipeline = build_demo(FakeEmbeddingProvider())
    result = pipeline.query("What are the travel reimbursement rules?")

    answer = answer_qa._generate_qa_answer(
        result,
        pipeline.index.documents,
        FakeProvider(),
    )

    assert answer["text"] == "The policy reimburses the expense."
    assert answer["citation_format_valid"] is False
    assert answer["cited_evidence"] == []


def test_hybrid_search_combines_lexical_and_dense_candidates():
    chunks = [
        Chunk("dense", "a", "sector", "unrelated text", 1),
        Chunk("keyword", "b", "sector", "contractor reimbursement policy", 1),
    ]
    hits = [SearchHit(chunks[0], 0.9, 1)]
    ranked = comparators.HybridSearch(chunks).rank("contractor reimbursement", hits, 2)
    assert {chunk.id for chunk in ranked} == {"dense", "keyword"}


def test_page_merging_only_expands_pages_with_enough_matches():
    chunks = [
        Chunk("a", "doc", "sector", "a", 1, source_location="page 1"),
        Chunk("b", "doc", "sector", "b", 2, source_location="page 1"),
        Chunk("c", "doc", "sector", "c", 3, source_location="page 1"),
        Chunk("d", "doc", "sector", "d", 4, source_location="page 2"),
        Chunk("e", "doc", "sector", "e", 5, source_location="page 2"),
    ]
    hits = [SearchHit(chunks[0], 0.9, 1), SearchHit(chunks[1], 0.8, 2), SearchHit(chunks[3], 0.7, 3)]
    assert [chunk.id for chunk in comparators.PageMerger(chunks).merge(hits)] == ["a", "b", "c", "d"]


def test_cross_encoder_reranks_question_passage_pairs():
    class Encoding:
        ids = [1, 2]
        attention_mask = [1, 1]
        type_ids = [0, 1]

    class FakeTokenizer:
        def encode_batch(self, pairs):
            assert pairs == [("question", "low"), ("question", "high")]
            return [Encoding(), Encoding()]

    class FakeSession:
        def get_inputs(self):
            return [type("Input", (), {"name": "input_ids"})()]

        def run(self, _outputs, inputs):
            assert inputs["input_ids"].shape == (2, 2)
            return [np.array([[0.1], [0.9]])]

    ranker = comparators.CrossEncoder.__new__(comparators.CrossEncoder)
    ranker.tokenizer = FakeTokenizer()
    ranker.session = FakeSession()
    chunks = [Chunk("low", "a", "s", "low", 1), Chunk("high", "b", "s", "high", 1)]
    assert [chunk.id for chunk in ranker.rerank("question", chunks)] == ["high", "low"]

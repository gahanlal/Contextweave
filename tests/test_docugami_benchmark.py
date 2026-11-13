import importlib.util
from pathlib import Path

import numpy as np
import pytest

from isthmus.models import Chunk, SearchHit


path = Path(__file__).resolve().parents[1] / "evaluation" / "docugami_benchmark.py"
spec = importlib.util.spec_from_file_location("docugami_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
comparator_spec = importlib.util.spec_from_file_location(
    "docugami_comparators", path.with_name("docugami_comparators.py")
)
comparators = importlib.util.module_from_spec(comparator_spec)
comparator_spec.loader.exec_module(comparators)


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

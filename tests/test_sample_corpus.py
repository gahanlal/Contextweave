from pathlib import Path

from isthmus.document_parsing import parse_document_bytes
from isthmus.evaluation.dataset import load_dataset
from isthmus.evaluation.runner import evaluate_dataset
from isthmus.index import InMemoryIndex
from conftest import FakeEmbeddingProvider


def test_sample_labels_match_the_distributed_documents():
    root = Path(__file__).resolve().parents[1] / "sample_data"
    index = InMemoryIndex(FakeEmbeddingProvider())
    for path in (root / "travel-policy.txt", root / "expense-security.txt"):
        document, chunks, relationships = parse_document_bytes(path.read_bytes(), path.name, "Sample")
        index.add_document(document, chunks, relationships)
    report = evaluate_dataset(index, load_dataset(root / "evaluation.jsonl"))
    assert report.case_count == 4
    assert len(report.per_case) == 20

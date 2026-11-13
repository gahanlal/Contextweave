import json
import sys
from types import SimpleNamespace

import pytest

from isthmus import cli, ingestion
from conftest import FakeEmbeddingProvider
from isthmus.index import SQLiteIndex
from isthmus.models import Chunk, Document


def test_query_cli_can_generate_answer(tmp_path, monkeypatch, capsys):
    database = tmp_path / "test.db"
    provider = FakeEmbeddingProvider()
    with SQLiteIndex(provider, database) as index:
        index.add_document(
            Document("doc", "Policy", "Finance"),
            [Chunk("chunk", "doc", "Finance", "Approved travel is reimbursed.", 1)],
        )
    monkeypatch.setattr(cli, "OpenAIEmbeddingProvider", lambda **kwargs: provider)
    monkeypatch.setattr(
        cli,
        "answer_provider",
        lambda *args: type("AnswerProvider", (), {
            "generate": lambda self, prompt: '{"answer":"Approved travel is reimbursed [S1]."}',
            "assess": lambda self, prompt: '{"supported":true}',
        })(),
    )
    monkeypatch.setattr(sys, "argv", ["isthmus", "query", "travel", "--database", str(database), "--model", "embed", "--answer", "--answer-model", "answer"])

    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert result["answer"].endswith("[S1].")
    assert result["citations"][0]["chunk_id"] == "chunk"


@pytest.mark.parametrize("sector_args", [[], ["--sector", "  "]])
def test_index_cli_requires_nonempty_sector_before_opening_provider(monkeypatch, sector_args, capsys):
    monkeypatch.setattr(sys, "argv", ["isthmus", "index-pdfs", "--model", "embed", *sector_args])
    monkeypatch.setattr(cli, "OpenAIEmbeddingProvider", lambda **kwargs: pytest.fail("provider opened"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "sector" in capsys.readouterr().err


def test_index_cli_assigns_sector_to_every_pdf(tmp_path, monkeypatch, capsys):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "cyber-law.pdf").write_bytes(b"First content")
    (pdf_dir / "finance.pdf").write_bytes(b"Second content")
    database = tmp_path / "index.db"
    provider = FakeEmbeddingProvider()
    monkeypatch.setattr(cli, "OpenAIEmbeddingProvider", lambda **kwargs: provider)
    monkeypatch.setattr(
        ingestion, "PdfReader",
        lambda stream: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: stream.read().decode())]),
    )
    monkeypatch.setattr(sys, "argv", [
        "isthmus", "index-pdfs", "--pdf-dir", str(pdf_dir), "--database", str(database),
        "--model", "embed", "--sector", "Custom",
    ])
    cli.main()
    assert "2" in capsys.readouterr().out
    with SQLiteIndex(provider, database) as index:
        assert len(index.documents) == 2
        assert {doc.sector_id for doc in index.documents.values()} == {"Custom"}
        assert {chunk.sector_id for chunk in index.chunks.values()} == {"Custom"}

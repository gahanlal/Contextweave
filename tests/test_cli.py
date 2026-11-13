import json
import sys

from contextweave import cli
from contextweave.embeddings import HashingEmbeddingProvider
from contextweave.index import SQLiteIndex
from contextweave.models import Chunk, Document


def test_query_cli_can_generate_answer(tmp_path, monkeypatch, capsys):
    database = tmp_path / "test.db"
    provider = HashingEmbeddingProvider()
    with SQLiteIndex(provider, database) as index:
        index.add_document(
            Document("doc", "Policy", "Finance"),
            [Chunk("chunk", "doc", "Finance", "Approved travel is reimbursed.", 1)],
        )
    monkeypatch.setattr(cli, "OpenAIEmbeddingProvider", lambda **kwargs: provider)
    monkeypatch.setattr(
        cli,
        "OpenAIAnswerProvider",
        lambda: type("AnswerProvider", (), {"generate": lambda self, prompt: '{"answer":"Approved travel is reimbursed [S1]."}'})(),
    )
    monkeypatch.setattr(sys, "argv", ["contextweave", "query", "travel", "--database", str(database), "--answer"])

    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert result["answer"].endswith("[S1].")
    assert result["citations"][0]["chunk_id"] == "chunk"

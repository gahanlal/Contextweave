from types import SimpleNamespace

import pytest

from contextweave.embeddings import OpenAIEmbeddingProvider


def test_openai_provider_batches_and_restores_input_order(monkeypatch):
    class FakeClient:
        class Embeddings:
            @staticmethod
            def create(**kwargs):
                assert kwargs["input"] == ["first", "second"]
                assert kwargs["model"] == "text-embedding-3-small"
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                        SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                    ]
                )

        embeddings = Embeddings()

    monkeypatch.setattr(
        "contextweave.embeddings.OpenAI",
        lambda api_key: FakeClient(),
    )
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    assert provider.embed_many(["first", "second"]) == (
        (1.0, 0.0),
        (0.0, 1.0),
    )


def test_openai_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider()

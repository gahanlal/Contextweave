from types import SimpleNamespace

import pytest

from isthmus.embeddings import OpenAIEmbeddingProvider


def test_openai_provider_batches_and_restores_input_order(monkeypatch):
    class FakeClient:
        class Embeddings:
            @staticmethod
            def create(**kwargs):
                assert kwargs["input"] == ["first", "second"]
                assert kwargs["model"] == "chosen-embedding"
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                        SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                    ]
                )

        embeddings = Embeddings()

    monkeypatch.setattr(
        "isthmus.embeddings.OpenAI",
        lambda **kwargs: FakeClient(),
    )
    provider = OpenAIEmbeddingProvider(model="chosen-embedding", api_key="test-key")

    assert provider.embed_many(["first", "second"]) == (
        (1.0, 0.0),
        (0.0, 1.0),
    )


def test_openai_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ISTHMUS_EMBEDDING_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider("chosen-embedding")


def test_provider_identity_distinguishes_endpoint_and_dimensions(monkeypatch):
    monkeypatch.setattr("isthmus.embeddings.OpenAI", lambda **kwargs: object())
    original = OpenAIEmbeddingProvider("chosen", 512, "test")
    other_endpoint = OpenAIEmbeddingProvider("chosen", 512, "test", "http://localhost:8000/v1")
    other_size = OpenAIEmbeddingProvider("chosen", 256, "test")
    assert original.provider_id == "openai:chosen:512"
    assert len({original.provider_id, other_endpoint.provider_id, other_size.provider_id}) == 3


def test_provider_rejects_credentials_in_endpoint(monkeypatch):
    with pytest.raises(ValueError, match="without credentials"):
        OpenAIEmbeddingProvider("chosen", api_key="test", base_url="https://user:pass@example.com/v1")

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Sequence
from typing import Protocol
from urllib.parse import urlsplit

from openai import OpenAI


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> tuple[float, ...]:
        """Return a normalized vector for text."""


def token_limit(provider: EmbeddingProvider) -> tuple[Callable[[str], int], int] | None:
    counter = getattr(provider, "count_tokens", None)
    maximum = getattr(provider, "max_input_tokens", None)
    if counter is None and maximum is None:
        return None
    if not callable(counter) or not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ValueError("A token-aware embedding provider must supply count_tokens and a positive max_input_tokens.")
    return counter, maximum


def count_tokens(counter: Callable[[str], int], text: str) -> int:
    count = counter(text)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Embedding token counter must return a non-negative integer.")
    return count


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str,
        dimensions: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Specify an embedding model ID.")
        if dimensions is not None and dimensions < 1:
            raise ValueError("Embedding dimensions must be positive.")
        if base_url:
            parsed = urlsplit(base_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Embedding endpoint must be an HTTP(S) URL without credentials or query parameters.")
        if api_key is None:
            api_key = os.environ.get("ISTHMUS_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set ISTHMUS_EMBEDDING_API_KEY or OPENAI_API_KEY in the environment or .env file."
            )
        self.model = model.strip()
        self.dimensions = dimensions
        endpoint = f":{hashlib.sha256(base_url.encode()).hexdigest()[:16]}" if base_url else ""
        self.provider_id = f"openai:{self.model}:{dimensions or 'default'}{endpoint}"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        kwargs = {"input": list(texts), "model": self.model}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        vectors: list[tuple[float, ...] | None] = [None] * len(texts)
        for item in response.data:
            if item.index < 0 or item.index >= len(vectors) or vectors[item.index] is not None:
                raise RuntimeError("Embedding service returned invalid vector indexes.")
            vectors[item.index] = tuple(item.embedding)
        if any(vector is None for vector in vectors):
            raise RuntimeError(
                f"Embedding service returned fewer than {len(texts)} vectors."
            )
        return tuple(vector for vector in vectors if vector is not None)


def embed_many(
    provider: EmbeddingProvider,
    texts: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    budget = token_limit(provider)
    if budget is not None:
        counter, maximum = budget
        for text in texts:
            length = count_tokens(counter, text)
            if length > maximum:
                raise ValueError(
                    f"Embedding input has {length} tokens, exceeding the provider limit of {maximum}; "
                    "split the text with the provider's tokenizer before indexing or querying."
                )
    batch_method = getattr(provider, "embed_many", None)
    if batch_method is not None:
        vectors = tuple(batch_method(texts))
        if len(vectors) != len(texts):
            raise ValueError("embedding provider returned an unexpected batch size")
        return vectors
    return tuple(provider.embed(text) for text in texts)


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not magnitude:
        return tuple(vector)
    if abs(magnitude - 1.0) < 1e-12:
        return tuple(vector)
    return tuple(value / magnitude for value in vector)


def normalized_cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    cosine = sum(a * b for a, b in zip(left, right))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

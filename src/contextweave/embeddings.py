from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> tuple[float, ...]:
        """Return a normalized vector for text."""


class HashingEmbeddingProvider:
    """A deterministic, dependency-free lexical vectorizer for local experiments.

    Replace this provider with a semantic embedding implementation for production
    experiments. This baseline does not claim semantic understanding.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self.dimensions = dimensions
        self.provider_id = f"hashing-v1:{dimensions}"

    def embed(self, text: str) -> tuple[float, ...]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        vector = [0.0] * self.dimensions
        for feature in words:
            _accumulate_feature(vector, feature)
        for left, right in zip(words, words[1:]):
            _accumulate_feature(vector, f"{left}_{right}")
        return normalize_vector(vector)


def _accumulate_feature(vector: list[float], feature: str) -> None:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4], "big") % len(vector)
    sign = 1.0 if digest[4] & 1 else -1.0
    vector[index] += sign


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = 512,
        api_key: str | None = None,
    ) -> None:
        if api_key is None:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY in the environment or your local .env file."
            )
        self.model = model
        self.dimensions = dimensions
        self.provider_id = f"openai:{model}:{dimensions or 'default'}"
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        response = self.client.embeddings.create(
            input=list(texts),
            model=self.model,
            dimensions=self.dimensions,
        )
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

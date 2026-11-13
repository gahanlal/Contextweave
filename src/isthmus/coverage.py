from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from isthmus.embeddings import normalized_cosine_similarity
from isthmus.models import Chunk, QueryPlan, QueryRequirement
from isthmus.query import terms


SUFFICIENT = 0.72
PARTIAL = 0.35


def requirement_score(
    requirement: QueryRequirement,
    chunk: Chunk,
    semantic_similarity: float | None = None,
    topic_terms: frozenset[str] = frozenset(),
) -> float:
    if not requirement.terms:
        return 0.0
    content_terms = terms(chunk.content)
    matches = len(requirement.terms & content_terms)
    if requirement.id == "exceptions":
        lexical = float(matches > 0)
    else:
        lexical = matches / len(requirement.terms)
    if semantic_similarity is None:
        score = lexical
    elif requirement.id != "exceptions" and matches == 0 and semantic_similarity >= 0.90 and len(content_terms) >= 3:
        return semantic_similarity * 0.85
    else:
        score = 0.65 * semantic_similarity + 0.35 * lexical
    if requirement.id != "exceptions" and len(requirement.terms) >= 2 and matches == 1:
        return min(score, PARTIAL)
    if requirement.id == "exceptions":
        topic_matches = len(topic_terms & content_terms)
        if matches == 0 or topic_matches < min(2, len(topic_terms)) or not topic_terms:
            return min(score, PARTIAL)
    return score


def aggregate_coverage(
    plan: QueryPlan,
    chunks: Iterable[Chunk],
    score_chunk: Callable[[Chunk], Mapping[str, float]],
) -> dict[str, float]:
    scores = {requirement.id: 0.0 for requirement in plan.requirements}
    for chunk in chunks:
        for requirement_id, score in score_chunk(chunk).items():
            scores[requirement_id] = max(scores[requirement_id], score)
    return scores


def semantic_scores(
    plan: QueryPlan,
    chunk: Chunk,
    vectors: Mapping[str, tuple[float, ...]],
) -> dict[str, float]:
    topic_terms = frozenset().union(
        *(requirement.terms for requirement in plan.requirements if requirement.id != "exceptions")
    )
    return {
        requirement.id: requirement_score(
            requirement,
            chunk,
            normalized_cosine_similarity(vectors[requirement.id], chunk.embedding),
            topic_terms,
        )
        for requirement in plan.requirements
    }


def coverage_status(score: float) -> str:
    if score >= SUFFICIENT:
        return "sufficient"
    if score >= PARTIAL:
        return "partial"
    return "insufficient"

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from isthmus.models import (
    Chunk,
    EvidencePackage,
    QueryPlan,
    Relationship,
    RelationshipType,
    SearchHit,
    TraversalPath,
)
from isthmus.coverage import SUFFICIENT, aggregate_coverage, requirement_score


@dataclass(frozen=True, slots=True)
class TraversalConfig:
    max_seeds_per_document: int = 3
    max_nodes: int = 40
    max_depth: int = 6
    max_low_similarity_bridges: int = 2
    max_backward: int = 12
    max_forward: int = 12
    max_tokens_per_document: int = 1800
    include_threshold: float = 0.62
    qualifies_threshold: float = 0.42


@dataclass(frozen=True, slots=True)
class _State:
    chunk_id: str
    direction: RelationshipType | None
    depth: int
    low_similarity_streak: int
    path: tuple[str, ...]
    edge_types: tuple[RelationshipType, ...]


class AdaptiveGraphRetriever:
    """Bounded, bidirectional traversal with separate visit/traverse/include rules."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        relationships: Iterable[Relationship],
        similarity: Callable[[str, Chunk], float],
        config: TraversalConfig | None = None,
    ) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.chunks_by_document: dict[str, dict[str, Chunk]] = {}
        for chunk in self.chunks.values():
            self.chunks_by_document.setdefault(chunk.document_id, {})[chunk.id] = chunk
        self.similarity = similarity
        self.config = config or TraversalConfig()
        self.adjacency: dict[str, list[Relationship]] = {}
        for edge in relationships:
            source = self.chunks.get(edge.source_chunk_id)
            target = self.chunks.get(edge.target_chunk_id)
            if source is None or target is None or source.document_id != target.document_id:
                continue
            self.adjacency.setdefault(edge.source_chunk_id, []).append(edge)

    def retrieve(
        self,
        document_id: str,
        query_plan: QueryPlan,
        seed_hits: Iterable[SearchHit],
        similarity: Callable[[str, Chunk], float] | None = None,
        score_requirements: Callable[[Chunk], dict[str, float]] | None = None,
    ) -> EvidencePackage:
        document_chunks = self.chunks_by_document.get(document_id, {})
        seeds = []
        seen_seeds: set[str] = set()
        for hit in sorted(seed_hits, key=lambda item: (-item.similarity, item.rank)):
            chunk_id = hit.chunk.id
            if (
                chunk_id in document_chunks
                and chunk_id not in seen_seeds
                and len(seeds) < self.config.max_seeds_per_document
            ):
                seeds.append(chunk_id)
                seen_seeds.add(chunk_id)
        if not seeds:
            return EvidencePackage(
                document_id=document_id,
                seed_chunk_ids=(),
                visited_chunk_ids=(),
                included_chunk_ids=(),
                similarities={},
                paths=(),
                relationship_types_followed=(),
                supported_requirements=frozenset(),
                token_count=0,
                explanation="No global retrieval seeds were found for this document.",
            )

        query = query_plan.original_query
        similarity_for_query = similarity or self.similarity
        relevant_edges = self._eligible_relationships(query_plan)
        queue: deque[_State] = deque()
        for seed in seeds:
            queue.append(_State(seed, None, 0, 0, (seed,), ()))
            queue.append(_State(seed, RelationshipType.PREVIOUS, 0, 0, (seed,), ()))
            queue.append(_State(seed, RelationshipType.NEXT, 0, 0, (seed,), ()))

        visited: list[str] = []
        included: list[str] = []
        similarities: dict[str, float] = {}
        paths: list[TraversalPath] = []
        followed: list[RelationshipType] = []
        visited_set: set[str] = set()
        included_set: set[str] = set()
        expanded_states: set[tuple[str, RelationshipType | None]] = set()
        direction_counts = {RelationshipType.PREVIOUS: 0, RelationshipType.NEXT: 0}
        requirement_scores_cache: dict[str, dict[str, float]] = {}
        topic_terms = frozenset().union(
            *(requirement.terms for requirement in query_plan.requirements if requirement.id != "exceptions")
        )
        def scores_for(chunk: Chunk) -> dict[str, float]:
            if chunk.id not in requirement_scores_cache:
                requirement_scores_cache[chunk.id] = (
                    score_requirements(chunk) if score_requirements is not None else {
                        requirement.id: requirement_score(requirement, chunk, topic_terms=topic_terms)
                        for requirement in query_plan.requirements
                    }
                )
            return requirement_scores_cache[chunk.id]
        token_count = 0

        while queue and len(visited) < self.config.max_nodes:
            state = queue.popleft()
            if state.chunk_id not in document_chunks:
                continue
            state_key = (state.chunk_id, state.direction)
            if state_key in expanded_states:
                continue
            expanded_states.add(state_key)
            chunk = document_chunks[state.chunk_id]
            score = similarities.get(state.chunk_id)
            if score is None:
                score = similarity_for_query(query, chunk)
                similarities[state.chunk_id] = score
            if state.chunk_id not in visited_set:
                visited_set.add(state.chunk_id)
                visited.append(state.chunk_id)
            supports_requirement = any(score >= SUFFICIENT for score in scores_for(chunk).values())
            should_include = score >= self.config.include_threshold or supports_requirement
            if state.chunk_id in seen_seeds:
                should_include = True
            if should_include and state.chunk_id not in included_set:
                if token_count + chunk.token_count <= self.config.max_tokens_per_document:
                    included.append(state.chunk_id)
                    included_set.add(state.chunk_id)
                    token_count += chunk.token_count

            if state.depth >= self.config.max_depth:
                continue
            edges = self._edges_for(
                state.chunk_id,
                state.direction,
                relevant_edges,
                query_plan,
            )
            for edge in edges:
                next_id = edge.target_chunk_id
                next_chunk = document_chunks.get(next_id)
                if next_chunk is None:
                    continue
                if edge.relationship_type in direction_counts:
                    if direction_counts[edge.relationship_type] >= self._direction_limit(edge.relationship_type):
                        continue
                    direction_counts[edge.relationship_type] += 1
                next_score = similarities.get(next_id)
                if next_score is None:
                    next_score = similarity_for_query(query, next_chunk)
                    similarities[next_id] = next_score
                is_qualifies = edge.relationship_type == RelationshipType.QUALIFIES
                low_streak = 0 if next_score >= self.config.include_threshold else state.low_similarity_streak + 1
                can_traverse = (
                    next_score >= self.config.include_threshold
                    or low_streak <= self.config.max_low_similarity_bridges
                    or (is_qualifies and query_plan.asks_exceptions)
                )
                if is_qualifies and query_plan.asks_exceptions:
                    can_traverse = next_score >= self.config.qualifies_threshold or low_streak <= self.config.max_low_similarity_bridges
                if not can_traverse:
                    continue
                paths.append(
                    TraversalPath(
                        chunk_ids=state.path + (next_id,),
                        relationship_types=state.edge_types + (edge.relationship_type,),
                    )
                )
                if edge.relationship_type not in followed:
                    followed.append(edge.relationship_type)
                queue.append(
                    _State(
                        chunk_id=next_id,
                        direction=state.direction,
                        depth=state.depth + 1,
                        low_similarity_streak=low_streak,
                        path=state.path + (next_id,),
                        edge_types=state.edge_types + (edge.relationship_type,),
                    )
                )

        coverage = aggregate_coverage(
            query_plan,
            (document_chunks[chunk_id] for chunk_id in included),
            scores_for,
        )
        return EvidencePackage(
            document_id=document_id,
            seed_chunk_ids=tuple(seeds),
            visited_chunk_ids=tuple(visited),
            included_chunk_ids=tuple(
                sorted(included, key=lambda chunk_id: document_chunks[chunk_id].sequence_number)
            ),
            similarities=similarities,
            paths=tuple(paths),
            relationship_types_followed=tuple(followed),
            supported_requirements=frozenset(key for key, value in coverage.items() if value >= SUFFICIENT),
            requirement_scores=coverage,
            token_count=token_count,
            explanation=(
                f"Expanded {len(seeds)} seed(s) bidirectionally; visited {len(visited)} "
                f"chunk(s), included {len(included)} after similarity and token checks."
            ),
        )

    def _eligible_relationships(self, query_plan: QueryPlan) -> set[RelationshipType]:
        relationships = {RelationshipType.PREVIOUS, RelationshipType.NEXT}
        if query_plan.asks_exceptions:
            relationships.add(RelationshipType.QUALIFIES)
        return relationships

    def _edges_for(
        self,
        chunk_id: str,
        direction: RelationshipType | None,
        eligible: set[RelationshipType],
        query_plan: QueryPlan,
    ) -> list[Relationship]:
        result = []
        for edge in self.adjacency.get(chunk_id, []):
            if edge.relationship_type not in eligible:
                continue
            if direction in (RelationshipType.PREVIOUS, RelationshipType.NEXT):
                if edge.relationship_type != direction:
                    continue
            elif edge.relationship_type == RelationshipType.PREVIOUS or edge.relationship_type == RelationshipType.NEXT:
                continue
            result.append(edge)
        return result

    def _direction_limit(self, relationship: RelationshipType) -> int:
        if relationship == RelationshipType.PREVIOUS:
            return self.config.max_backward
        return self.config.max_forward

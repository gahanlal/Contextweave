from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
from typing import Literal

from isthmus.coverage import aggregate_coverage, coverage_status, semantic_scores
from isthmus.embeddings import embed_many, normalize_vector
from isthmus.graph import AdaptiveGraphRetriever, TraversalConfig
from isthmus.index import InMemoryIndex
from isthmus.models import Chunk, EvidencePackage, QueryPlan, RetrievalResult, SearchHit
from isthmus.query import QueryPlanner, planner_from_environment, understand_query
from isthmus.selection import (
    SectorSelector,
    discover_documents,
    rerank_documents,
    score_sectors,
    select_sectors,
)
from openai import OpenAIError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    global_top_k: int = 30
    sector_threshold: float = 0.75
    fallback_sector_margin: float = 0.10
    candidate_document_pool: int = 5
    fallback_document_pool: int = 10
    target_documents: int = 3
    max_selected_documents: int = 5
    document_selection_policy: Literal["fixed_target", "marginal_coverage"] = "marginal_coverage"
    minimum_marginal_coverage: float = 0.05
    global_token_budget: int = 4500
    traversal: TraversalConfig = TraversalConfig()


class Isthmus:
    """Orchestrates global retrieval through document-level evidence selection."""

    def __init__(
        self,
        index: InMemoryIndex,
        config: PipelineConfig | None = None,
        query_planner: QueryPlanner | None = None,
        sector_selector: SectorSelector | None = None,
    ) -> None:
        self.index = index
        self.config = config or PipelineConfig()
        self.query_planner = query_planner if query_planner is not None else planner_from_environment()
        self.sector_selector = sector_selector
        self._graph_retriever: AdaptiveGraphRetriever | None = None
        self._graph_index_version = -1

    def query(
        self,
        text: str,
        *,
        query_vector: tuple[float, ...] | None = None,
        global_hits: tuple[SearchHit, ...] | None = None,
    ) -> RetrievalResult:
        plan = self.query_planner.plan(text) if self.query_planner else understand_query(text)
        if query_vector is None:
            query_vector = self.index.embed_query(text)
        requirement_vectors = {
            requirement.id: query_vector
            for requirement in plan.requirements
            if requirement.description == plan.original_query
        }
        additional = [
            requirement for requirement in plan.requirements
            if requirement.id not in requirement_vectors
        ]
        if additional:
            vectors = embed_many(
                self.index.embedding_provider,
                [requirement.description for requirement in additional],
            )
            requirement_vectors.update(
                (requirement.id, normalize_vector(vector))
                for requirement, vector in zip(additional, vectors)
            )
        hits = global_hits
        if hits is None:
            hits = self.index.search_by_vector(
                query_vector,
                limit=self.config.global_top_k,
            )
        sector_scores = score_sectors(hits)
        heuristic_sectors = select_sectors(
            sector_scores,
            threshold=self.config.sector_threshold,
            fallback_margin=self.config.fallback_sector_margin,
        )
        selected_sectors = heuristic_sectors
        if self.sector_selector is not None:
            try:
                selected_sectors = self.sector_selector.select(
                    plan.original_query,
                    sector_scores,
                    heuristic_sectors,
                )
            except (ValueError, OpenAIError) as error:
                logger.warning(
                    "Sector selector failed; using similarity-based selection: %s",
                    type(error).__name__,
                )
        candidate_documents = discover_documents(
            hits,
            selected_sectors,
            limit=self.config.candidate_document_pool,
        )
        hit_groups: dict[str, list[SearchHit]] = {}
        for hit in hits:
            hit_groups.setdefault(hit.chunk.document_id, []).append(hit)

        graph_retriever = self._get_graph_retriever()
        similarity = lambda _query, chunk: self.index.similarity_to_vector(
            query_vector,
            chunk,
        )
        requirement_similarity = lambda chunk: semantic_scores(plan, chunk, requirement_vectors)
        expansion_order = []
        packages: list[EvidencePackage] = []
        for document_id in candidate_documents:
            expansion_order.append(document_id)
            packages.append(
                graph_retriever.retrieve(
                    document_id,
                    plan,
                    hit_groups[document_id],
                    similarity=similarity,
                    score_requirements=requirement_similarity,
                )
            )

        selected_documents, assessments = rerank_documents(
            packages,
            plan,
            target_documents=self.config.target_documents,
            max_documents=self.config.max_selected_documents,
            selection_policy=self.config.document_selection_policy,
            minimum_marginal_coverage=self.config.minimum_marginal_coverage,
        )
        context = self._assemble_context(selected_documents, packages, plan, requirement_similarity)
        coverage = self._coverage(plan, context, requirement_similarity)
        fallback_used = False
        if any(value != "sufficient" for value in coverage.values()):
            fallback_candidates = discover_documents(
                hits,
                selected_sectors,
                limit=max(self.config.candidate_document_pool, self.config.fallback_document_pool),
            )
            extra_documents = tuple(
                document_id for document_id in fallback_candidates
                if document_id not in candidate_documents
            )
            if extra_documents:
                fallback_used = True
                for document_id in extra_documents:
                    expansion_order.append(document_id)
                    packages.append(
                        graph_retriever.retrieve(
                            document_id,
                            plan,
                            hit_groups[document_id],
                            similarity=similarity,
                            score_requirements=requirement_similarity,
                        )
                    )
                candidate_documents += extra_documents
                selected_documents, assessments = rerank_documents(
                    packages,
                    plan,
                    target_documents=self.config.target_documents,
                    max_documents=self.config.max_selected_documents,
                    selection_policy=self.config.document_selection_policy,
                    minimum_marginal_coverage=self.config.minimum_marginal_coverage,
                )
                context = self._assemble_context(selected_documents, packages, plan, requirement_similarity)
                coverage = self._coverage(plan, context, requirement_similarity)
        return RetrievalResult(
            query_plan=plan,
            global_hits=hits,
            sector_scores=sector_scores,
            selected_sector_ids=selected_sectors,
            candidate_document_ids=candidate_documents,
            evidence_packages=tuple(packages),
            document_assessments=assessments,
            selected_document_ids=selected_documents,
            context=context,
            evidence_coverage=coverage,
            expansion_order=tuple(expansion_order),
            fallback_used=fallback_used,
        )

    def _get_graph_retriever(self) -> AdaptiveGraphRetriever:
        if (
            self._graph_retriever is None
            or self._graph_index_version != self.index.version
        ):
            self._graph_retriever = AdaptiveGraphRetriever(
                chunks=self.index.chunks.values(),
                relationships=self.index.relationships,
                similarity=self.index.similarity,
                config=self.config.traversal,
            )
            self._graph_index_version = self.index.version
        return self._graph_retriever

    def _assemble_context(
        self,
        selected_document_ids: tuple[str, ...],
        packages: list[EvidencePackage],
        plan: QueryPlan,
        score_requirements: Callable[[Chunk], Mapping[str, float]],
    ) -> tuple[Chunk, ...]:
        package_by_id = {package.document_id: package for package in packages}
        candidates = []
        seen: set[tuple[str, str]] = set()
        for document_id in selected_document_ids:
            package = package_by_id[document_id]
            for chunk_id in package.included_chunk_ids:
                chunk = self.index.chunks[chunk_id]
                key = (chunk.document_id, chunk.content_hash or chunk.content)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((chunk, score_requirements(chunk)))
        coverage = {requirement.id: 0.0 for requirement in plan.requirements}
        selected: set[str] = set()
        used_tokens = 0
        while True:
            eligible = [
                (position, chunk, scores)
                for position, (chunk, scores) in enumerate(candidates)
                if chunk.id not in selected
                and used_tokens + chunk.token_count <= self.config.global_token_budget
            ]
            if not eligible:
                break
            _, chunk, scores = max(
                eligible,
                key=lambda item: (
                    sum(max(0.0, item[2].get(key, 0.0) - value) for key, value in coverage.items()),
                    -item[0],
                ),
            )
            selected.add(chunk.id)
            used_tokens += chunk.token_count
            for key in coverage:
                coverage[key] = max(coverage[key], scores.get(key, 0.0))
        return tuple(chunk for chunk, _ in candidates if chunk.id in selected)

    @staticmethod
    def _coverage(plan, context: tuple[Chunk, ...], scorer) -> dict[str, str]:
        scores = aggregate_coverage(plan, context, scorer)
        return {key: coverage_status(score) for key, score in scores.items()}

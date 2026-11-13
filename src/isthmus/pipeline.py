from __future__ import annotations

from dataclasses import dataclass

from isthmus.graph import AdaptiveGraphRetriever, TraversalConfig
from isthmus.index import InMemoryIndex
from isthmus.models import Chunk, EvidencePackage, RetrievalResult, SearchHit
from isthmus.query import understand_query
from isthmus.selection import discover_documents, rerank_documents, score_sectors, select_sectors


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    global_top_k: int = 30
    sector_threshold: float = 0.75
    fallback_sector_margin: float = 0.10
    candidate_document_pool: int = 5
    fallback_document_pool: int = 10
    target_documents: int = 3
    max_selected_documents: int = 5
    global_token_budget: int = 4500
    traversal: TraversalConfig = TraversalConfig()


class Isthmus:
    """Orchestrates global retrieval through document-level evidence selection."""

    def __init__(self, index: InMemoryIndex, config: PipelineConfig | None = None) -> None:
        self.index = index
        self.config = config or PipelineConfig()
        self._graph_retriever: AdaptiveGraphRetriever | None = None
        self._graph_index_version = -1

    def query(
        self,
        text: str,
        *,
        query_vector: tuple[float, ...] | None = None,
        global_hits: tuple[SearchHit, ...] | None = None,
    ) -> RetrievalResult:
        plan = understand_query(text)
        if query_vector is None:
            query_vector = self.index.embed_query(text)
        hits = global_hits
        if hits is None:
            hits = self.index.search_by_vector(
                query_vector,
                limit=self.config.global_top_k,
            )
        sector_scores = score_sectors(hits)
        selected_sectors = select_sectors(
            sector_scores,
            threshold=self.config.sector_threshold,
            fallback_margin=self.config.fallback_sector_margin,
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
                )
            )

        selected_documents, assessments = rerank_documents(
            packages,
            plan,
            target_documents=self.config.target_documents,
            max_documents=self.config.max_selected_documents,
        )
        context = self._assemble_context(selected_documents, packages)
        coverage = self._coverage(plan, context)
        fallback_used = False
        if "insufficient" in coverage.values():
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
                        )
                    )
                candidate_documents += extra_documents
                selected_documents, assessments = rerank_documents(
                    packages,
                    plan,
                    target_documents=self.config.target_documents,
                    max_documents=self.config.max_selected_documents,
                )
                context = self._assemble_context(selected_documents, packages)
                coverage = self._coverage(plan, context)
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
    ) -> tuple[Chunk, ...]:
        package_by_id = {package.document_id: package for package in packages}
        result = []
        seen: set[tuple[str, str]] = set()
        used_tokens = 0
        for document_id in selected_document_ids:
            package = package_by_id[document_id]
            for chunk_id in package.included_chunk_ids:
                chunk = self.index.chunks[chunk_id]
                key = (chunk.document_id, chunk.content_hash or chunk.content)
                if key in seen:
                    continue
                if used_tokens + chunk.token_count > self.config.global_token_budget:
                    continue
                result.append(chunk)
                seen.add(key)
                used_tokens += chunk.token_count
        return tuple(result)

    @staticmethod
    def _coverage(plan, context: tuple[Chunk, ...]) -> dict[str, str]:
        supported = AdaptiveGraphRetriever._supported_requirements(plan, context)
        return {
            requirement.id: (
                "sufficient" if requirement.id in supported else "insufficient"
            )
            for requirement in plan.requirements
        }

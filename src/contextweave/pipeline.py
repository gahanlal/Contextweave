from __future__ import annotations

from dataclasses import dataclass

from contextweave.graph import AdaptiveGraphRetriever, TraversalConfig
from contextweave.index import InMemoryIndex
from contextweave.models import Chunk, EvidencePackage, RetrievalResult, SearchHit
from contextweave.query import understand_query
from contextweave.selection import discover_documents, rerank_documents, score_sectors, select_sectors


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    global_top_k: int = 30
    sector_threshold: float = 0.75
    fallback_sector_margin: float = 0.10
    candidate_document_pool: int = 5
    target_documents: int = 3
    max_selected_documents: int = 5
    global_token_budget: int = 4500
    traversal: TraversalConfig = TraversalConfig()


class ContextWeave:
    """Orchestrates global retrieval through document-level evidence selection."""

    def __init__(self, index: InMemoryIndex, config: PipelineConfig | None = None) -> None:
        self.index = index
        self.config = config or PipelineConfig()

    def query(self, text: str) -> RetrievalResult:
        plan = understand_query(text)
        hits = self.index.search(text, limit=self.config.global_top_k)
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
        hit_groups: dict[str, list[SearchHit]] = {doc_id: [] for doc_id in candidate_documents}
        for hit in hits:
            if hit.chunk.document_id in hit_groups:
                hit_groups[hit.chunk.document_id].append(hit)

        graph_retriever = AdaptiveGraphRetriever(
            chunks=self.index.chunks.values(),
            relationships=self.index.relationships,
            similarity=lambda query, chunk: self.index.similarity(query, chunk),
            config=self.config.traversal,
        )
        expansion_order = []
        packages: list[EvidencePackage] = []
        for document_id in candidate_documents:
            expansion_order.append(document_id)
            packages.append(graph_retriever.retrieve(document_id, plan, hit_groups[document_id]))

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
        )

    def _assemble_context(
        self,
        selected_document_ids: tuple[str, ...],
        packages: list[EvidencePackage],
    ) -> tuple[Chunk, ...]:
        package_by_id = {package.document_id: package for package in packages}
        document_order = {
            document_id: position
            for position, document_id in enumerate(selected_document_ids)
        }
        selected_chunks = []
        for document_id in selected_document_ids:
            package = package_by_id[document_id]
            selected_chunks.extend(
                self.index.chunks[chunk_id] for chunk_id in package.included_chunk_ids
            )
        selected_chunks.sort(
            key=lambda chunk: (
                document_order[chunk.document_id],
                chunk.sequence_number,
            )
        )
        result = []
        seen: set[tuple[str, str]] = set()
        used_tokens = 0
        for chunk in selected_chunks:
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

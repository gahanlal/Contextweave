from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from contextweave.graph import AdaptiveGraphRetriever
from contextweave.index import InMemoryIndex
from contextweave.models import Chunk, EvidencePackage, QueryPlan, SearchHit
from contextweave.pipeline import ContextWeave, PipelineConfig
from contextweave.query import understand_query
from contextweave.selection import (
    discover_documents,
    score_sectors,
    select_sectors,
)

BASELINE_NAMES = (
    "A: top-k",
    "B: fixed window",
    "C: adaptive graph",
    "D: sector + graph",
    "E: document reranking",
)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    name: str
    chunks: tuple[Chunk, ...]
    document_ids: tuple[str, ...]
    latency_ms: float


def run_baselines(
    index: InMemoryIndex,
    query: str,
    config: PipelineConfig | None = None,
    pipeline: ContextWeave | None = None,
) -> tuple[BaselineResult, ...]:
    config = config or PipelineConfig()
    pipeline = pipeline or ContextWeave(index, config)
    shared_started = perf_counter()
    query_vector = index.embed_query(query)
    hits = index.search_by_vector(query_vector, limit=config.global_top_k)
    shared_latency_ms = (perf_counter() - shared_started) * 1000

    results = []

    def add_result(
        name: str,
        chunks: tuple[Chunk, ...],
        started: float,
        extra_latency_ms: float = 0.0,
    ) -> None:
        results.append(
            BaselineResult(
                name=name,
                chunks=chunks,
                document_ids=_document_ids(chunks),
                latency_ms=(
                    shared_latency_ms
                    + extra_latency_ms
                    + (perf_counter() - started) * 1000
                ),
            )
        )

    started = perf_counter()
    add_result("A: top-k", tuple(hit.chunk for hit in hits), started)

    started = perf_counter()
    fixed_window = _fixed_window(index, hits)
    add_result("B: fixed window", fixed_window, started)

    plan = understand_query(query)
    sector_scores = score_sectors(hits)
    all_sectors = tuple(sector_scores)
    unfiltered_documents = discover_documents(
        hits,
        all_sectors,
        limit=config.candidate_document_pool,
    )
    selected_sectors = select_sectors(
        sector_scores,
        threshold=config.sector_threshold,
        fallback_margin=config.fallback_sector_margin,
    )
    routed_documents = discover_documents(
        hits,
        selected_sectors,
        limit=config.candidate_document_pool,
    )

    graph_started = perf_counter()
    retriever = AdaptiveGraphRetriever(
        chunks=index.chunks.values(),
        relationships=index.relationships,
        similarity=index.similarity,
        config=config.traversal,
    )
    graph_setup_ms = (perf_counter() - graph_started) * 1000
    seed_groups = _group_hits(hits)
    graph_similarity = lambda _query, chunk: index.similarity_to_vector(
        query_vector,
        chunk,
    )

    started = perf_counter()
    adaptive_chunks = _expand(
        retriever,
        plan,
        unfiltered_documents,
        seed_groups,
        {},
        graph_similarity,
        index,
    )
    add_result(
        "C: adaptive graph",
        adaptive_chunks,
        started,
        graph_setup_ms,
    )

    started = perf_counter()
    routed_chunks = _expand(
        retriever,
        plan,
        routed_documents,
        seed_groups,
        {},
        graph_similarity,
        index,
    )
    add_result(
        "D: sector + graph",
        routed_chunks,
        started,
        graph_setup_ms,
    )

    started = perf_counter()
    final_result = pipeline.query(
        query,
        query_vector=query_vector,
        global_hits=hits,
    )
    add_result(
        "E: document reranking",
        final_result.context,
        started,
        graph_setup_ms,
    )
    return tuple(results)


def _fixed_window(
    index: InMemoryIndex,
    hits: tuple[SearchHit, ...],
    window: int = 1,
) -> tuple[Chunk, ...]:
    by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in index.chunks.values():
        by_document[chunk.document_id].append(chunk)
    positions = {}
    for document_chunks in by_document.values():
        document_chunks.sort(key=lambda chunk: chunk.sequence_number)
        positions.update(
            (chunk.id, (document_chunks, position))
            for position, chunk in enumerate(document_chunks)
        )

    selected: dict[str, Chunk] = {}
    for hit in hits:
        document_chunks, position = positions[hit.chunk.id]
        start = max(0, position - window)
        stop = min(len(document_chunks), position + window + 1)
        for chunk in document_chunks[start:stop]:
            selected.setdefault(chunk.id, chunk)
    return tuple(selected.values())


def _group_hits(hits: tuple[SearchHit, ...]) -> dict[str, list[SearchHit]]:
    grouped: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.chunk.document_id].append(hit)
    return grouped


def _expand(
    retriever: AdaptiveGraphRetriever,
    plan: QueryPlan,
    document_ids: tuple[str, ...],
    seed_groups: dict[str, list[SearchHit]],
    package_cache: dict[str, EvidencePackage],
    similarity: Callable[[str, Chunk], float],
    index: InMemoryIndex,
) -> tuple[Chunk, ...]:
    selected = {}
    for document_id in document_ids:
        package = package_cache.get(document_id)
        if package is None:
            package = retriever.retrieve(
                document_id,
                plan,
                seed_groups.get(document_id, ()),
                similarity=similarity,
            )
            package_cache[document_id] = package
        for chunk_id in package.included_chunk_ids:
            selected.setdefault(chunk_id, index.chunks[chunk_id])
    return tuple(selected.values())


def _document_ids(chunks: tuple[Chunk, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(chunk.document_id for chunk in chunks))

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from heapq import nlargest, nsmallest

from contextweave.models import (
    DocumentAssessment,
    EvidencePackage,
    QueryPlan,
    SearchHit,
)


def score_sectors(hits: Iterable[SearchHit]) -> dict[str, float]:
    by_sector: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        by_sector[hit.chunk.sector_id].append(hit)
    scores = {}
    for sector_id, sector_hits in by_sector.items():
        top = nsmallest(5, sector_hits, key=lambda hit: (-hit.similarity, hit.rank))
        peak = top[0].similarity
        strong_average = sum(hit.similarity for hit in top) / len(top)
        document_coverage = len({hit.chunk.document_id for hit in top})
        coverage_bonus = min(document_coverage / 4, 1.0)
        scores[sector_id] = 0.55 * peak + 0.35 * strong_average + 0.10 * coverage_bonus
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def select_sectors(
    scores: dict[str, float],
    threshold: float = 0.75,
    fallback_margin: float = 0.10,
) -> tuple[str, ...]:
    selected = [sector for sector, score in scores.items() if score >= threshold]
    near_threshold = [
        sector for sector, score in scores.items()
        if threshold - fallback_margin <= score < threshold
    ]
    if not selected and scores:
        selected = [next(iter(scores))]
    selected.extend(sector for sector in near_threshold if sector not in selected)
    return tuple(selected)


def discover_documents(
    hits: Iterable[SearchHit],
    selected_sector_ids: Iterable[str],
    limit: int = 5,
) -> tuple[str, ...]:
    sectors = set(selected_sector_ids)
    by_document: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        if hit.chunk.sector_id in sectors:
            by_document[hit.chunk.document_id].append(hit)
    ranked = []
    for document_id, document_hits in by_document.items():
        strongest = nsmallest(
            4,
            document_hits,
            key=lambda hit: (-hit.similarity, hit.rank),
        )
        peak = strongest[0].similarity
        average = sum(hit.similarity for hit in strongest) / len(strongest)
        sections = len({hit.chunk.section_id for hit in strongest if hit.chunk.section_id})
        score = 0.65 * peak + 0.25 * average + 0.10 * min(sections / 3, 1.0)
        ranked.append((score, document_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(document_id for _, document_id in ranked[:limit])


def rerank_documents(
    packages: Iterable[EvidencePackage],
    query_plan: QueryPlan,
    target_documents: int = 3,
    max_documents: int = 5,
    minimum_marginal_score: float = 0.18,
) -> tuple[tuple[str, ...], tuple[DocumentAssessment, ...]]:
    """Select complete document evidence packages after all expansions are ready."""
    package_list = list(packages)
    if not package_list:
        return (), ()

    package_by_id = {package.document_id: package for package in package_list}
    all_requirements = {requirement.id for requirement in query_plan.requirements}
    relevance = {
        package.document_id: (
            0.65 * max(package.similarities.values(), default=0.0)
            + 0.35 * (
                sum(nlargest(3, package.similarities.values()))
                / max(1, min(3, len(package.similarities)))
            )
        )
        for package in package_list
    }

    selected: list[str] = []
    covered: set[str] = set()
    assessments: dict[str, DocumentAssessment] = {}
    while len(selected) < min(max_documents, len(package_list)):
        remaining = [doc_id for doc_id in package_by_id if doc_id not in selected]
        if not remaining:
            break
        ranked_candidates = []
        for document_id in remaining:
            package = package_by_id[document_id]
            new_coverage = package.supported_requirements - covered
            marginal_coverage = len(new_coverage) / max(1, len(all_requirements))
            score = 0.55 * marginal_coverage + 0.45 * relevance[document_id]
            ranked_candidates.append((score, document_id, new_coverage, marginal_coverage))
        ranked_candidates.sort(key=lambda item: (-item[0], item[1]))
        score, document_id, new_coverage, marginal_coverage = ranked_candidates[0]
        if selected and not new_coverage and (
            covered >= all_requirements or score < minimum_marginal_score
        ):
            break
        selected.append(document_id)
        covered.update(package_by_id[document_id].supported_requirements)
        assessments[document_id] = DocumentAssessment(
            document_id=document_id,
            score=score,
            relevance_score=relevance[document_id],
            coverage_score=marginal_coverage,
            supported_requirements=package_by_id[document_id].supported_requirements,
            selected=True,
            explanation=(
                f"Selected with relevance {relevance[document_id]:.3f}; "
                f"added {len(new_coverage)} previously uncovered requirement(s)."
            ),
        )
        if covered >= all_requirements:
            break

    for document_id, package in package_by_id.items():
        if document_id in assessments:
            continue
        missing = package.supported_requirements - covered
        assessments[document_id] = DocumentAssessment(
            document_id=document_id,
            score=0.45 * relevance[document_id],
            relevance_score=relevance[document_id],
            coverage_score=len(missing) / max(1, len(all_requirements)),
            supported_requirements=package.supported_requirements,
            selected=False,
            explanation=(
                "Excluded as redundant or lower-value after selected evidence covered "
                "the requirements."
            ),
        )
    return tuple(selected), tuple(assessments[package.document_id] for package in package_list)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RelationshipType(StrEnum):
    PREVIOUS = "PREVIOUS"
    NEXT = "NEXT"
    PARENT = "PARENT"
    CHILD = "CHILD"
    QUALIFIES = "QUALIFIES"
    REFERENCES = "REFERENCES"


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    sector_id: str
    revision_id: str = "1"


@dataclass(frozen=True, slots=True)
class Sector:
    id: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    sector_id: str
    content: str
    sequence_number: int
    embedding: tuple[float, ...] = ()
    section_id: str = ""
    heading_path: tuple[str, ...] = ()
    source_location: str = ""
    chunk_type: str = "paragraph"
    token_count: int = 0
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.token_count == 0:
            object.__setattr__(self, "token_count", len(self.content.split()))


@dataclass(frozen=True, slots=True)
class Relationship:
    source_chunk_id: str
    target_chunk_id: str
    relationship_type: RelationshipType
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryRequirement:
    id: str
    description: str
    terms: frozenset[str]


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_query: str
    intent: str
    requirements: tuple[QueryRequirement, ...]
    asks_exceptions: bool
    is_multi_part: bool


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    similarity: float
    rank: int


@dataclass(frozen=True, slots=True)
class TraversalPath:
    chunk_ids: tuple[str, ...]
    relationship_types: tuple[RelationshipType, ...]


@dataclass(frozen=True, slots=True)
class EvidencePackage:
    document_id: str
    seed_chunk_ids: tuple[str, ...]
    visited_chunk_ids: tuple[str, ...]
    included_chunk_ids: tuple[str, ...]
    similarities: dict[str, float]
    paths: tuple[TraversalPath, ...]
    relationship_types_followed: tuple[RelationshipType, ...]
    supported_requirements: frozenset[str]
    token_count: int
    explanation: str
    requirement_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentAssessment:
    document_id: str
    score: float
    relevance_score: float
    coverage_score: float
    supported_requirements: frozenset[str]
    selected: bool
    explanation: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query_plan: QueryPlan
    global_hits: tuple[SearchHit, ...]
    sector_scores: dict[str, float]
    selected_sector_ids: tuple[str, ...]
    candidate_document_ids: tuple[str, ...]
    evidence_packages: tuple[EvidencePackage, ...]
    document_assessments: tuple[DocumentAssessment, ...]
    selected_document_ids: tuple[str, ...]
    context: tuple[Chunk, ...]
    evidence_coverage: dict[str, str]
    expansion_order: tuple[str, ...]
    fallback_used: bool = False

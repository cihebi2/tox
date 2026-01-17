from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceQuery:
    sequence: str
    length: int
    model_id: str | None = None
    p_toxic: float | None = None
    uncertainty: float | None = None


@dataclass(frozen=True)
class EvidenceHit:
    hit_id: str
    sequence: str
    label: int | None = None
    similarity: float | None = None

    identity: float | None = None
    coverage: float | None = None
    evalue: float | None = None

    species: str | None = None
    protein_name: str | None = None

    source_db: str = "internal"
    evidence_level: str | None = None
    annotations: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvidenceSummary:
    top_k: int
    num_hits: int
    max_similarity: float | None
    mean_similarity: float | None
    pos_rate: float | None
    has_toxprot: bool | None = None


@dataclass(frozen=True)
class EvidenceReport:
    timestamp: str
    query: EvidenceQuery
    hits: list[EvidenceHit]
    summary: EvidenceSummary
    meta: dict[str, Any] | None = None


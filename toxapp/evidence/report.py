from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from ..data import sanitize_sequence
from ..retrieval import RetrievalIndex, query as query_tfidf
from .schema import EvidenceHit, EvidenceQuery, EvidenceReport, EvidenceSummary


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def build_internal_evidence_report(
    index: RetrievalIndex,
    sequence: str,
    *,
    top_k: int = 16,
    model_id: str | None = None,
    p_toxic: float | None = None,
    uncertainty: float | None = None,
) -> EvidenceReport:
    seq = sanitize_sequence(sequence)
    raw_hits = query_tfidf(index, seq, top_k=int(top_k))

    hits: list[EvidenceHit] = []
    for i, h in enumerate(raw_hits, start=1):
        hits.append(
            EvidenceHit(
                hit_id=f"internal_{i}",
                sequence=str(h.get("sequence", "")),
                label=int(h["label"]) if h.get("label") is not None else None,
                similarity=float(h["similarity"]) if h.get("similarity") is not None else None,
                source_db="internal",
            )
        )

    sims = [h.similarity for h in hits if h.similarity is not None]
    labs = [h.label for h in hits if h.label is not None]
    max_sim = max(sims) if sims else None
    mean_sim = (sum(sims) / len(sims)) if sims else None
    pos_rate = (sum(1 for y in labs if int(y) == 1) / len(labs)) if labs else None

    summary = EvidenceSummary(
        top_k=int(top_k),
        num_hits=int(len(hits)),
        max_similarity=float(max_sim) if max_sim is not None else None,
        mean_similarity=float(mean_sim) if mean_sim is not None else None,
        pos_rate=float(pos_rate) if pos_rate is not None else None,
        has_toxprot=None,
    )

    report = EvidenceReport(
        timestamp=now_timestamp(),
        query=EvidenceQuery(
            sequence=seq,
            length=int(len(seq)),
            model_id=model_id,
            p_toxic=p_toxic,
            uncertainty=uncertainty,
        ),
        hits=hits,
        summary=summary,
        meta={"backend": "internal_tfidf", "index_type": "char_ngram_tfidf"},
    )
    return report


def report_to_dict(report: EvidenceReport) -> dict:
    return asdict(report)


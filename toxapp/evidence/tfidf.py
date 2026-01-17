from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from ..data import sanitize_sequence
from ..retrieval import RetrievalIndex, build_retrieval_index
from .features import compute_evidence_features


@dataclass(frozen=True)
class TfidfEvidenceConfig:
    top_k: int = 32
    ngram_range: tuple[int, int] = (3, 5)
    min_df: int = 2
    batch_size: int = 256
    exclude_self: bool = True


def build_internal_tfidf_index(
    sequences: list[str],
    labels: list[int],
    *,
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
) -> RetrievalIndex:
    return build_retrieval_index(
        sequences,
        labels,
        ngram_range=ngram_range,
        min_df=min_df,
    )


def compute_tfidf_evidence_feature_matrix(
    index: RetrievalIndex,
    sequences: list[str],
    *,
    cfg: TfidfEvidenceConfig,
) -> np.ndarray:
    """
    Returns: [N, F] float32 feature matrix.
    Evidence source: internal labeled sequence DB (RetrievalIndex).
    """
    seqs = [sanitize_sequence(s) for s in sequences]
    n = len(seqs)
    if n == 0:
        return np.zeros((0, 12), dtype=np.float32)

    top_k = int(cfg.top_k)
    if top_k <= 0:
        raise ValueError("cfg.top_k must be > 0.")

    labels_arr = np.asarray(index.labels, dtype=np.int64)
    seq_to_indices: dict[str, list[int]] = {}
    for i, s in enumerate(index.sequences):
        seq_to_indices.setdefault(s, []).append(int(i))

    out = np.zeros((n, 12), dtype=np.float32)

    for start in range(0, n, int(cfg.batch_size)):
        batch_seqs = seqs[start : start + int(cfg.batch_size)]
        q = index.vectorizer.transform(batch_seqs)
        sims = linear_kernel(q, index.matrix)  # [B, Ndb]
        sims = np.asarray(sims, dtype=np.float64)

        if bool(cfg.exclude_self):
            for bi, s in enumerate(batch_seqs):
                idx = seq_to_indices.get(s)
                if idx:
                    sims[bi, idx] = -1e18

        k = min(top_k, sims.shape[1])
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(top_idx.shape[0])[:, None]
        top_scores = sims[rows, top_idx]
        order = np.argsort(-top_scores, axis=1)
        top_idx = top_idx[rows, order]
        top_scores = top_scores[rows, order]
        top_labels = labels_arr[top_idx]

        for bi in range(top_idx.shape[0]):
            out[start + bi] = compute_evidence_features(top_scores[bi], top_labels[bi])

        if (start // int(cfg.batch_size)) % 50 == 0:
            print(f"  tfidf evidence: {start}/{n}")

    return out


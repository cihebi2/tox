from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np


@dataclass(frozen=True)
class EmbeddingRetrievalIndex:
    embeddings: np.ndarray  # [N, D] float32, L2-normalized
    sequences: list[str]
    labels: list[int]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def build_embedding_index(embeddings: np.ndarray, sequences: list[str], labels: list[int]) -> EmbeddingRetrievalIndex:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D [N, D].")
    if len(sequences) != embeddings.shape[0] or len(labels) != embeddings.shape[0]:
        raise ValueError("Length mismatch between embeddings, sequences and labels.")
    emb = embeddings.astype(np.float32, copy=False)
    emb = _l2_normalize(emb)
    return EmbeddingRetrievalIndex(embeddings=emb, sequences=list(sequences), labels=list(map(int, labels)))


def save_embedding_index(index: EmbeddingRetrievalIndex, path: str) -> None:
    joblib.dump(index, path)


def load_embedding_index(path: str) -> EmbeddingRetrievalIndex:
    return joblib.load(path)


def query_embedding(index: EmbeddingRetrievalIndex, query_emb: np.ndarray, *, top_k: int = 8) -> list[dict]:
    if top_k <= 0:
        return []
    q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
    q = _l2_normalize(q)
    sims = (q @ index.embeddings.T).ravel()
    top_k = min(int(top_k), int(sims.shape[0]))
    idx = np.argpartition(-sims, top_k - 1)[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    out: list[dict] = []
    for i in idx.tolist():
        out.append(
            {
                "sequence": index.sequences[int(i)],
                "label": int(index.labels[int(i)]),
                "similarity": float(sims[int(i)]),
            }
        )
    return out


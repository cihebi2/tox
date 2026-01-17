from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .data import sanitize_sequence


@dataclass(frozen=True)
class RetrievalIndex:
    vectorizer: TfidfVectorizer
    matrix: object  # scipy sparse
    sequences: list[str]
    labels: list[int]


def build_retrieval_index(
    sequences: list[str],
    labels: list[int],
    *,
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
) -> RetrievalIndex:
    sequences = [sanitize_sequence(s) for s in sequences]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, lowercase=False, min_df=min_df)
    matrix = vectorizer.fit_transform(sequences)
    return RetrievalIndex(vectorizer=vectorizer, matrix=matrix, sequences=sequences, labels=list(map(int, labels)))


def save_retrieval_index(index: RetrievalIndex, path: str) -> None:
    joblib.dump(index, path)


def load_retrieval_index(path: str) -> RetrievalIndex:
    return joblib.load(path)


def query(index: RetrievalIndex, sequence: str, *, top_k: int = 8) -> list[dict]:
    sequence = sanitize_sequence(sequence)
    q = index.vectorizer.transform([sequence])
    sims = linear_kernel(q, index.matrix).ravel()

    if top_k <= 0:
        return []
    top_idx = np.argpartition(-sims, min(top_k, len(sims)) - 1)[:top_k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    results = []
    for i in top_idx:
        results.append(
            {
                "sequence": index.sequences[int(i)],
                "label": int(index.labels[int(i)]),
                "similarity": float(sims[int(i)]),
            }
        )
    return results


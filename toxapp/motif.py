from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .data import sanitize_sequence


@dataclass(frozen=True)
class MotifMiningConfig:
    ngram_min: int = 3
    ngram_max: int = 5
    min_df: int = 5
    top_n: int = 512
    alpha: float = 0.5  # add-alpha smoothing for log-odds


def _to_clean_list(seqs: Iterable[str]) -> list[str]:
    return [sanitize_sequence(s) for s in seqs]


def mine_discriminative_motifs(
    sequences: Sequence[str],
    labels: Sequence[int],
    cfg: MotifMiningConfig = MotifMiningConfig(),
) -> pd.DataFrame:
    """
    Mine discriminative k-mer motifs (3-5mer by default) using class log-odds.

    Returns a DataFrame with columns:
      motif, k, count_pos, count_neg, log_odds, direction
    where direction is 'toxic' if log_odds>0 else 'non_toxic'.
    """
    if len(sequences) != len(labels):
        raise ValueError("Sequences and labels length mismatch.")

    seqs = _to_clean_list(sequences)
    y = np.asarray([int(v) for v in labels], dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("labels must be 1D.")

    ngram_range = (int(cfg.ngram_min), int(cfg.ngram_max))
    if ngram_range[0] <= 0 or ngram_range[1] < ngram_range[0]:
        raise ValueError("Invalid ngram range.")

    vec = CountVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        lowercase=False,
        min_df=int(cfg.min_df),
    )
    X = vec.fit_transform(seqs)  # [N, V] sparse counts
    vocab = np.asarray(vec.get_feature_names_out(), dtype=object)
    if X.shape[1] == 0:
        raise RuntimeError("No motifs found. Try lowering min_df or adjusting ngram range.")

    pos_mask = y == 1
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        raise ValueError("Both classes must be present to mine motifs.")

    count_pos = np.asarray(X[pos_mask].sum(axis=0)).ravel().astype(np.float64)
    count_neg = np.asarray(X[neg_mask].sum(axis=0)).ravel().astype(np.float64)

    total_pos = float(count_pos.sum())
    total_neg = float(count_neg.sum())
    V = float(X.shape[1])
    alpha = float(cfg.alpha)

    # log-odds with symmetric Dirichlet smoothing
    log_odds = np.log((count_pos + alpha) / (total_pos + alpha * V)) - np.log((count_neg + alpha) / (total_neg + alpha * V))

    top_n = int(cfg.top_n)
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    n_pos = top_n // 2
    n_neg = top_n - n_pos
    idx_pos = np.argsort(-log_odds)[: min(n_pos, log_odds.shape[0])]
    idx_neg = np.argsort(log_odds)[: min(n_neg, log_odds.shape[0])]
    idx = np.unique(np.concatenate([idx_pos, idx_neg], axis=0))

    motifs = vocab[idx].tolist()
    df = pd.DataFrame(
        {
            "motif": motifs,
            "k": [len(m) for m in motifs],
            "count_pos": count_pos[idx].astype(np.int64),
            "count_neg": count_neg[idx].astype(np.int64),
            "log_odds": log_odds[idx].astype(np.float64),
        }
    )
    df["direction"] = np.where(df["log_odds"] > 0, "toxic", "non_toxic")
    df = df.sort_values(by="log_odds", ascending=False, kind="mergesort").reset_index(drop=True)
    return df


def build_motif_vectorizer(motifs: Sequence[str], ngram_min: int, ngram_max: int) -> CountVectorizer:
    motifs = [sanitize_sequence(m) for m in motifs]
    return CountVectorizer(
        analyzer="char",
        ngram_range=(int(ngram_min), int(ngram_max)),
        lowercase=False,
        vocabulary=list(motifs),
    )


def encode_motif_counts(
    vectorizer: CountVectorizer,
    sequences: Sequence[str],
    *,
    normalize_by_length: bool = True,
) -> np.ndarray:
    seqs = _to_clean_list(sequences)
    X = vectorizer.transform(seqs)  # sparse counts [N, M]
    out = X.toarray().astype(np.float32, copy=False)
    if normalize_by_length:
        lengths = np.asarray([max(1, len(s)) for s in seqs], dtype=np.float32).reshape(-1, 1)
        out = out / lengths
    return out


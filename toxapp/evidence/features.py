from __future__ import annotations

import numpy as np


def evidence_feature_names(prefix: str = "evi_topk") -> list[str]:
    return [
        f"{prefix}_mean_sim",
        f"{prefix}_max_sim",
        f"{prefix}_std_sim",
        f"{prefix}_median_sim",
        f"{prefix}_pos_rate",
        f"{prefix}_weighted_pos_rate",
        f"{prefix}_pos_max_sim",
        f"{prefix}_neg_max_sim",
        f"{prefix}_pos_mean_sim",
        f"{prefix}_neg_mean_sim",
        f"{prefix}_pos_count",
        f"{prefix}_neg_count",
    ]


def compute_evidence_features(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    scores: [K] float
    labels: [K] {0,1}
    returns: [12] float32 features
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("scores/labels must be 1D with same length.")
    if scores.shape[0] == 0:
        return np.zeros((12,), dtype=np.float32)

    pos_mask = labels == 1
    neg_mask = ~pos_mask

    mean_sim = float(np.mean(scores))
    max_sim = float(np.max(scores))
    std_sim = float(np.std(scores))
    median_sim = float(np.median(scores))

    pos_rate = float(np.mean(pos_mask.astype(np.float64)))
    w = np.clip(scores, 0.0, None)
    weighted_pos_rate = float(np.sum(w * pos_mask.astype(np.float64)) / max(1e-8, float(np.sum(w))))

    pos_count = int(np.sum(pos_mask))
    neg_count = int(np.sum(neg_mask))

    pos_max = float(np.max(scores[pos_mask])) if pos_count > 0 else 0.0
    neg_max = float(np.max(scores[neg_mask])) if neg_count > 0 else 0.0
    pos_mean = float(np.mean(scores[pos_mask])) if pos_count > 0 else 0.0
    neg_mean = float(np.mean(scores[neg_mask])) if neg_count > 0 else 0.0

    return np.asarray(
        [
            mean_sim,
            max_sim,
            std_sim,
            median_sim,
            pos_rate,
            weighted_pos_rate,
            pos_max,
            neg_max,
            pos_mean,
            neg_mean,
            float(pos_count),
            float(neg_count),
        ],
        dtype=np.float32,
    )


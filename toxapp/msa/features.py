from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import AMINO_ACIDS

AA_LIST = list(AMINO_ACIDS)
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}


@dataclass(frozen=True)
class MsaFeatureConfig:
    min_depth: int = 2
    low_entropy_thr: float = 0.3  # normalized entropy in [0,1]


def msa_summary_feature_names(prefix: str = "msa") -> list[str]:
    return [
        f"{prefix}_depth",
        f"{prefix}_depth_log1p",
        f"{prefix}_mean_gap_rate",
        f"{prefix}_mean_non_gap_rate",
        f"{prefix}_mean_entropy",
        f"{prefix}_median_entropy",
        f"{prefix}_min_entropy",
        f"{prefix}_max_entropy",
        f"{prefix}_frac_low_entropy",
    ]


def profile_frequencies(aligned_seqs: list[str]) -> np.ndarray:
    """
    aligned_seqs: list[str] with equal length, composed of AA + '-'
    returns: [L, 20] float32 frequencies over non-gap residues per position.
    """
    if not aligned_seqs:
        return np.zeros((0, 20), dtype=np.float32)
    L = len(aligned_seqs[0])
    depth = len(aligned_seqs)
    counts = np.zeros((L, 20), dtype=np.float64)
    nongap = np.zeros((L,), dtype=np.float64)

    for s in aligned_seqs:
        if len(s) != L:
            raise ValueError("Aligned sequences must have equal length.")
        for i, ch in enumerate(s):
            if ch == "-":
                continue
            idx = AA_TO_IDX.get(ch)
            if idx is None:
                continue
            counts[i, idx] += 1.0
            nongap[i] += 1.0

    denom = np.maximum(1.0, nongap)[:, None]
    freq = counts / denom
    return freq.astype(np.float32, copy=False)


def _entropy_from_freq(freq: np.ndarray) -> np.ndarray:
    # freq: [L, 20]
    p = np.clip(freq.astype(np.float64, copy=False), 1e-12, 1.0)
    h = -np.sum(p * np.log(p), axis=1)  # nats
    h_max = np.log(20.0)
    return (h / max(1e-12, h_max)).astype(np.float32, copy=False)  # normalized to [0,1]


def msa_summary_features(aligned_seqs: list[str], *, cfg: MsaFeatureConfig | None = None) -> np.ndarray:
    """
    aligned_seqs: list[str] with equal length, composed of AA + '-'
    returns: [9] float32
    """
    cfg = cfg or MsaFeatureConfig()
    if not aligned_seqs or len(aligned_seqs) < int(cfg.min_depth):
        return np.zeros((9,), dtype=np.float32)

    L = len(aligned_seqs[0])
    depth = len(aligned_seqs)
    gap_counts = np.zeros((L,), dtype=np.float64)
    for s in aligned_seqs:
        if len(s) != L:
            raise ValueError("Aligned sequences must have equal length.")
        gap_counts += (np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8) == ord("-")).astype(np.float64)

    gap_rate = gap_counts / max(1.0, float(depth))
    mean_gap = float(np.mean(gap_rate))
    mean_non_gap = float(1.0 - mean_gap)

    freq = profile_frequencies(aligned_seqs)
    ent = _entropy_from_freq(freq)

    mean_ent = float(np.mean(ent))
    med_ent = float(np.median(ent))
    min_ent = float(np.min(ent))
    max_ent = float(np.max(ent))
    frac_low = float(np.mean((ent <= float(cfg.low_entropy_thr)).astype(np.float32)))

    return np.asarray(
        [
            float(depth),
            float(np.log1p(depth)),
            mean_gap,
            mean_non_gap,
            mean_ent,
            med_ent,
            min_ent,
            max_ent,
            frac_low,
        ],
        dtype=np.float32,
    )


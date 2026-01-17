from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import AMINO_ACIDS
from .data import sanitize_sequence


@dataclass(frozen=True)
class PhyschemFeatureSpec:
    name: str
    dim: int


# Kyte-Doolittle hydrophobicity scale (commonly used; higher = more hydrophobic)
_KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# Average residue molecular weight (Da) (approx; good enough for global features)
_RESIDUE_MW = {
    "A": 89.09,
    "C": 121.16,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}

# Simple charge proxy at physiological pH (~7.4)
_CHARGE = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,  # partially protonated
}

_AROMATIC = set("FWY")
_POLAR = set("STNQHYC")  # a coarse set, for global fraction features
_POS = set("KRH")
_NEG = set("DE")


def global_feature_names() -> list[str]:
    """
    Global physchem features for a sequence.
    Returns a stable, documented order of feature names.
    """
    names: list[str] = [
        "length",
        "log1p_length",
        "net_charge",
        "mean_charge",
        "mean_hydrophobicity_kd",
        "std_hydrophobicity_kd",
        "mean_residue_mw",
        "std_residue_mw",
        "frac_aromatic",
        "frac_polar",
        "frac_pos",
        "frac_neg",
        "frac_hydrophobic_kd_pos",
    ]
    names.extend([f"aa_frac_{aa}" for aa in AMINO_ACIDS])
    return names


def global_physchem_features(seq: str) -> np.ndarray:
    """
    Compute lightweight, structure-free global physchem features from sequence only.

    Output shape: [F] float32 where F=len(global_feature_names()).
    """
    seq = sanitize_sequence(seq)
    L = len(seq)
    if L <= 0:
        raise ValueError("Empty sequence after sanitization.")

    charges = np.zeros((L,), dtype=np.float32)
    hydro = np.zeros((L,), dtype=np.float32)
    mw = np.zeros((L,), dtype=np.float32)

    aromatic = 0
    polar = 0
    pos = 0
    neg = 0
    hydro_pos = 0

    for i, aa in enumerate(seq):
        c = float(_CHARGE.get(aa, 0.0))
        h = float(_KYTE_DOOLITTLE[aa])
        m = float(_RESIDUE_MW[aa])

        charges[i] = c
        hydro[i] = h
        mw[i] = m

        if aa in _AROMATIC:
            aromatic += 1
        if aa in _POLAR:
            polar += 1
        if aa in _POS:
            pos += 1
        if aa in _NEG:
            neg += 1
        if h > 0:
            hydro_pos += 1

    comp = np.zeros((len(AMINO_ACIDS),), dtype=np.float32)
    for j, aa in enumerate(AMINO_ACIDS):
        comp[j] = float(seq.count(aa)) / float(L)

    feat = np.concatenate(
        [
            np.asarray(
                [
                    float(L),
                    float(math.log1p(L)),
                    float(charges.sum()),
                    float(charges.mean()),
                    float(hydro.mean()),
                    float(hydro.std(ddof=0)),
                    float(mw.mean()),
                    float(mw.std(ddof=0)),
                    float(aromatic / L),
                    float(polar / L),
                    float(pos / L),
                    float(neg / L),
                    float(hydro_pos / L),
                ],
                dtype=np.float32,
            ),
            comp,
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    expected = len(global_feature_names())
    if feat.shape[0] != expected:
        raise RuntimeError(f"Physchem feature dim mismatch: got {feat.shape[0]} expected {expected}")
    return feat


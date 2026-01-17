from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import AA_TO_ID, PAD_ID


def load_sequences_labels(csv_path: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(csv_path)
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must contain columns 'sequence' and 'label': {csv_path}")
    sequences = df["sequence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return sequences, labels


def sanitize_sequence(seq: str) -> str:
    seq = "".join(seq.split()).upper()
    if not seq:
        raise ValueError("Empty sequence.")
    invalid = sorted({ch for ch in seq if ch not in AA_TO_ID})
    if invalid:
        raise ValueError(f"Invalid amino acids: {''.join(invalid)}")
    return seq


def encode_sequence(seq: str) -> list[int]:
    seq = sanitize_sequence(seq)
    return [AA_TO_ID[ch] for ch in seq]


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[str], labels: Sequence[int]):
        if len(sequences) != len(labels):
            raise ValueError("Sequences and labels length mismatch.")
        self.sequences = list(sequences)
        self.labels = list(labels)

    def __len__(self) -> int:  # noqa: D401
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.sequences[idx], int(self.labels[idx])


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor  # [B, L]
    attention_mask: torch.Tensor  # [B, L] bool
    labels: torch.Tensor  # [B]
    sequences: list[str]


def collate_batch(examples: Iterable[tuple[str, int]]) -> Batch:
    sequences, labels = zip(*examples)
    encoded = [encode_sequence(s) for s in sequences]
    max_len = max(len(x) for x in encoded)

    input_ids = torch.full((len(encoded), max_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = True

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=torch.tensor(labels, dtype=torch.long),
        sequences=list(sequences),
    )


@dataclass(frozen=True)
class RawBatch:
    labels: torch.Tensor  # [B]
    sequences: list[str]


def collate_raw_batch(examples: Iterable[tuple[str, int]]) -> RawBatch:
    sequences, labels = zip(*examples)
    return RawBatch(labels=torch.tensor(labels, dtype=torch.long), sequences=list(sequences))

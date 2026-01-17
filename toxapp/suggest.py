from __future__ import annotations

from dataclasses import dataclass

from .constants import AMINO_ACIDS
from .data import sanitize_sequence


@dataclass(frozen=True)
class Mutation:
    position_1based: int
    from_aa: str
    to_aa: str
    mutant: str


def generate_neighbor_single_mutations(seq: str, neighbor: str) -> list[Mutation]:
    seq = sanitize_sequence(seq)
    neighbor = sanitize_sequence(neighbor)
    if len(seq) != len(neighbor):
        return []
    muts: list[Mutation] = []
    for i, (a, b) in enumerate(zip(seq, neighbor), start=1):
        if a != b:
            mutant = seq[: i - 1] + b + seq[i:]
            muts.append(Mutation(position_1based=i, from_aa=a, to_aa=b, mutant=mutant))
    return muts


def generate_all_single_mutations(seq: str) -> list[Mutation]:
    seq = sanitize_sequence(seq)
    muts: list[Mutation] = []
    for i, a in enumerate(seq, start=1):
        for b in AMINO_ACIDS:
            if b == a:
                continue
            mutant = seq[: i - 1] + b + seq[i:]
            muts.append(Mutation(position_1based=i, from_aa=a, to_aa=b, mutant=mutant))
    return muts


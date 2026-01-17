from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..constants import AMINO_ACIDS

AA_SET = set(AMINO_ACIDS)


@dataclass(frozen=True)
class A3mReadConfig:
    max_sequences: int = 2048
    drop_invalid_length: bool = True


def _strip_insertions(seq: str) -> str:
    # In A3M, lower-case letters are insertions w.r.t query; remove them.
    out = []
    for ch in seq:
        if "a" <= ch <= "z":
            continue
        if ch == ".":
            out.append("-")
            continue
        if ch == "*":
            continue
        ch = ch.upper()
        if ch in AA_SET or ch == "-":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


def read_a3m(path: str | Path, *, cfg: A3mReadConfig | None = None) -> list[str]:
    """
    Return aligned sequences (upper-case + '-' only) with equal length.
    """
    cfg = cfg or A3mReadConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    records: list[str] = []
    cur: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    records.append(_strip_insertions("".join(cur)))
                    cur = []
                    if len(records) >= int(cfg.max_sequences):
                        break
                continue
            cur.append(line)

        if cur and len(records) < int(cfg.max_sequences):
            records.append(_strip_insertions("".join(cur)))

    if not records:
        return []

    L = len(records[0])
    if L == 0:
        return []

    if bool(cfg.drop_invalid_length):
        records = [s for s in records if len(s) == L]

    return records


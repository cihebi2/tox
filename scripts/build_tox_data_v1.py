from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


AA_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
AA_AMBIGUOUS_TO_X = {
    "U": "X",
    "Z": "X",
    "O": "X",
    "B": "X",
    "J": "X",
}
AA_ALLOWED = AA_STANDARD | {"X"}


def normalize_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq.strip().upper())
    # remove non-letter tokens commonly seen in FASTA/alignments
    seq = seq.replace("-", "").replace("*", "")
    seq = re.sub(r"[^A-Z]", "", seq)
    seq = "".join(AA_AMBIGUOUS_TO_X.get(ch, ch) for ch in seq)
    return seq


def is_valid_sequence(seq: str) -> bool:
    return len(seq) > 0 and all(ch in AA_ALLOWED for ch in seq)


def iter_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    header: Optional[str] = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SourceSpec:
    dataset_id: str
    project: str
    level: str  # peptide/protein
    split: str
    file_path: Path
    label_rule: str


def load_toxgin_csv(spec: SourceSpec) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(spec.file_path)
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Unexpected columns in {spec.file_path}: {df.columns.tolist()}")

    records: List[Dict[str, Any]] = []
    stats = {
        "dataset_id": spec.dataset_id,
        "project": spec.project,
        "level": spec.level,
        "split": spec.split,
        "file_path": str(spec.file_path),
        "label_rule": spec.label_rule,
        "total_rows": int(len(df)),
        "kept_rows": 0,
        "dropped_invalid_seq": 0,
        "normalized_changed": 0,
        "label_counts": {"0": 0, "1": 0},
    }

    for i, row in df.iterrows():
        raw_seq = str(row["sequence"])
        seq = normalize_sequence(raw_seq)
        if seq != raw_seq.strip().upper():
            stats["normalized_changed"] += 1
        if not is_valid_sequence(seq):
            stats["dropped_invalid_seq"] += 1
            continue

        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Invalid label {label} in {spec.file_path} row {i}")

        stats["label_counts"][str(label)] += 1
        records.append(
            {
                "sequence": seq,
                "label": label,
                "source_dataset": spec.dataset_id,
                "source_project": spec.project,
                "source_level": spec.level,
                "source_split": spec.split,
                "source_file": str(spec.file_path),
                "source_id": f"row_{i}",
            }
        )

    stats["kept_rows"] = len(records)
    return pd.DataFrame(records), stats


def load_toxipep_fasta(spec: SourceSpec) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "dataset_id": spec.dataset_id,
        "project": spec.project,
        "level": spec.level,
        "split": spec.split,
        "file_path": str(spec.file_path),
        "label_rule": spec.label_rule,
        "total_records": 0,
        "kept_records": 0,
        "dropped_unknown_label": 0,
        "dropped_invalid_seq": 0,
        "normalized_changed": 0,
        "label_counts": {"0": 0, "1": 0},
    }

    for header, raw_seq in iter_fasta(spec.file_path):
        stats["total_records"] += 1
        h = header.strip()
        h_lower = h.lower()
        if h_lower.startswith("pos"):
            label = 1
        elif h_lower.startswith("neg"):
            label = 0
        else:
            stats["dropped_unknown_label"] += 1
            continue

        seq = normalize_sequence(raw_seq)
        if seq != raw_seq.strip().upper():
            stats["normalized_changed"] += 1
        if not is_valid_sequence(seq):
            stats["dropped_invalid_seq"] += 1
            continue

        stats["label_counts"][str(label)] += 1
        records.append(
            {
                "sequence": seq,
                "label": label,
                "source_dataset": spec.dataset_id,
                "source_project": spec.project,
                "source_level": spec.level,
                "source_split": spec.split,
                "source_file": str(spec.file_path),
                "source_id": h,
            }
        )

    stats["kept_records"] = len(records)
    return pd.DataFrame(records), stats


def load_toxmsrc_fasta_with_header_label(spec: SourceSpec) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "dataset_id": spec.dataset_id,
        "project": spec.project,
        "level": spec.level,
        "split": spec.split,
        "file_path": str(spec.file_path),
        "label_rule": spec.label_rule,
        "total_records": 0,
        "kept_records": 0,
        "dropped_unknown_label": 0,
        "dropped_invalid_seq": 0,
        "normalized_changed": 0,
        "label_counts": {"0": 0, "1": 0},
    }

    for header, raw_seq in iter_fasta(spec.file_path):
        stats["total_records"] += 1
        h = header.strip()

        label: Optional[int] = None
        if "|" in h:
            maybe = h.split("|")[-1].strip()
            if maybe in {"0", "1"}:
                label = int(maybe)
        if label is None:
            stats["dropped_unknown_label"] += 1
            continue

        seq = normalize_sequence(raw_seq)
        if seq != raw_seq.strip().upper():
            stats["normalized_changed"] += 1
        if not is_valid_sequence(seq):
            stats["dropped_invalid_seq"] += 1
            continue

        stats["label_counts"][str(label)] += 1
        records.append(
            {
                "sequence": seq,
                "label": label,
                "source_dataset": spec.dataset_id,
                "source_project": spec.project,
                "source_level": spec.level,
                "source_split": spec.split,
                "source_file": str(spec.file_path),
                "source_id": h,
            }
        )

    stats["kept_records"] = len(records)
    return pd.DataFrame(records), stats


def load_hypeptox_fuse_fasta(spec: SourceSpec, fixed_label: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "dataset_id": spec.dataset_id,
        "project": spec.project,
        "level": spec.level,
        "split": spec.split,
        "file_path": str(spec.file_path),
        "label_rule": spec.label_rule,
        "fixed_label": fixed_label,
        "total_records": 0,
        "kept_records": 0,
        "dropped_invalid_seq": 0,
        "normalized_changed": 0,
        "label_counts": {"0": 0, "1": 0},
    }

    for header, raw_seq in iter_fasta(spec.file_path):
        stats["total_records"] += 1
        seq = normalize_sequence(raw_seq)
        if seq != raw_seq.strip().upper():
            stats["normalized_changed"] += 1
        if not is_valid_sequence(seq):
            stats["dropped_invalid_seq"] += 1
            continue

        stats["label_counts"][str(fixed_label)] += 1
        records.append(
            {
                "sequence": seq,
                "label": fixed_label,
                "source_dataset": spec.dataset_id,
                "source_project": spec.project,
                "source_level": spec.level,
                "source_split": spec.split,
                "source_file": str(spec.file_path),
                "source_id": header.strip(),
            }
        )

    stats["kept_records"] = len(records)
    return pd.DataFrame(records), stats


def load_toxdl2_protein_fasta(spec: SourceSpec) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "dataset_id": spec.dataset_id,
        "project": spec.project,
        "level": spec.level,
        "split": spec.split,
        "file_path": str(spec.file_path),
        "label_rule": spec.label_rule,
        "total_records": 0,
        "kept_records": 0,
        "dropped_unknown_label": 0,
        "dropped_invalid_seq": 0,
        "normalized_changed": 0,
        "label_counts": {"0": 0, "1": 0},
    }

    for header, raw_seq in iter_fasta(spec.file_path):
        stats["total_records"] += 1
        # Header example: "B1NWR6\t\t1" -> id=B1NWR6, label=1
        header_tokens = header.strip().split()
        if not header_tokens:
            stats["dropped_unknown_label"] += 1
            continue
        maybe_label = header_tokens[-1]
        if maybe_label not in {"0", "1"}:
            stats["dropped_unknown_label"] += 1
            continue
        label = int(maybe_label)
        rec_id = header_tokens[0]

        seq = normalize_sequence(raw_seq)
        if seq != raw_seq.strip().upper():
            stats["normalized_changed"] += 1
        if not is_valid_sequence(seq):
            stats["dropped_invalid_seq"] += 1
            continue

        stats["label_counts"][str(label)] += 1
        records.append(
            {
                "sequence": seq,
                "label": label,
                "source_dataset": spec.dataset_id,
                "source_project": spec.project,
                "source_level": spec.level,
                "source_split": spec.split,
                "source_file": str(spec.file_path),
                "source_id": rec_id,
            }
        )

    stats["kept_records"] = len(records)
    return pd.DataFrame(records), stats


def build_dedup(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Returns:
      - resolved_df: unique sequences with a single label
      - conflict_df: unique sequences with multiple labels
      - provenance_jsonl: list of dicts (one per unique sequence) for JSONL output
    """
    resolved_rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []
    provenance_rows: List[Dict[str, Any]] = []

    grouped = df.groupby("sequence", sort=False)
    for sequence, g in grouped:
        labels = sorted({int(x) for x in g["label"].tolist()})
        sources = g[
            [
                "source_project",
                "source_dataset",
                "source_split",
                "source_file",
                "source_id",
            ]
        ].to_dict(orient="records")
        source_datasets = sorted({x["source_dataset"] for x in sources})
        source_projects = sorted({x["source_project"] for x in sources})

        provenance_rows.append(
            {
                "sequence": sequence,
                "labels": labels,
                "sources": sources,
            }
        )

        base_row = {
            "sequence": sequence,
            "labels": json.dumps(labels, ensure_ascii=False),
            "n_records": int(len(g)),
            "n_source_datasets": int(len(source_datasets)),
            "source_datasets": ";".join(source_datasets),
            "source_projects": ";".join(source_projects),
        }

        if len(labels) == 1:
            resolved_rows.append(
                {
                    **base_row,
                    "label": labels[0],
                }
            )
        else:
            conflict_rows.append(base_row)

    resolved_df = pd.DataFrame(resolved_rows)
    conflict_df = pd.DataFrame(conflict_rows)

    if not resolved_df.empty:
        resolved_df = resolved_df.sort_values(
            by=["n_source_datasets", "n_records"], ascending=False
        )
    else:
        resolved_df = pd.DataFrame(
            columns=[
                "sequence",
                "labels",
                "label",
                "n_records",
                "n_source_datasets",
                "source_datasets",
                "source_projects",
            ]
        )

    if not conflict_df.empty:
        conflict_df = conflict_df.sort_values(
            by=["n_source_datasets", "n_records"], ascending=False
        )
    else:
        conflict_df = pd.DataFrame(
            columns=[
                "sequence",
                "labels",
                "n_records",
                "n_source_datasets",
                "source_datasets",
                "source_projects",
            ]
        )
    return resolved_df, conflict_df, provenance_rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_catalog(path: Path, sources: List[SourceSpec]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_id",
                "project",
                "level",
                "split",
                "file_path",
                "label_rule",
            ],
        )
        writer.writeheader()
        for s in sources:
            writer.writerow(
                {
                    "dataset_id": s.dataset_id,
                    "project": s.project,
                    "level": s.level,
                    "split": s.split,
                    "file_path": str(s.file_path),
                    "label_rule": s.label_rule,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build toxicity dataset v1 (merge + dedup + provenance).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/toxicity_data_v1",
        help="Output directory inside this repo.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    ensure_dir(out_dir)

    # -------- Sources (peptides) --------
    peptide_sources: List[SourceSpec] = [
        SourceSpec(
            dataset_id="toxgin_train",
            project="ToxGIN",
            level="peptide",
            split="train",
            file_path=Path("/root/private_data/dd_model/ToxGIN/train_sequence.csv"),
            label_rule="CSV column `label` (1=toxic, 0=non-toxic).",
        ),
        SourceSpec(
            dataset_id="toxgin_test",
            project="ToxGIN",
            level="peptide",
            split="test",
            file_path=Path("/root/private_data/dd_model/ToxGIN/test_sequence.csv"),
            label_rule="CSV column `label` (1=toxic, 0=non-toxic).",
        ),
        SourceSpec(
            dataset_id="toxipep_0.9_train",
            project="ToxiPep",
            level="peptide",
            split="train",
            file_path=Path("/root/private_data/dd_model/ToxiPep/Dataset/0.9/train.txt"),
            label_rule="FASTA header prefix: `>pos*` => 1, `>neg*` => 0.",
        ),
        SourceSpec(
            dataset_id="toxipep_0.9_test",
            project="ToxiPep",
            level="peptide",
            split="test",
            file_path=Path("/root/private_data/dd_model/ToxiPep/Dataset/0.9/test.txt"),
            label_rule="FASTA header prefix: `>pos*` => 1, `>neg*` => 0.",
        ),
        SourceSpec(
            dataset_id="toxipep_0.8_train",
            project="ToxiPep",
            level="peptide",
            split="train",
            file_path=Path("/root/private_data/dd_model/ToxiPep/Dataset/0.8/train.txt"),
            label_rule="FASTA header prefix: `>pos*` => 1, `>neg*` => 0.",
        ),
        SourceSpec(
            dataset_id="toxipep_0.8_test",
            project="ToxiPep",
            level="peptide",
            split="test",
            file_path=Path("/root/private_data/dd_model/ToxiPep/Dataset/0.8/test.txt"),
            label_rule="FASTA header prefix: `>pos*` => 1, `>neg*` => 0.",
        ),
        SourceSpec(
            dataset_id="toxipep_independent",
            project="ToxiPep",
            level="peptide",
            split="independent",
            file_path=Path("/root/private_data/dd_model/ToxiPep/Dataset/Independent set/independent test set.txt"),
            label_rule="FASTA header prefix: `>pos*` => 1, `>neg*` => 0.",
        ),
        SourceSpec(
            dataset_id="toxmsrc_train",
            project="ToxMSRC",
            level="peptide",
            split="train",
            file_path=Path("/root/private_data/dd_model/ToxMSRC/Raw data/train_data.fasta"),
            label_rule="FASTA header suffix: `|1` => 1, `|0` => 0.",
        ),
        SourceSpec(
            dataset_id="toxmsrc_test1",
            project="ToxMSRC",
            level="peptide",
            split="test1",
            file_path=Path("/root/private_data/dd_model/ToxMSRC/Raw data/test1.fasta"),
            label_rule="FASTA header suffix: `|1` => 1, `|0` => 0.",
        ),
        SourceSpec(
            dataset_id="toxmsrc_test2",
            project="ToxMSRC",
            level="peptide",
            split="test2",
            file_path=Path("/root/private_data/dd_model/ToxMSRC/Raw data/test2.fasta"),
            label_rule="FASTA header suffix: `|1` => 1, `|0` => 0.",
        ),
        SourceSpec(
            dataset_id="hypeptoxfuse_train_pos",
            project="HyPepTox-Fuse",
            level="peptide",
            split="train_pos",
            file_path=Path("/root/private_data/dd_model/toxin/HyPepTox-Fuse/raw_dataset/train_pos.fa"),
            label_rule="File-level label: `train_pos.fa` => 1.",
        ),
        SourceSpec(
            dataset_id="hypeptoxfuse_train_neg",
            project="HyPepTox-Fuse",
            level="peptide",
            split="train_neg",
            file_path=Path("/root/private_data/dd_model/toxin/HyPepTox-Fuse/raw_dataset/train_neg.fa"),
            label_rule="File-level label: `train_neg.fa` => 0.",
        ),
        SourceSpec(
            dataset_id="hypeptoxfuse_test_pos",
            project="HyPepTox-Fuse",
            level="peptide",
            split="test_pos",
            file_path=Path("/root/private_data/dd_model/toxin/HyPepTox-Fuse/raw_dataset/test_pos.fa"),
            label_rule="File-level label: `test_pos.fa` => 1.",
        ),
        SourceSpec(
            dataset_id="hypeptoxfuse_test_neg",
            project="HyPepTox-Fuse",
            level="peptide",
            split="test_neg",
            file_path=Path("/root/private_data/dd_model/toxin/HyPepTox-Fuse/raw_dataset/test_neg.fa"),
            label_rule="File-level label: `test_neg.fa` => 0.",
        ),
    ]

    # -------- Sources (proteins; kept separate) --------
    protein_sources: List[SourceSpec] = [
        SourceSpec(
            dataset_id="toxdl2_train",
            project="ToxDL2",
            level="protein",
            split="train",
            file_path=Path("/root/private_data/dd_model/ToxDL2/data/protein_sequences/train.fasta"),
            label_rule="FASTA header last token is label (0/1).",
        ),
        SourceSpec(
            dataset_id="toxdl2_valid",
            project="ToxDL2",
            level="protein",
            split="valid",
            file_path=Path("/root/private_data/dd_model/ToxDL2/data/protein_sequences/valid.fasta"),
            label_rule="FASTA header last token is label (0/1).",
        ),
        SourceSpec(
            dataset_id="toxdl2_test",
            project="ToxDL2",
            level="protein",
            split="test",
            file_path=Path("/root/private_data/dd_model/ToxDL2/data/protein_sequences/test.fasta"),
            label_rule="FASTA header last token is label (0/1).",
        ),
        SourceSpec(
            dataset_id="toxdl2_independent",
            project="ToxDL2",
            level="protein",
            split="independent",
            file_path=Path("/root/private_data/dd_model/ToxDL2/data/protein_sequences/independent.fasta"),
            label_rule="FASTA header last token is label (0/1).",
        ),
    ]

    write_catalog(out_dir / "sources_catalog.csv", peptide_sources + protein_sources)

    # -------- Build peptide raw --------
    peptide_frames: List[pd.DataFrame] = []
    peptide_stats: List[Dict[str, Any]] = []
    for spec in peptide_sources:
        if spec.project == "ToxGIN":
            df, st = load_toxgin_csv(spec)
        elif spec.project == "ToxiPep":
            df, st = load_toxipep_fasta(spec)
        elif spec.project == "ToxMSRC":
            df, st = load_toxmsrc_fasta_with_header_label(spec)
        elif spec.project == "HyPepTox-Fuse":
            fixed = 1 if "pos" in spec.dataset_id else 0
            df, st = load_hypeptox_fuse_fasta(spec, fixed_label=fixed)
        else:
            raise ValueError(f"Unknown peptide project: {spec.project}")
        peptide_frames.append(df)
        peptide_stats.append(st)

    peptide_raw = pd.concat(peptide_frames, ignore_index=True)
    ensure_dir(out_dir / "peptide")
    peptide_raw.to_csv(out_dir / "peptide" / "all_raw.csv", index=False)

    peptide_resolved, peptide_conflicts, peptide_prov = build_dedup(peptide_raw)
    peptide_resolved.to_csv(out_dir / "peptide" / "dedup_resolved.csv", index=False)
    peptide_conflicts.to_csv(out_dir / "peptide" / "dedup_conflicts.csv", index=False)
    write_jsonl(out_dir / "peptide" / "provenance.jsonl", peptide_prov)

    peptide_overall_stats: Dict[str, Any] = {
        "level": "peptide",
        "n_raw_records": int(len(peptide_raw)),
        "n_raw_unique_sequences": int(peptide_raw["sequence"].nunique()),
        "n_dedup_resolved": int(len(peptide_resolved)),
        "n_dedup_conflicts": int(len(peptide_conflicts)),
        "n_duplicates_collapsed": int(len(peptide_raw) - peptide_raw["sequence"].nunique()),
        "n_cross_dataset_overlap_sequences": int(
            (peptide_raw.groupby("sequence")["source_dataset"].nunique() > 1).sum()
        ),
        "per_source": peptide_stats,
    }
    (out_dir / "peptide" / "stats.json").write_text(
        json.dumps(peptide_overall_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # -------- Build protein raw (ToxDL2), kept separate --------
    protein_frames: List[pd.DataFrame] = []
    protein_stats: List[Dict[str, Any]] = []
    for spec in protein_sources:
        df, st = load_toxdl2_protein_fasta(spec)
        protein_frames.append(df)
        protein_stats.append(st)

    protein_raw = pd.concat(protein_frames, ignore_index=True)
    ensure_dir(out_dir / "protein")
    protein_raw.to_csv(out_dir / "protein" / "all_raw.csv", index=False)

    protein_resolved, protein_conflicts, protein_prov = build_dedup(protein_raw)
    protein_resolved.to_csv(out_dir / "protein" / "dedup_resolved.csv", index=False)
    protein_conflicts.to_csv(out_dir / "protein" / "dedup_conflicts.csv", index=False)
    write_jsonl(out_dir / "protein" / "provenance.jsonl", protein_prov)

    protein_overall_stats: Dict[str, Any] = {
        "level": "protein",
        "n_raw_records": int(len(protein_raw)),
        "n_raw_unique_sequences": int(protein_raw["sequence"].nunique()),
        "n_dedup_resolved": int(len(protein_resolved)),
        "n_dedup_conflicts": int(len(protein_conflicts)),
        "n_duplicates_collapsed": int(len(protein_raw) - protein_raw["sequence"].nunique()),
        "n_cross_dataset_overlap_sequences": int(
            (protein_raw.groupby("sequence")["source_dataset"].nunique() > 1).sum()
        ),
        "per_source": protein_stats,
    }
    (out_dir / "protein" / "stats.json").write_text(
        json.dumps(protein_overall_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Wrote dataset to: {out_dir}")
    print(f"[Peptide] raw={len(peptide_raw)} unique={peptide_raw['sequence'].nunique()} resolved={len(peptide_resolved)} conflicts={len(peptide_conflicts)}")
    print(f"[Protein] raw={len(protein_raw)} unique={protein_raw['sequence'].nunique()} resolved={len(protein_resolved)} conflicts={len(protein_conflicts)}")


if __name__ == "__main__":
    main()

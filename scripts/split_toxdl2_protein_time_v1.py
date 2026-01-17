from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_source_datasets(value: str) -> set[str]:
    toks = []
    for tok in str(value).split(";"):
        tok = tok.strip()
        if tok:
            toks.append(tok)
    return set(toks)


def assign_split(source_datasets: set[str], priority: list[tuple[str, str]]) -> str:
    for split_name, dataset_id in priority:
        if dataset_id in source_datasets:
            return split_name
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/toxicity_data_v1/protein/dedup_resolved.csv",
        help="Protein-level dedup CSV produced by build_tox_data_v1.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/toxicity_data_v1/splits/protein_toxdl2_time_v1",
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = now_timestamp()
    df = pd.read_csv(in_path)
    required = {"sequence", "label", "source_datasets"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {in_path}: {sorted(missing)}")

    priority: list[tuple[str, str]] = [
        ("independent", "toxdl2_independent"),
        ("test", "toxdl2_test"),
        ("val", "toxdl2_valid"),
        ("train", "toxdl2_train"),
    ]

    df["source_datasets_set"] = df["source_datasets"].astype(str).map(parse_source_datasets)
    df["split"] = df["source_datasets_set"].map(lambda s: assign_split(s, priority))

    # Sanity: expect all rows to be from ToxDL2 splits.
    unknown = df[df["split"] == "unknown"]
    if not unknown.empty:
        example = unknown.head(3)[["sequence", "source_datasets"]].to_dict(orient="records")
        raise ValueError(f"Found {len(unknown)} rows with unknown split. Example: {example}")

    # Ensure no overlap across the derived splits (sequence is unique already).
    split_counts = (
        df.groupby(["split", "label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["split", "label"])
        .to_dict(orient="records")
    )

    # Export per split.
    for split in ["train", "val", "test", "independent"]:
        out_path = out_dir / f"protein_toxdl2_{split}.csv"
        df[df["split"] == split][["sequence", "label"]].to_csv(out_path, index=False)

    # Export full with provenance fields for audit/debug.
    audit_cols = [
        "sequence",
        "label",
        "split",
        "n_records",
        "n_source_datasets",
        "source_datasets",
        "source_projects",
    ]
    keep = [c for c in audit_cols if c in df.columns]
    (out_dir / "protein_toxdl2_all_with_split.csv").write_text(
        df[keep].to_csv(index=False), encoding="utf-8"
    )

    manifest = {
        "timestamp": ts,
        "input_csv": str(in_path),
        "input_sha256": sha256_file(in_path),
        "priority": [{"split": s, "dataset_id": d} for s, d in priority],
        "split_label_counts": split_counts,
        "total_rows": int(df.shape[0]),
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print("timestamp:", ts)
    print("wrote:", out_dir)
    for row in split_counts:
        print(row)


if __name__ == "__main__":
    main()


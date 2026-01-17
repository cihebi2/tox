from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_timestamp() -> str:
    # Keep consistent with other docs: "YYYY-MM-DD HH:MM:SS +0800"
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def stratified_train_val_test_split(
    df: pd.DataFrame,
    label_col: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-9:
        raise ValueError("train/val/test sizes must sum to 1.0")

    indices = df.index
    labels = df[label_col]

    train_idx, tmp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=labels,
    )

    tmp = df.loc[tmp_idx]
    val_ratio_in_tmp = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        tmp.index,
        test_size=(1.0 - val_ratio_in_tmp),
        random_state=seed,
        stratify=tmp[label_col],
    )

    return pd.Index(train_idx), pd.Index(val_idx), pd.Index(test_idx)


def counts_by_split(df: pd.DataFrame, split_col: str, label_col: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split, g in df.groupby(split_col):
        vc = g[label_col].value_counts().to_dict()
        out[str(split)] = {str(k): int(v) for k, v in vc.items()}
        out[str(split)]["total"] = int(len(g))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Split toxicity dataset into train/val/test (stratified).")
    parser.add_argument(
        "--id90-csv",
        type=str,
        default="data/toxicity_data_v1/peptide_id90/dedup_id90_resolved.csv",
        help="Input: 90% similarity dedup representatives.",
    )
    parser.add_argument(
        "--members-csv",
        type=str,
        default="data/toxicity_data_v1/peptide_id90/cluster_members.csv",
        help="Input: mapping from exact-dedup sequences to id90 clusters.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/toxicity_data_v1/splits/peptide_id90_seed69_80_10_10",
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=69, help="Random seed.")
    parser.add_argument("--train-size", type=float, default=0.8, help="Train proportion.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation proportion.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test proportion.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    id90_csv = (repo_root / args.id90_csv).resolve()
    members_csv = (repo_root / args.members_csv).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    ensure_dir(out_dir)

    reps = pd.read_csv(id90_csv)
    if "cluster_id" not in reps.columns or "label" not in reps.columns:
        raise ValueError(f"Unexpected columns in {id90_csv}: {reps.columns.tolist()}")

    # Split on cluster representatives (each cluster_id is unique row)
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        reps, label_col="label", train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )

    reps = reps.copy()
    reps.loc[train_idx, "split"] = "train"
    reps.loc[val_idx, "split"] = "val"
    reps.loc[test_idx, "split"] = "test"

    # Cluster split file
    cluster_split = reps[["cluster_id", "label", "cluster_size", "split"]].copy()
    cluster_split.to_csv(out_dir / "clusters_split.csv", index=False)

    # Representative split files
    for split in ["train", "val", "test"]:
        reps[reps["split"] == split].drop(columns=["split"]).to_csv(out_dir / f"peptide_id90_{split}.csv", index=False)

    # Propagate split to exact-dedup sequences (still unique sequences, but not similarity-deduped)
    members = pd.read_csv(members_csv)
    if "cluster_id" not in members.columns or "sequence" not in members.columns or "label" not in members.columns:
        raise ValueError(f"Unexpected columns in {members_csv}: {members.columns.tolist()}")

    members = members.merge(cluster_split[["cluster_id", "split"]], on="cluster_id", how="left", validate="many_to_one")
    if members["split"].isna().any():
        raise RuntimeError("Some sequences could not be assigned a split; check cluster_id consistency.")

    members_out = members.drop(columns=["seq_id"])
    members_out.to_csv(out_dir / "peptide_exact_by_cluster.csv", index=False)
    for split in ["train", "val", "test"]:
        members_out[members_out["split"] == split].to_csv(out_dir / f"peptide_exact_{split}.csv", index=False)

    # Stats
    stats = {
        "timestamp": now_timestamp(),
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_size),
            "val": float(args.val_size),
            "test": float(args.test_size),
        },
        "inputs": {
            "id90_csv": str(id90_csv),
            "members_csv": str(members_csv),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "rep_counts_by_split": counts_by_split(reps, split_col="split", label_col="label"),
            "exact_counts_by_split": counts_by_split(members_out, split_col="split", label_col="label"),
        },
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] wrote splits to: {out_dir}")
    print("[rep]", stats["outputs"]["rep_counts_by_split"])
    print("[exact]", stats["outputs"]["exact_counts_by_split"])


if __name__ == "__main__":
    main()


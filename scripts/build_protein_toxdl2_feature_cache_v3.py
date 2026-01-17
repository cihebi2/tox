from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.data import load_sequences_labels, sanitize_sequence  # noqa: E402
from toxapp.msa.a3m import A3mReadConfig, read_a3m  # noqa: E402
from toxapp.msa.features import MsaFeatureConfig, msa_summary_feature_names, msa_summary_features  # noqa: E402


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seq_id_sha1(seq: str, *, n: int = 16) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[: int(n)]


def load_split(csv_path: str) -> tuple[list[str], list[int]]:
    seqs, y = load_sequences_labels(csv_path)
    seqs = [sanitize_sequence(s) for s in seqs]
    y = [int(v) for v in y]
    return seqs, y


def _load_base_cache(base_cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plm = np.load(base_cache_dir / f"plm_{split}.npy")
    phys = np.load(base_cache_dir / f"phys_{split}.npy")
    motif = np.load(base_cache_dir / f"motif_{split}.npy")
    return plm, phys, motif


def compute_msa_summary_matrix(
    seqs: list[str],
    *,
    msa_dir: Path | None,
    a3m_cfg: A3mReadConfig,
    feat_cfg: MsaFeatureConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    feats = np.zeros((len(seqs), len(msa_summary_feature_names())), dtype=np.float32)
    stats = {"total": int(len(seqs)), "found": 0, "missing": 0, "empty": 0, "invalid": 0}

    if msa_dir is None:
        stats["missing"] = int(len(seqs))
        return feats, stats

    for i, s in enumerate(seqs):
        sid = seq_id_sha1(s)
        path = msa_dir / f"{sid}.a3m"
        if not path.exists():
            stats["missing"] += 1
            continue

        try:
            aligned = read_a3m(path, cfg=a3m_cfg)
        except Exception:
            stats["invalid"] += 1
            continue

        if not aligned:
            stats["empty"] += 1
            continue

        feats[i] = msa_summary_features(aligned, cfg=feat_cfg)
        stats["found"] += 1

        if (i + 1) % 2000 == 0:
            print(f"  msa summary: {i+1}/{len(seqs)} found={stats['found']} missing={stats['missing']}")

    return feats, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache-dir", type=str, default="data/feature_cache_v2/protein_toxdl2_v2")
    parser.add_argument("--out-dir", type=str, default="data/feature_cache_v3/protein_toxdl2_v3")
    parser.add_argument("--msa-dir", type=str, default="", help="Optional dir containing <sha1[:16]>.a3m per sequence")

    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/toxicity_data_v1/splits/protein_toxdl2_time_v1/protein_toxdl2_train.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/toxicity_data_v1/splits/protein_toxdl2_time_v1/protein_toxdl2_val.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/toxicity_data_v1/splits/protein_toxdl2_time_v1/protein_toxdl2_test.csv",
    )
    parser.add_argument(
        "--independent-csv",
        type=str,
        default="data/toxicity_data_v1/splits/protein_toxdl2_time_v1/protein_toxdl2_independent.csv",
    )

    parser.add_argument("--a3m-max-seqs", type=int, default=2048)
    parser.add_argument("--a3m-keep-bad-length", action="store_true")
    parser.add_argument("--msa-min-depth", type=int, default=2)
    parser.add_argument("--msa-low-entropy-thr", type=float, default=0.3)
    args = parser.parse_args()

    base_cache_dir = Path(args.base_cache_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    msa_dir = Path(args.msa_dir) if str(args.msa_dir).strip() else None
    if msa_dir is not None:
        ensure_dir(msa_dir)

    print("timestamp:", now_timestamp())
    print("base_cache_dir:", base_cache_dir)
    print("out_dir:", out_dir)
    print("msa_dir:", str(msa_dir) if msa_dir is not None else "(none)")

    splits: Dict[str, Tuple[list[str], list[int], str]] = {
        "train": (*load_split(args.train_csv), args.train_csv),
        "val": (*load_split(args.val_csv), args.val_csv),
        "test": (*load_split(args.test_csv), args.test_csv),
        "independent": (*load_split(args.independent_csv), args.independent_csv),
    }

    a3m_cfg = A3mReadConfig(
        max_sequences=int(args.a3m_max_seqs),
        drop_invalid_length=not bool(args.a3m_keep_bad_length),
    )
    feat_cfg = MsaFeatureConfig(
        min_depth=int(args.msa_min_depth),
        low_entropy_thr=float(args.msa_low_entropy_thr),
    )
    msa_names = msa_summary_feature_names()
    (out_dir / "msa_feature_names.json").write_text(json.dumps(msa_names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    msa_stats: dict[str, dict[str, int]] = {}
    for split, (seqs, y, csv_path) in splits.items():
        print(f"\n=== split: {split} ===")
        print("n=", len(seqs), "pos=", int(sum(int(v) for v in y)))
        print("csv:", csv_path)

        plm, phys, motif = _load_base_cache(base_cache_dir, split)
        if plm.shape[0] != len(seqs) or phys.shape[0] != len(seqs) or motif.shape[0] != len(seqs):
            raise ValueError(f"base cache row mismatch for split={split}")

        np.save(out_dir / f"plm_{split}.npy", plm)
        np.save(out_dir / f"phys_{split}.npy", phys)
        np.save(out_dir / f"motif_{split}.npy", motif)

        seq_ids = [seq_id_sha1(s) for s in seqs]
        (out_dir / f"seq_ids_{split}.txt").write_text("\n".join(seq_ids) + "\n", encoding="utf-8")

        print("computing msa summary features...")
        msa_x, stats = compute_msa_summary_matrix(seqs, msa_dir=msa_dir, a3m_cfg=a3m_cfg, feat_cfg=feat_cfg)
        np.save(out_dir / f"msa_{split}.npy", msa_x)
        msa_stats[split] = stats
        print("msa stats:", stats)

    meta = {
        "timestamp": now_timestamp(),
        "base_cache_dir": str(base_cache_dir),
        "out_dir": str(out_dir),
        "msa": {
            "msa_dir": str(msa_dir) if msa_dir is not None else "",
            "a3m_read_cfg": {"max_sequences": int(a3m_cfg.max_sequences), "drop_invalid_length": bool(a3m_cfg.drop_invalid_length)},
            "msa_feat_cfg": {"min_depth": int(feat_cfg.min_depth), "low_entropy_thr": float(feat_cfg.low_entropy_thr)},
            "feature_names": str(out_dir / "msa_feature_names.json"),
            "stats": msa_stats,
        },
        "splits": {k: {"csv": v[2], "n": len(v[0]), "pos": int(sum(int(x) for x in v[1]))} for k, v in splits.items()},
        "outputs": {
            "msa_train": str(out_dir / "msa_train.npy"),
            "msa_val": str(out_dir / "msa_val.npy"),
            "msa_test": str(out_dir / "msa_test.npy"),
            "msa_independent": str(out_dir / "msa_independent.npy"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nsaved:", out_dir / "meta.json")


if __name__ == "__main__":
    main()


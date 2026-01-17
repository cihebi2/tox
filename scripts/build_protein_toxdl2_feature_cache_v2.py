from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.data import load_sequences_labels, sanitize_sequence  # noqa: E402
from toxapp.motif import MotifMiningConfig, build_motif_vectorizer, encode_motif_counts, mine_discriminative_motifs  # noqa: E402
from toxapp.physchem import global_feature_names, global_physchem_features  # noqa: E402
from toxapp.plm import EsmFeaturizer, EsmFeaturizerConfig  # noqa: E402


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_split(csv_path: str) -> tuple[list[str], list[int]]:
    seqs, y = load_sequences_labels(csv_path)
    seqs = [sanitize_sequence(s) for s in seqs]
    y = [int(v) for v in y]
    return seqs, y


def compute_physchem_matrix(seqs: list[str]) -> np.ndarray:
    feats = [global_physchem_features(s) for s in seqs]
    return np.stack(feats, axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def compute_plm_pool_matrix(
    featurizer: EsmFeaturizer,
    seqs: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    n = len(seqs)
    d = int(featurizer.hidden_size)
    out = np.zeros((n, d), dtype=np.float16)
    for start in range(0, n, int(batch_size)):
        batch = seqs[start : start + int(batch_size)]
        emb = featurizer.encode(batch)  # [B, D] float32 on device
        out[start : start + len(batch), :] = emb.detach().cpu().numpy().astype(np.float16, copy=False)
        if (start // int(batch_size)) % 50 == 0:
            print(f"  plm pooled: {start}/{n}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plm-path", type=str, default="/root/group_data/qiuleyu/esm2_t12_35M_UR50D")
    parser.add_argument("--out-dir", type=str, default="data/feature_cache_v2/protein_toxdl2_v2")

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

    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--motif-min-df", type=int, default=5)
    parser.add_argument("--motif-top-n", type=int, default=512)
    parser.add_argument("--motif-alpha", type=float, default=0.5)

    parser.add_argument("--plm-batch-size", type=int, default=32)
    parser.add_argument("--chunk-batch-size", type=int, default=8)
    parser.add_argument("--max-residues", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("timestamp:", now_timestamp())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    splits: Dict[str, Tuple[list[str], list[int], str]] = {
        "train": (*_load_split(args.train_csv), args.train_csv),
        "val": (*_load_split(args.val_csv), args.val_csv),
        "test": (*_load_split(args.test_csv), args.test_csv),
        "independent": (*_load_split(args.independent_csv), args.independent_csv),
    }

    # 1) Motif mining on train
    train_seqs, train_y, _ = splits["train"]
    mining_cfg = MotifMiningConfig(
        ngram_min=int(args.ngram_min),
        ngram_max=int(args.ngram_max),
        min_df=int(args.motif_min_df),
        top_n=int(args.motif_top_n),
        alpha=float(args.motif_alpha),
    )
    print("mining motifs:", mining_cfg)
    motif_df = mine_discriminative_motifs(train_seqs, train_y, mining_cfg)
    motif_path = out_dir / "motifs.tsv"
    motif_df.to_csv(motif_path, sep="\t", index=False)
    motifs = motif_df["motif"].astype(str).tolist()
    print("saved motifs:", motif_path, "n=", len(motifs))

    motif_vec = build_motif_vectorizer(motifs, mining_cfg.ngram_min, mining_cfg.ngram_max)

    # 2) Physchem + motif features per split
    phys_names = global_feature_names()
    (out_dir / "physchem_feature_names.json").write_text(json.dumps(phys_names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for split, (seqs, y, csv_path) in splits.items():
        print(f"\n=== split: {split} ===")
        print("n=", len(seqs), "pos=", int(sum(int(v) for v in y)))
        print("csv:", csv_path)

        print("computing physchem...")
        phys = compute_physchem_matrix(seqs)
        np.save(out_dir / f"phys_{split}.npy", phys)

        print("computing motif...")
        motif_x = encode_motif_counts(motif_vec, seqs, normalize_by_length=True)
        np.save(out_dir / f"motif_{split}.npy", motif_x.astype(np.float32, copy=False))

    # 3) PLM pooled embedding per split (cached, float16)
    print("\nloading PLM:", args.plm_path)
    featurizer = EsmFeaturizer(
        EsmFeaturizerConfig(
            model_path=args.plm_path,
            max_residues=int(args.max_residues),
            stride=int(args.stride),
            chunk_batch_size=int(args.chunk_batch_size),
            use_amp=(not bool(args.no_amp)),
            device=str(device),
            freeze_plm=True,
        )
    )

    for split, (seqs, _y, _csv_path) in splits.items():
        print(f"\n=== split: {split} PLM pooled ===")
        plm = compute_plm_pool_matrix(featurizer, seqs, batch_size=int(args.plm_batch_size))
        np.save(out_dir / f"plm_{split}.npy", plm)

    meta = {
        "timestamp": now_timestamp(),
        "device": str(device),
        "plm": {
            "model_path": args.plm_path,
            "max_residues": int(args.max_residues),
            "stride": int(args.stride),
            "chunk_batch_size": int(args.chunk_batch_size),
            "use_amp": bool(not args.no_amp),
            "hidden_size": int(featurizer.hidden_size),
        },
        "motif_mining": {
            "ngram_min": int(mining_cfg.ngram_min),
            "ngram_max": int(mining_cfg.ngram_max),
            "min_df": int(mining_cfg.min_df),
            "top_n": int(mining_cfg.top_n),
            "alpha": float(mining_cfg.alpha),
            "motifs_tsv": str(motif_path),
        },
        "splits": {k: {"csv": v[2], "n": len(v[0]), "pos": int(sum(int(x) for x in v[1]))} for k, v in splits.items()},
        "outputs": {
            "motifs_tsv": str(out_dir / "motifs.tsv"),
            "phys_feature_names": str(out_dir / "physchem_feature_names.json"),
            "plm_train": str(out_dir / "plm_train.npy"),
            "phys_train": str(out_dir / "phys_train.npy"),
            "motif_train": str(out_dir / "motif_train.npy"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nsaved:", out_dir / "meta.json")


if __name__ == "__main__":
    main()


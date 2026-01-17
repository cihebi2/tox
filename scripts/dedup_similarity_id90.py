from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class CDHitParams:
    identity: float = 0.90
    word_length: int = 2
    threads: int = 0
    memory_mb: int = 0
    min_length: int = 1
    aS: float = 0.90
    aL: float = 0.90
    global_identity: int = 1
    desc_len: int = 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_fasta(path: Path, records: Iterable[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for seq_id, seq in records:
            f.write(f">{seq_id}\n{seq}\n")


def run_cd_hit(in_fa: Path, out_fa: Path, params: CDHitParams) -> None:
    cmd = [
        "cd-hit",
        "-i",
        str(in_fa),
        "-o",
        str(out_fa),
        "-c",
        str(params.identity),
        "-n",
        str(params.word_length),
        "-T",
        str(params.threads),
        "-M",
        str(params.memory_mb),
        "-l",
        str(params.min_length),
        "-aS",
        str(params.aS),
        "-aL",
        str(params.aL),
        "-G",
        str(params.global_identity),
        "-d",
        str(params.desc_len),
    ]
    subprocess.run(cmd, check=True)


def parse_clstr(path: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (seq_id, cluster_id_str)
    """
    pairs: List[Tuple[str, str]] = []
    current_cluster: str | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                current_cluster = line.split()[-1]
                continue
            if current_cluster is None:
                continue
            gt = line.find(">")
            if gt == -1:
                continue
            after = line[gt + 1 :]
            dots = after.find("...")
            if dots == -1:
                seq_id = after.split()[0]
            else:
                seq_id = after[:dots]
            pairs.append((seq_id, current_cluster))
    return pairs


def cluster_stats(member_df: pd.DataFrame) -> Dict[str, int]:
    sizes = member_df.groupby("cluster_id").size()
    return {
        "n_clusters": int(sizes.shape[0]),
        "n_members": int(member_df.shape[0]),
        "cluster_size_min": int(sizes.min()) if not sizes.empty else 0,
        "cluster_size_p50": int(sizes.median()) if not sizes.empty else 0,
        "cluster_size_p95": int(sizes.quantile(0.95)) if not sizes.empty else 0,
        "cluster_size_max": int(sizes.max()) if not sizes.empty else 0,
    }


def build_label_subset(df: pd.DataFrame, label: int, prefix: str) -> pd.DataFrame:
    sub = df[df["label"] == label].copy()
    sub = sub.reset_index(drop=True)
    sub["seq_id"] = [f"{prefix}{i:06d}" for i in range(len(sub))]
    return sub


def write_params(path: Path, params: CDHitParams, extra: Dict[str, str]) -> None:
    payload = {
        "tool": "cd-hit",
        "params": params.__dict__,
        **extra,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def detect_representatives(cluster_members: pd.DataFrame) -> pd.DataFrame:
    # Representative is the first sequence that appears in the .clstr file for that cluster with '*'.
    # We reconstruct this by reading the representative IDs from the out_fa (cd-hit outputs reps in fasta).
    # Here, we approximate by picking the first member encountered per cluster_id ordering in member_df,
    # but we override later using reps_fa parsing.
    return (
        cluster_members.sort_values(["cluster_id"])
        .groupby("cluster_id", as_index=False)
        .first()[["cluster_id", "seq_id"]]
        .rename(columns={"seq_id": "rep_seq_id"})
    )


def parse_fasta_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="90% sequence similarity dedup (CD-HIT) for peptide dataset.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/toxicity_data_v1/peptide/dedup_resolved.csv",
        help="Input CSV (exact-dedup resolved).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/toxicity_data_v1/peptide_id90",
        help="Output directory.",
    )
    parser.add_argument("--threads", type=int, default=0, help="Threads for CD-HIT (0=all).")
    parser.add_argument("--identity", type=float, default=0.90, help="Sequence identity threshold.")
    parser.add_argument("--aS", type=float, default=0.90, help="Alignment coverage for shorter sequence.")
    parser.add_argument("--aL", type=float, default=0.90, help="Alignment coverage for longer sequence.")
    parser.add_argument("--word-length", type=int, default=2, help="CD-HIT word length (-n).")
    parser.add_argument("--min-length", type=int, default=1, help="Min sequence length to keep (-l).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_csv = (repo_root / args.input_csv).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    ensure_dir(out_dir)

    df = pd.read_csv(input_csv)
    required_cols = {"sequence", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")

    params = CDHitParams(
        identity=float(args.identity),
        word_length=int(args.word_length),
        threads=int(args.threads),
        memory_mb=0,
        min_length=int(args.min_length),
        aS=float(args.aS),
        aL=float(args.aL),
        global_identity=1,
        desc_len=0,
    )

    # Save input snapshot
    df.to_csv(out_dir / "input_dedup_resolved.csv", index=False)

    # Build per-label subsets and FASTA
    neg = build_label_subset(df, label=0, prefix="n")
    pos = build_label_subset(df, label=1, prefix="p")

    write_fasta(out_dir / "neg.fa", zip(neg["seq_id"], neg["sequence"]))
    write_fasta(out_dir / "pos.fa", zip(pos["seq_id"], pos["sequence"]))

    neg.to_csv(out_dir / "neg_index.csv", index=False)
    pos.to_csv(out_dir / "pos_index.csv", index=False)

    # Run CD-HIT for each label separately (label-aware dedup)
    neg_out = out_dir / "neg_rep.fa"
    pos_out = out_dir / "pos_rep.fa"
    if not (neg_out.exists() and (neg_out.with_suffix(neg_out.suffix + ".clstr")).exists()):
        run_cd_hit(out_dir / "neg.fa", neg_out, params)
    if not (pos_out.exists() and (pos_out.with_suffix(pos_out.suffix + ".clstr")).exists()):
        run_cd_hit(out_dir / "pos.fa", pos_out, params)

    # Parse clusters
    neg_pairs = parse_clstr(out_dir / "neg_rep.fa.clstr")
    pos_pairs = parse_clstr(out_dir / "pos_rep.fa.clstr")

    neg_members = pd.DataFrame(neg_pairs, columns=["seq_id", "cluster_id"])
    pos_members = pd.DataFrame(pos_pairs, columns=["seq_id", "cluster_id"])

    neg_members = neg_members.merge(neg, on="seq_id", how="left")
    pos_members = pos_members.merge(pos, on="seq_id", how="left")

    # Determine representatives from rep fasta
    neg_rep_ids = set(parse_fasta_ids(out_dir / "neg_rep.fa"))
    pos_rep_ids = set(parse_fasta_ids(out_dir / "pos_rep.fa"))

    def build_rep_map(members: pd.DataFrame, rep_ids: set[str]) -> pd.DataFrame:
        reps = members[members["seq_id"].isin(rep_ids)][["seq_id", "cluster_id", "sequence"]].copy()
        if reps["cluster_id"].duplicated().any():
            # defensive: keep first
            reps = reps.drop_duplicates("cluster_id", keep="first")
        reps = reps.rename(columns={"seq_id": "rep_seq_id", "sequence": "rep_sequence"})
        return reps[["cluster_id", "rep_seq_id", "rep_sequence"]]

    neg_reps = build_rep_map(neg_members, neg_rep_ids)
    pos_reps = build_rep_map(pos_members, pos_rep_ids)

    # Cluster-level aggregation
    def aggregate(members: pd.DataFrame, reps: pd.DataFrame) -> pd.DataFrame:
        members = members.copy()
        members["source_dataset_list"] = members["source_datasets"].astype(str).str.split(";")
        members["source_project_list"] = members["source_projects"].astype(str).str.split(";")
        # cluster stats
        agg = members.groupby("cluster_id").agg(
            label=("label", "first"),
            cluster_size=("seq_id", "size"),
            n_records_sum=("n_records", "sum"),
        )
        # union of source datasets/projects
        src_ds = (
            members.explode("source_dataset_list")
            .groupby("cluster_id")["source_dataset_list"]
            .unique()
            .apply(lambda x: sorted({s for s in x if isinstance(s, str) and s}))
        )
        src_pr = (
            members.explode("source_project_list")
            .groupby("cluster_id")["source_project_list"]
            .unique()
            .apply(lambda x: sorted({s for s in x if isinstance(s, str) and s}))
        )
        agg = agg.join(src_ds.rename("source_datasets_union"))
        agg = agg.join(src_pr.rename("source_projects_union"))
        agg = agg.reset_index()
        agg["n_source_datasets_union"] = agg["source_datasets_union"].apply(len)
        agg["source_datasets_union"] = agg["source_datasets_union"].apply(lambda xs: ";".join(xs))
        agg["source_projects_union"] = agg["source_projects_union"].apply(lambda xs: ";".join(xs))
        agg = agg.merge(reps, on="cluster_id", how="left")
        # prefer rep_sequence as sequence column for downstream training
        agg = agg.rename(columns={"rep_sequence": "sequence"})
        # cluster_id namespace (avoid collision between label clusters)
        return agg[
            [
                "cluster_id",
                "rep_seq_id",
                "sequence",
                "label",
                "cluster_size",
                "n_records_sum",
                "n_source_datasets_union",
                "source_datasets_union",
                "source_projects_union",
            ]
        ]

    neg_agg = aggregate(neg_members, neg_reps)
    pos_agg = aggregate(pos_members, pos_reps)

    # Prefix cluster ids to avoid collision when merging labels
    neg_agg["cluster_id"] = "neg_" + neg_agg["cluster_id"].astype(str)
    pos_agg["cluster_id"] = "pos_" + pos_agg["cluster_id"].astype(str)

    neg_members["cluster_id"] = "neg_" + neg_members["cluster_id"].astype(str)
    pos_members["cluster_id"] = "pos_" + pos_members["cluster_id"].astype(str)

    # Save outputs
    members_all = pd.concat([neg_members, pos_members], ignore_index=True)
    reps_all = pd.concat([neg_agg, pos_agg], ignore_index=True)

    members_all.to_csv(out_dir / "cluster_members.csv", index=False)
    reps_all.to_csv(out_dir / "dedup_id90_resolved.csv", index=False)

    # Write representative FASTA (combined)
    write_fasta(out_dir / "dedup_id90_resolved.fa", zip(reps_all["cluster_id"], reps_all["sequence"]))

    # Stats
    stats = {
        "input_csv": str(input_csv),
        "output_dir": str(out_dir),
        "params": params.__dict__,
        "input": {
            "n_sequences": int(df.shape[0]),
            "label_counts": df["label"].value_counts().to_dict(),
        },
        "neg": cluster_stats(neg_members),
        "pos": cluster_stats(pos_members),
        "output": {
            "n_sequences": int(reps_all.shape[0]),
            "label_counts": reps_all["label"].value_counts().to_dict(),
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Record tool/version
    help_run = subprocess.run(["cd-hit", "-h"], check=False, capture_output=True, text=True)
    help_text = (help_run.stdout or "") + ("\n" + help_run.stderr if help_run.stderr else "")
    first_line = next((ln for ln in help_text.splitlines() if ln.strip()), "")
    write_params(out_dir / "run_manifest.json", params, {"cd_hit_banner": first_line})

    print(f"[OK] wrote: {out_dir}")
    print(f"[Input] {df.shape[0]} seqs -> [Output] {reps_all.shape[0]} seqs")


if __name__ == "__main__":
    main()

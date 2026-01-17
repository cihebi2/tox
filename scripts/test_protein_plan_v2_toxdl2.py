from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.evidential import alpha_to_probs, alpha_total_evidence, alpha_uncertainty, dirichlet_evidence_loss  # noqa: E402
from toxapp.fusion_v2 import EvidentialFusionV2, FusionV2Config  # noqa: E402
from toxapp.retrieval_embed import build_embedding_index, save_embedding_index  # noqa: E402


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int], plm: np.ndarray, phys: np.ndarray, motif: np.ndarray):
        if len(sequences) != len(labels):
            raise ValueError("sequences/labels length mismatch.")
        n = len(sequences)
        if plm.shape[0] != n or phys.shape[0] != n or motif.shape[0] != n:
            raise ValueError("feature matrix row count mismatch with sequences.")
        self.sequences = sequences
        self.labels = np.asarray(labels, dtype=np.int64)
        self.plm = plm
        self.phys = phys
        self.motif = motif

    def __len__(self) -> int:  # noqa: D401
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            "sequence": self.sequences[idx],
            "label": int(self.labels[idx]),
            "plm": self.plm[idx],
            "phys": self.phys[idx],
            "motif": self.motif[idx],
        }


def collate_features(examples: list[dict]) -> dict:
    seqs = [e["sequence"] for e in examples]
    y = torch.tensor([int(e["label"]) for e in examples], dtype=torch.long)
    plm = torch.tensor(np.stack([e["plm"] for e in examples], axis=0), dtype=torch.float32)
    phys = torch.tensor(np.stack([e["phys"] for e in examples], axis=0), dtype=torch.float32)
    motif = torch.tensor(np.stack([e["motif"] for e in examples], axis=0), dtype=torch.float32)
    return {"sequences": seqs, "labels": y, "plm": plm, "phys": phys, "motif": motif}


@torch.no_grad()
def predict_loader(model: EvidentialFusionV2, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    p_toxic: List[float] = []
    uncertainty: List[float] = []
    total_evidence: List[float] = []
    sequences: List[str] = []
    fused_list: List[np.ndarray] = []

    for batch in loader:
        plm = batch["plm"].to(device)
        phys = batch["phys"].to(device)
        motif = batch["motif"].to(device)
        out = model(plm, phys, motif, return_att_maps=False)
        alpha = out["alphas"].detach().cpu()
        probs = alpha_to_probs(alpha).numpy()
        p = probs[:, 1]
        y = batch["labels"].detach().cpu().numpy()

        y_true.extend(y.tolist())
        p_toxic.extend(p.tolist())
        uncertainty.extend(alpha_uncertainty(alpha).numpy().tolist())
        total_evidence.extend(alpha_total_evidence(alpha).numpy().tolist())
        sequences.extend(batch["sequences"])
        fused_list.extend(out["fused"].detach().cpu().numpy())

    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "p_toxic": np.asarray(p_toxic, dtype=np.float64),
        "uncertainty": np.asarray(uncertainty, dtype=np.float64),
        "total_evidence": np.asarray(total_evidence, dtype=np.float64),
        "sequences": np.asarray(sequences, dtype=object),
        "fused": np.asarray(fused_list, dtype=np.float32),
    }


def confusion_at_threshold(y_true: np.ndarray, p_pos: np.ndarray, thr: float) -> Dict[str, int]:
    pred = (p_pos >= float(thr)).astype(np.int64)
    y = y_true.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_at_threshold(y_true: np.ndarray, p_pos: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p_pos >= float(thr)).astype(np.int64)
    y = y_true.astype(np.int64)
    conf = confusion_at_threshold(y, p_pos, thr)
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    bacc = 0.5 * (sens + spec)

    return {
        "thr": float(thr),
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(bacc),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def ranking_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> Dict[str, float]:
    y = y_true.astype(np.int64)
    return {"auroc": float(roc_auc_score(y, p_pos)), "auprc": float(average_precision_score(y, p_pos))}


def select_threshold(y_true: np.ndarray, p_pos: np.ndarray, metric: str = "mcc") -> Dict[str, float]:
    metric = metric.lower()
    if metric not in {"mcc", "f1", "bacc"}:
        raise ValueError("select metric must be one of: mcc, f1, bacc")

    thr = np.unique(np.clip(p_pos.astype(np.float64), 0.0, 1.0))
    thr = np.unique(np.concatenate([thr, np.asarray([0.0, 0.5, 1.0])]))
    thr.sort()

    best_thr = 0.5
    best_val = -1e9
    best_report: Dict[str, float] = {}
    for t in thr:
        m = metrics_at_threshold(y_true, p_pos, float(t))
        v = float(m[metric])
        if v > best_val:
            best_val = v
            best_thr = float(t)
            best_report = m
    return {"metric": metric, "best_thr": float(best_thr), "best": best_report}


def load_split(csv_path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(csv_path)
    seqs = df["sequence"].astype(str).tolist()
    y = df["label"].astype(int).tolist()
    return seqs, y


def load_cache(cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plm = np.load(cache_dir / f"plm_{split}.npy")
    phys = np.load(cache_dir / f"phys_{split}.npy")
    motif = np.load(cache_dir / f"motif_{split}.npy")
    return plm, phys, motif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="data/feature_cache_v2/protein_toxdl2_v2")
    parser.add_argument("--fusion", type=str, default="tan", choices=["tan", "trilinear"])

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

    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--tan-heads", type=int, default=2)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--select-metric", type=str, default="mcc", choices=["mcc", "f1", "bacc"])
    parser.add_argument("--save-att-samples", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir not found: {cache_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"artifacts/protein_plan_test_v2_toxdl2_{args.fusion}")
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("timestamp:", now_timestamp())
    print("device:", device)
    print("fusion:", args.fusion)
    print("cache_dir:", cache_dir)
    print("out_dir:", out_dir)

    train_seqs, train_y = load_split(Path(args.train_csv))
    val_seqs, val_y = load_split(Path(args.val_csv))
    test_seqs, test_y = load_split(Path(args.test_csv))
    indep_seqs, indep_y = load_split(Path(args.independent_csv))

    plm_train, phys_train, motif_train = load_cache(cache_dir, "train")
    plm_val, phys_val, motif_val = load_cache(cache_dir, "val")
    plm_test, phys_test, motif_test = load_cache(cache_dir, "test")
    plm_ind, phys_ind, motif_ind = load_cache(cache_dir, "independent")

    train_ds = FeatureDataset(train_seqs, train_y, plm_train, phys_train, motif_train)
    val_ds = FeatureDataset(val_seqs, val_y, plm_val, phys_val, motif_val)
    test_ds = FeatureDataset(test_seqs, test_y, plm_test, phys_test, motif_test)
    indep_ds = FeatureDataset(indep_seqs, indep_y, plm_ind, phys_ind, motif_ind)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_features)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_features)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_features)
    indep_loader = DataLoader(indep_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_features)

    class_weights = None
    if not args.no_class_weights:
        n0 = int(sum(1 for y in train_y if int(y) == 0))
        n1 = int(sum(1 for y in train_y if int(y) == 1))
        w0 = 1.0
        w1 = float(n0 / max(1, n1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    cfg = FusionV2Config(
        plm_dim=int(plm_train.shape[1]),
        phys_dim=int(phys_train.shape[1]),
        motif_dim=int(motif_train.shape[1]),
        adapter_dim=int(args.adapter_dim),
        fusion=str(args.fusion),
        tan_heads=int(args.tan_heads),
        dropout=float(args.dropout),
        head_hidden=int(args.head_hidden),
        num_classes=2,
    )
    model = EvidentialFusionV2(cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = -1e9
    best_epoch = 0
    best_path = out_dir / "evi_toxdl2_v2.pt"
    patience_left = int(args.patience)
    history: List[Dict[str, object]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        lam = min(1.0, epoch / max(1, int(args.kl_warmup_epochs)))
        running = 0.0
        n = 0

        for batch in train_loader:
            y_one_hot = F.one_hot(batch["labels"].to(device), num_classes=2).float()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(batch["plm"].to(device), batch["phys"].to(device), batch["motif"].to(device))
                loss = dirichlet_evidence_loss(
                    y_one_hot,
                    out["alphas"],
                    lam=float(lam),
                    class_weights=class_weights,
                    focal_gamma=float(args.focal_gamma),
                )
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += float(loss.item()) * batch["labels"].shape[0]
            n += batch["labels"].shape[0]

        train_loss = running / max(1, n)
        val_out = predict_loader(model, val_loader, device)
        val_rank = ranking_metrics(val_out["y_true"], val_out["p_toxic"])
        val_thr = select_threshold(val_out["y_true"], val_out["p_toxic"], metric=args.select_metric)
        score = float(val_thr["best"][args.select_metric])

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "lam": float(lam),
                "val_auroc": float(val_rank["auroc"]),
                "val_auprc": float(val_rank["auprc"]),
                "val_best_thr": float(val_thr["best_thr"]),
                "val_best_metrics": val_thr["best"],
            }
        )
        print(
            f"epoch {epoch:03d} loss={train_loss:.4f} lam={lam:.2f} "
            f"val_auprc={val_rank['auprc']:.4f} val_auroc={val_rank['auroc']:.4f} "
            f"val_{args.select_metric}={score:.4f} thr={val_thr['best_thr']:.4f}"
        )

        if score > best_val:
            best_val = score
            best_epoch = int(epoch)
            patience_left = int(args.patience)
            torch.save({"model_state_dict": model.state_dict(), "cfg": asdict(cfg)}, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model = EvidentialFusionV2(FusionV2Config(**payload["cfg"])).to(device)
    best_model.load_state_dict(payload["model_state_dict"])
    best_model.eval()

    # threshold selection (val, best ckpt)
    val_out = predict_loader(best_model, val_loader, device)
    val_rank = ranking_metrics(val_out["y_true"], val_out["p_toxic"])
    val_thr = select_threshold(val_out["y_true"], val_out["p_toxic"], metric=args.select_metric)
    best_thr = float(val_thr["best_thr"])

    def eval_split(name: str, loader: DataLoader) -> Tuple[Dict[str, object], pd.DataFrame, np.ndarray]:
        out = predict_loader(best_model, loader, device)
        rank = ranking_metrics(out["y_true"], out["p_toxic"])
        m_05 = metrics_at_threshold(out["y_true"], out["p_toxic"], 0.5)
        m_best = metrics_at_threshold(out["y_true"], out["p_toxic"], best_thr)
        df_pred = pd.DataFrame(
            {
                "sequence": out["sequences"],
                "label": out["y_true"],
                "p_toxic": out["p_toxic"],
                "uncertainty": out["uncertainty"],
                "total_evidence": out["total_evidence"],
                "pred_0.5": (out["p_toxic"] >= 0.5).astype(np.int64),
                "pred_best": (out["p_toxic"] >= best_thr).astype(np.int64),
            }
        )
        return (
            {
                "name": name,
                "n": int(len(out["y_true"])),
                "pos": int(out["y_true"].sum()),
                "neg": int(len(out["y_true"]) - int(out["y_true"].sum())),
                "ranking": rank,
                "thr_0.5": m_05,
                "thr_best": m_best,
            },
            df_pred,
            out["fused"],
        )

    val_report, _, _ = eval_split("val", val_loader)
    test_report, test_df, _ = eval_split("test", test_loader)
    indep_report, indep_df, _ = eval_split("independent", indep_loader)

    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    indep_df.to_csv(out_dir / "independent_predictions.csv", index=False)

    # retrieval index (train fused embeddings, cosine)
    train_loader_noshuf = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_features)
    train_out = predict_loader(best_model, train_loader_noshuf, device)
    emb_index = build_embedding_index(train_out["fused"], list(train_out["sequences"].tolist()), list(train_out["y_true"].tolist()))
    save_embedding_index(emb_index, str(out_dir / "retrieval_fused.joblib"))

    # optional: save a small sample of TAN attention maps for analysis/visualization
    att_sample_path = None
    if str(args.fusion) == "tan" and int(args.save_att_samples) > 0:
        n_s = int(min(int(args.save_att_samples), len(indep_ds)))
        idx = np.random.choice(len(indep_ds), size=n_s, replace=False)
        plm_s = torch.tensor(plm_ind[idx], dtype=torch.float32, device=device)
        phys_s = torch.tensor(phys_ind[idx], dtype=torch.float32, device=device)
        motif_s = torch.tensor(motif_ind[idx], dtype=torch.float32, device=device)
        out = best_model(plm_s, phys_s, motif_s, return_att_maps=True)
        att = out.get("att_maps", torch.empty(0)).detach().cpu().to(dtype=torch.float16).numpy()
        fused = out["fused"].detach().cpu().numpy().astype(np.float32, copy=False)
        sample = {
            "indices_in_independent": idx.tolist(),
            "sequences": [indep_seqs[int(i)] for i in idx.tolist()],
            "att_maps": att,
            "fused": fused,
        }
        att_sample_path = str(out_dir / "tan_att_samples.npz")
        np.savez_compressed(att_sample_path, **sample)

    summary = {
        "timestamp": now_timestamp(),
        "seed": int(args.seed),
        "device": str(device),
        "fusion": str(args.fusion),
        "cache_dir": str(cache_dir),
        "data": {
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
            "independent_csv": args.independent_csv,
            "train_counts": {"n": int(len(train_y)), "neg": int(sum(1 for y in train_y if int(y) == 0)), "pos": int(sum(1 for y in train_y if int(y) == 1))},
            "val_counts": {"n": int(len(val_y)), "neg": int(sum(1 for y in val_y if int(y) == 0)), "pos": int(sum(1 for y in val_y if int(y) == 1))},
            "test_counts": {"n": int(len(test_y)), "neg": int(sum(1 for y in test_y if int(y) == 0)), "pos": int(sum(1 for y in test_y if int(y) == 1))},
            "independent_counts": {"n": int(len(indep_y)), "neg": int(sum(1 for y in indep_y if int(y) == 0)), "pos": int(sum(1 for y in indep_y if int(y) == 1))},
        },
        "model": {"arch": "EvidentialFusionV2(PLM+phys+motif)+evidential", "cfg": asdict(cfg)},
        "train": {
            "epochs_ran": int(len(history)),
            "best_epoch": int(best_epoch),
            "best_val_metric": args.select_metric,
            "best_val_score": float(best_val),
            "best_ckpt": str(best_path),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "dropout": float(args.dropout),
            "kl_warmup_epochs": int(args.kl_warmup_epochs),
            "focal_gamma": float(args.focal_gamma),
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
            "history": history,
        },
        "threshold_selection": {"metric": args.select_metric, "best_thr": best_thr, "val_ranking": val_rank, "val_best": val_thr["best"]},
        "eval": {"val": val_report, "test": test_report, "independent": indep_report},
        "artifacts": {
            "test_predictions_csv": str(out_dir / "test_predictions.csv"),
            "independent_predictions_csv": str(out_dir / "independent_predictions.csv"),
            "retrieval_fused": str(out_dir / "retrieval_fused.joblib"),
            "tan_att_samples": att_sample_path,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("saved:", out_dir / "summary.json")
    print("best_thr:", best_thr)
    print("test auprc:", test_report["ranking"]["auprc"], "bacc(best):", test_report["thr_best"]["bacc"])
    print("independent auprc:", indep_report["ranking"]["auprc"], "bacc(best):", indep_report["thr_best"]["bacc"])


if __name__ == "__main__":
    main()

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.data import SequenceDataset, collate_batch, load_sequences_labels  # noqa: E402
from toxapp.evidential import alpha_to_probs, alpha_total_evidence, alpha_uncertainty, dirichlet_evidence_loss  # noqa: E402
from toxapp.inference import save_checkpoint  # noqa: E402
from toxapp.model import EvidentialToxModel, ModelConfig  # noqa: E402
from toxapp.retrieval import build_retrieval_index, save_retrieval_index  # noqa: E402


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_loader(model: EvidentialToxModel, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    p_toxic: List[float] = []
    uncertainty: List[float] = []
    total_evidence: List[float] = []
    sequences: List[str] = []

    for batch in loader:
        out = model(batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device))
        alpha = out["alphas"].detach().cpu()
        probs = alpha_to_probs(alpha).numpy()
        p = probs[:, 1]
        y = batch.labels.detach().cpu().numpy()

        y_true.extend(y.tolist())
        p_toxic.extend(p.tolist())
        uncertainty.extend(alpha_uncertainty(alpha).numpy().tolist())
        total_evidence.extend(alpha_total_evidence(alpha).numpy().tolist())
        sequences.extend(batch.sequences)

    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "p_toxic": np.asarray(p_toxic, dtype=np.float64),
        "uncertainty": np.asarray(uncertainty, dtype=np.float64),
        "total_evidence": np.asarray(total_evidence, dtype=np.float64),
        "sequences": np.asarray(sequences, dtype=object),
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
    return {
        "auroc": float(roc_auc_score(y, p_pos)),
        "auprc": float(average_precision_score(y, p_pos)),
    }


def select_threshold(y_true: np.ndarray, p_pos: np.ndarray, metric: str = "mcc") -> Dict[str, float]:
    metric = metric.lower()
    if metric not in {"mcc", "f1", "bacc"}:
        raise ValueError("select metric must be one of: mcc, f1, bacc")

    # Candidate thresholds from predicted probabilities (+ a few anchors).
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


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--out-dir", type=str, default="artifacts/protein_plan_test_v1_toxdl2")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--select-metric", type=str, default="mcc", choices=["mcc", "f1", "bacc"])
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("timestamp:", now_timestamp())
    print("device:", device)

    train_seqs, train_y = load_sequences_labels(args.train_csv)
    val_seqs, val_y = load_sequences_labels(args.val_csv)
    test_seqs, test_y = load_sequences_labels(args.test_csv)
    indep_seqs, indep_y = load_sequences_labels(args.independent_csv)

    train_ds = SequenceDataset(train_seqs, train_y)
    val_ds = SequenceDataset(val_seqs, val_y)
    test_ds = SequenceDataset(test_seqs, test_y)
    indep_ds = SequenceDataset(indep_seqs, indep_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    indep_loader = DataLoader(indep_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    cfg = ModelConfig(dropout=args.dropout)
    model = EvidentialToxModel(cfg).to(device)

    # class_weights: [w0, w1] where label=1 is toxic (minority in time splits)
    class_weights = None
    if not args.no_class_weights:
        n0 = int(sum(1 for y in train_y if int(y) == 0))
        n1 = int(sum(1 for y in train_y if int(y) == 1))
        w0 = 1.0
        w1 = float(n0 / max(1, n1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = -1e9
    best_epoch = 0
    best_path = out_dir / "evi_toxdl2.pt"
    patience_left = int(args.patience)

    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        lam = min(1.0, epoch / max(1, args.kl_warmup_epochs))
        running = 0.0
        n = 0

        for batch in train_loader:
            y_one_hot = torch.nn.functional.one_hot(batch.labels.to(device), num_classes=cfg.num_classes).float()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device))
                loss = dirichlet_evidence_loss(
                    y_one_hot,
                    out["alphas"],
                    lam=lam,
                    class_weights=class_weights,
                    focal_gamma=float(args.focal_gamma),
                )
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += float(loss.item()) * batch.labels.shape[0]
            n += batch.labels.shape[0]

        train_loss = running / max(1, n)

        val_out = predict_loader(model, val_loader, device)
        val_rank = ranking_metrics(val_out["y_true"], val_out["p_toxic"])
        val_thr = select_threshold(val_out["y_true"], val_out["p_toxic"], metric=args.select_metric)
        val_thr_metrics = val_thr["best"]

        score = float(val_thr_metrics[args.select_metric])
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "lam": float(lam),
                "val_auroc": float(val_rank["auroc"]),
                "val_auprc": float(val_rank["auprc"]),
                "val_best_thr": float(val_thr["best_thr"]),
                "val_best_metrics": val_thr_metrics,
            }
        )
        print(
            f"epoch {epoch:03d} loss={train_loss:.4f} lam={lam:.2f} "
            f"val_auprc={val_rank['auprc']:.4f} val_auroc={val_rank['auroc']:.4f} "
            f"val_{args.select_metric}={score:.4f} thr={val_thr['best_thr']:.4f}"
        )

        if score > best_val:
            best_val = score
            best_epoch = epoch
            patience_left = int(args.patience)
            save_checkpoint(best_path, model)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    # Load best
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model = EvidentialToxModel(ModelConfig(**ckpt["config"])).to(device)
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_model.eval()

    # Select threshold on val from best checkpoint
    val_out = predict_loader(best_model, val_loader, device)
    val_rank = ranking_metrics(val_out["y_true"], val_out["p_toxic"])
    val_thr = select_threshold(val_out["y_true"], val_out["p_toxic"], metric=args.select_metric)
    best_thr = float(val_thr["best_thr"])

    def eval_split(name: str, loader: DataLoader) -> Tuple[Dict[str, object], pd.DataFrame]:
        out = predict_loader(best_model, loader, device)
        y_true = out["y_true"]
        p = out["p_toxic"]
        rank = ranking_metrics(y_true, p)
        m_05 = metrics_at_threshold(y_true, p, 0.5)
        m_best = metrics_at_threshold(y_true, p, best_thr)

        df_pred = pd.DataFrame(
            {
                "sequence": out["sequences"],
                "label": y_true,
                "p_toxic": p,
                "uncertainty": out["uncertainty"],
                "total_evidence": out["total_evidence"],
                "pred_0.5": (p >= 0.5).astype(np.int64),
                "pred_best": (p >= best_thr).astype(np.int64),
            }
        )
        return (
            {
                "name": name,
                "n": int(len(y_true)),
                "pos": int(y_true.sum()),
                "neg": int(len(y_true) - int(y_true.sum())),
                "ranking": rank,
                "thr_0.5": m_05,
                "thr_best": m_best,
            },
            df_pred,
        )

    val_report, _ = eval_split("val", val_loader)
    test_report, test_df = eval_split("test", test_loader)
    indep_report, indep_df = eval_split("independent", indep_loader)

    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    indep_df.to_csv(out_dir / "independent_predictions.csv", index=False)

    retrieval = build_retrieval_index(train_seqs, train_y)
    save_retrieval_index(retrieval, str(out_dir / "retrieval.joblib"))

    summary = {
        "timestamp": now_timestamp(),
        "seed": int(args.seed),
        "device": str(device),
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
        "model": {
            "arch": "EvidentialToxModel",
            "config": asdict(cfg),
        },
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
            "retrieval_index": str(out_dir / "retrieval.joblib"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("saved:", out_dir / "summary.json")
    print("best_thr:", best_thr)
    print("test auprc:", test_report["ranking"]["auprc"], "bacc(best):", test_report["thr_best"]["bacc"])
    print("independent auprc:", indep_report["ranking"]["auprc"], "bacc(best):", indep_report["thr_best"]["bacc"])


if __name__ == "__main__":
    main()

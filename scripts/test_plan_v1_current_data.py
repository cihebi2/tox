from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
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
    roc_auc_score,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.data import SequenceDataset, collate_batch, load_sequences_labels  # noqa: E402
from toxapp.evidential import (  # noqa: E402
    alpha_to_probs,
    alpha_total_evidence,
    alpha_uncertainty,
    dirichlet_evidence_loss,
)
from toxapp.inference import predict_sequences, save_checkpoint  # noqa: E402
from toxapp.model import EvidentialToxModel, ModelConfig  # noqa: E402
from toxapp.retrieval import build_retrieval_index, query, save_retrieval_index  # noqa: E402
from toxapp.suggest import generate_all_single_mutations  # noqa: E402


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class EvalOutputs:
    y_true: np.ndarray
    p_toxic: np.ndarray
    pred: np.ndarray
    uncertainty: np.ndarray
    total_evidence: np.ndarray


@torch.no_grad()
def predict_loader(model: EvidentialToxModel, loader: DataLoader, device: torch.device) -> EvalOutputs:
    model.eval()
    y_true: List[int] = []
    p_toxic: List[float] = []
    pred: List[int] = []
    uncertainty: List[float] = []
    total_evidence: List[float] = []

    for batch in loader:
        out = model(batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device))
        alpha = out["alphas"].detach().cpu()
        probs = alpha_to_probs(alpha).numpy()
        p = probs[:, 1]
        y = batch.labels.detach().cpu().numpy()

        y_true.extend(y.tolist())
        p_toxic.extend(p.tolist())
        pred.extend((p >= 0.5).astype(np.int64).tolist())
        uncertainty.extend(alpha_uncertainty(alpha).numpy().tolist())
        total_evidence.extend(alpha_total_evidence(alpha).numpy().tolist())

    return EvalOutputs(
        y_true=np.asarray(y_true, dtype=np.int64),
        p_toxic=np.asarray(p_toxic, dtype=np.float64),
        pred=np.asarray(pred, dtype=np.int64),
        uncertainty=np.asarray(uncertainty, dtype=np.float64),
        total_evidence=np.asarray(total_evidence, dtype=np.float64),
    )


def compute_ece(y_true: np.ndarray, p_pos: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """
    ECE on confidence = max(p, 1-p).
    """
    if y_true.shape != p_pos.shape:
        raise ValueError("y_true and p_pos shape mismatch.")

    conf = np.maximum(p_pos, 1.0 - p_pos)
    pred = (p_pos >= 0.5).astype(np.int64)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(acc_bin - conf_bin)

    return {"ece": float(ece), "n_bins": int(n_bins)}


def compute_selective_accuracy(outputs: EvalOutputs, coverages: List[float]) -> List[Dict[str, float]]:
    order = np.argsort(outputs.uncertainty)  # low uncertainty first
    y = outputs.y_true[order]
    pred = outputs.pred[order]
    out: List[Dict[str, float]] = []
    n = len(y)
    for cov in coverages:
        k = max(1, int(round(n * cov)))
        acc = float((pred[:k] == y[:k]).mean())
        out.append({"coverage": float(cov), "n": int(k), "acc": acc})
    return out


def basic_metrics(outputs: EvalOutputs) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(outputs.y_true, outputs.pred)),
        "f1": float(f1_score(outputs.y_true, outputs.pred)),
        "mcc": float(matthews_corrcoef(outputs.y_true, outputs.pred)),
        "auroc": float(roc_auc_score(outputs.y_true, outputs.p_toxic)),
        "auprc": float(average_precision_score(outputs.y_true, outputs.p_toxic)),
    }


def evaluate_retrieval(
    index, test_sequences: List[str], test_labels: List[int], top_k: int = 5
) -> Dict[str, float]:
    top1_correct = 0
    topk_majority_correct = 0
    sims_top1: List[float] = []

    for seq, y in zip(test_sequences, test_labels):
        hits = query(index, seq, top_k=top_k)
        if not hits:
            continue
        top1 = hits[0]
        sims_top1.append(float(top1["similarity"]))
        top1_correct += int(int(top1["label"]) == int(y))

        labels = [int(h["label"]) for h in hits]
        maj = int(sum(labels) >= (len(labels) / 2))
        topk_majority_correct += int(maj == int(y))

    n = len(test_sequences)
    return {
        "top_k": int(top_k),
        "top1_acc": float(top1_correct / n),
        "topk_majority_acc": float(topk_majority_correct / n),
        "top1_similarity_mean": float(np.mean(sims_top1)) if sims_top1 else 0.0,
    }


def pick_design_cases(sequences: List[str], preds: List[dict], k: int = 5) -> List[str]:
    # Pick k most toxic, reasonably confident cases for demonstration.
    scored = []
    for s, p in zip(sequences, preds):
        scored.append((p["p_toxic"], -p["uncertainty"], s))
    scored.sort(reverse=True)
    return [s for _, _, s in scored[:k]]


def propose_single_mutations(
    model: EvidentialToxModel,
    sequences: List[str],
    device: torch.device,
    out_dir: Path,
    batch_size: int,
    top_n: int = 5,
) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    base_preds = predict_sequences(model, sequences, device=device, batch_size=batch_size)
    chosen = pick_design_cases(sequences, base_preds, k=min(5, len(sequences)))

    for seq in chosen:
        muts = generate_all_single_mutations(seq)
        mut_seqs = [m.mutant for m in muts]
        mut_preds = predict_sequences(model, mut_seqs, device=device, batch_size=batch_size)

        base = next(p for p in base_preds if p["sequence"] == seq)
        # sort by lower p_toxic, then lower uncertainty
        ranked = sorted(
            zip(muts, mut_preds),
            key=lambda t: (t[1]["p_toxic"], t[1]["uncertainty"]),
        )
        top = []
        for m, p in ranked[:top_n]:
            top.append(
                {
                    "mutation": f"{m.from_aa}{m.position_1based}{m.to_aa}",
                    "mutant": m.mutant,
                    "p_toxic": float(p["p_toxic"]),
                    "uncertainty": float(p["uncertainty"]),
                    "total_evidence": float(p["total_evidence"]),
                }
            )

        cases.append(
            {
                "sequence": seq,
                "base": {
                    "p_toxic": float(base["p_toxic"]),
                    "uncertainty": float(base["uncertainty"]),
                    "total_evidence": float(base["total_evidence"]),
                },
                "top_mutations": top,
            }
        )

    (out_dir / "design_suggestions_sample.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/toxicity_data_v1/splits/peptide_id90_seed69_80_10_10/peptide_id90_train.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/toxicity_data_v1/splits/peptide_id90_seed69_80_10_10/peptide_id90_val.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/toxicity_data_v1/splits/peptide_id90_seed69_80_10_10/peptide_id90_test.csv",
    )
    parser.add_argument("--out-dir", type=str, default="artifacts/plan_test_v1_peptide_id90_seed69")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--retrieval-topk", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seqs, train_y = load_sequences_labels(str((repo_root / args.train_csv).resolve()))
    val_seqs, val_y = load_sequences_labels(str((repo_root / args.val_csv).resolve()))
    test_seqs, test_y = load_sequences_labels(str((repo_root / args.test_csv).resolve()))

    train_ds = SequenceDataset(train_seqs, train_y)
    val_ds = SequenceDataset(val_seqs, val_y)
    test_ds = SequenceDataset(test_seqs, test_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    cfg = ModelConfig(dropout=args.dropout)
    model = EvidentialToxModel(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auroc = -1.0
    best_epoch = 0
    best_path = out_dir / "evi_tox.pt"

    history: List[Dict[str, object]] = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        lam = min(1.0, epoch / max(1, args.kl_warmup_epochs))
        running = 0.0
        n = 0
        for batch in train_loader:
            out = model(batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device))
            y = torch.nn.functional.one_hot(batch.labels.to(device), num_classes=cfg.num_classes).float()
            loss = dirichlet_evidence_loss(y, out["alphas"], lam=lam)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item()) * batch.labels.shape[0]
            n += batch.labels.shape[0]

        train_loss = running / max(1, n)
        val_out = predict_loader(model, val_loader, device)
        val_metrics = basic_metrics(val_out)

        row = {"epoch": epoch, "train_loss": float(train_loss), "lam": float(lam), "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            save_checkpoint(best_path, model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    # Load best
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    best_cfg = ModelConfig(**ckpt["config"])
    best_model = EvidentialToxModel(best_cfg).to(device)
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_model.eval()

    val_best = predict_loader(best_model, val_loader, device)
    test_out = predict_loader(best_model, test_loader, device)

    val_metrics = basic_metrics(val_best)
    test_metrics = basic_metrics(test_out)
    val_cal = compute_ece(val_best.y_true, val_best.p_toxic)
    test_cal = compute_ece(test_out.y_true, test_out.p_toxic)

    coverages = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
    val_sel = compute_selective_accuracy(val_best, coverages)
    test_sel = compute_selective_accuracy(test_out, coverages)

    # Retrieval index built on training set as "evidence"
    retrieval = build_retrieval_index(train_seqs, train_y)
    save_retrieval_index(retrieval, str(out_dir / "retrieval.joblib"))
    retrieval_eval = evaluate_retrieval(retrieval, test_seqs, test_y, top_k=args.retrieval_topk)

    # Save test predictions
    test_pred_df = pd.DataFrame(
        {
            "sequence": test_seqs,
            "label": test_y,
            "p_toxic": test_out.p_toxic,
            "pred": test_out.pred,
            "uncertainty": test_out.uncertainty,
            "total_evidence": test_out.total_evidence,
        }
    )
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # Design suggestion sanity check (sample)
    propose_single_mutations(best_model, test_seqs, device, out_dir, batch_size=args.batch_size, top_n=5)

    summary = {
        "timestamp": now_timestamp(),
        "device": str(device),
        "data": {
            "train_csv": str((repo_root / args.train_csv).resolve()),
            "val_csv": str((repo_root / args.val_csv).resolve()),
            "test_csv": str((repo_root / args.test_csv).resolve()),
            "train_n": int(len(train_seqs)),
            "val_n": int(len(val_seqs)),
            "test_n": int(len(test_seqs)),
        },
        "model_config": asdict(best_cfg),
        "train_config": {
            "seed": int(args.seed),
            "epochs_ran": int(len(history)),
            "epochs_max": int(args.epochs),
            "patience": int(args.patience),
            "best_epoch": int(best_epoch),
            "best_val_auroc": float(best_val_auroc),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "kl_warmup_epochs": int(args.kl_warmup_epochs),
        },
        "val": {
            "metrics": val_metrics,
            "ece": val_cal,
            "selective_acc": val_sel,
            "uncertainty_mean": float(val_best.uncertainty.mean()),
            "uncertainty_mean_correct": float(val_best.uncertainty[val_best.pred == val_best.y_true].mean()),
            "uncertainty_mean_incorrect": float(val_best.uncertainty[val_best.pred != val_best.y_true].mean())
            if (val_best.pred != val_best.y_true).any()
            else 0.0,
        },
        "test": {
            "metrics": test_metrics,
            "ece": test_cal,
            "selective_acc": test_sel,
            "retrieval": retrieval_eval,
            "uncertainty_mean": float(test_out.uncertainty.mean()),
            "uncertainty_mean_correct": float(test_out.uncertainty[test_out.pred == test_out.y_true].mean()),
            "uncertainty_mean_incorrect": float(test_out.uncertainty[test_out.pred != test_out.y_true].mean())
            if (test_out.pred != test_out.y_true).any()
            else 0.0,
        },
        "artifacts": {
            "checkpoint": str(best_path),
            "retrieval_index": str(out_dir / "retrieval.joblib"),
            "test_predictions_csv": str(out_dir / "test_predictions.csv"),
            "design_suggestions_sample": str(out_dir / "design_suggestions_sample.json"),
        },
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[DONE] summary:", out_dir / "summary.json")
    print("[DONE] test metrics:", test_metrics)


if __name__ == "__main__":
    main()

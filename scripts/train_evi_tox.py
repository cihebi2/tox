from __future__ import annotations

import sys

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.data import SequenceDataset, collate_batch, load_sequences_labels  # noqa: E402
from toxapp.evidential import alpha_to_probs, dirichlet_evidence_loss  # noqa: E402
from toxapp.inference import save_checkpoint  # noqa: E402
from toxapp.model import EvidentialToxModel, ModelConfig  # noqa: E402
from toxapp.retrieval import build_retrieval_index, save_retrieval_index  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: EvidentialToxModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    probs_list = []
    labels_list = []
    for batch in loader:
        out = model(batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device))
        probs = alpha_to_probs(out["alphas"]).detach().cpu().numpy()
        probs_list.append(probs)
        labels_list.append(batch.labels.detach().cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    p_tox = probs[:, 1]
    pred = (p_tox >= 0.5).astype(np.int64)
    return {
        "acc": float(accuracy_score(labels, pred)),
        "auroc": float(roc_auc_score(labels, p_tox)),
        "auprc": float(average_precision_score(labels, p_tox)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default="/root/private_data/dd_model/ToxGIN/train_sequence.csv",
        help="ToxGIN train_sequence.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="/root/private_data/dd_model/ToxGIN/test_sequence.csv",
        help="ToxGIN test_sequence.csv",
    )
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_sequences, train_labels = load_sequences_labels(args.train_csv)
    test_sequences, test_labels = load_sequences_labels(args.test_csv)

    tr_seqs, va_seqs, tr_y, va_y = train_test_split(
        train_sequences,
        train_labels,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_labels,
    )

    train_ds = SequenceDataset(tr_seqs, tr_y)
    val_ds = SequenceDataset(va_seqs, va_y)
    test_ds = SequenceDataset(test_sequences, test_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    cfg = ModelConfig(dropout=args.dropout)
    model = EvidentialToxModel(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_path = out_dir / "evi_tox.pt"

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

        val_metrics = evaluate(model, val_loader, device)
        train_loss = running / max(1, n)
        print(f"epoch {epoch:03d} loss={train_loss:.4f} lam={lam:.2f} val={val_metrics}")

        if val_metrics["auroc"] > best_val:
            best_val = val_metrics["auroc"]
            save_checkpoint(best_path, model)

    print("best checkpoint:", best_path)
    best_model = EvidentialToxModel(cfg).to(device)
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_model.eval()

    test_metrics = evaluate(best_model, test_loader, device)
    print("test:", test_metrics)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"val_best_auroc": best_val, "test": test_metrics}, ensure_ascii=False, indent=2))

    retrieval = build_retrieval_index(train_sequences, train_labels)
    save_retrieval_index(retrieval, str(out_dir / "retrieval.joblib"))
    print("saved retrieval index:", out_dir / "retrieval.joblib")


if __name__ == "__main__":
    main()

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
import torch.nn as nn
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toxapp.data import RawBatch, SequenceDataset, collate_raw_batch, load_sequences_labels  # noqa: E402
from toxapp.evidential import alpha_to_probs, alpha_total_evidence, alpha_uncertainty, dirichlet_evidence_loss  # noqa: E402
from toxapp.graph import GraphEncoderConfig, MeanGraphEncoder  # noqa: E402
from toxapp.plm_contact import EsmPseudoContactConfig, EsmPseudoContactExtractor  # noqa: E402
from toxapp.retrieval import build_retrieval_index, save_retrieval_index  # noqa: E402
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


class EvidentialMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(feats)
        alphas = F.softplus(logits) + 1.0
        return {"logits": logits, "alphas": alphas}


@dataclass(frozen=True)
class PredOutputs:
    y_true: np.ndarray
    p_toxic: np.ndarray
    uncertainty: np.ndarray
    total_evidence: np.ndarray
    sequences: np.ndarray
    fused_embeddings: np.ndarray  # [N, D]


def fused_encode_batch(
    extractor: EsmPseudoContactExtractor,
    graph_encoder: MeanGraphEncoder,
    sequences: list[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    chunks_per_seq = extractor.extract(sequences)
    fused: list[torch.Tensor] = []
    for chunks in chunks_per_seq:
        if not chunks:
            # fallback: zeros if extraction fails (should be rare)
            d_seq = int(extractor.hidden_size)
            d_graph = int(graph_encoder.cfg.hidden_dim)
            fused.append(torch.zeros((d_seq + d_graph,), dtype=torch.float32, device=device))
            continue

        num = torch.zeros((int(extractor.hidden_size) + int(graph_encoder.cfg.hidden_dim),), dtype=torch.float32, device=device)
        den = 0.0
        for ch in chunks:
            seq_emb = ch.residue_embeddings.mean(dim=0)  # [480]
            graph_emb = graph_encoder(ch.residue_embeddings, ch.edge_index)  # [256]
            vec = torch.cat([seq_emb, graph_emb], dim=-1).to(dtype=torch.float32)
            w = float(ch.weight)
            num = num + vec * w
            den += w
        fused.append(num / max(1e-6, den))
    return torch.stack(fused, dim=0)  # [B, D]


@torch.no_grad()
def predict_loader(
    extractor: EsmPseudoContactExtractor,
    graph_encoder: MeanGraphEncoder,
    head: EvidentialMLPHead,
    loader: DataLoader,
    device: torch.device,
) -> PredOutputs:
    head.eval()
    y_true: List[int] = []
    p_toxic: List[float] = []
    uncertainty: List[float] = []
    total_evidence: List[float] = []
    sequences: List[str] = []
    fused_embeddings: List[np.ndarray] = []

    for batch in loader:
        assert isinstance(batch, RawBatch)
        feats = fused_encode_batch(extractor, graph_encoder, batch.sequences, device=device)  # [B, D]
        out = head(feats)
        alpha = out["alphas"].detach().cpu()
        probs = alpha_to_probs(alpha).numpy()
        p = probs[:, 1]
        y = batch.labels.detach().cpu().numpy()

        y_true.extend(y.tolist())
        p_toxic.extend(p.tolist())
        uncertainty.extend(alpha_uncertainty(alpha).numpy().tolist())
        total_evidence.extend(alpha_total_evidence(alpha).numpy().tolist())
        sequences.extend(batch.sequences)
        fused_embeddings.extend(feats.detach().cpu().numpy())

    return PredOutputs(
        y_true=np.asarray(y_true, dtype=np.int64),
        p_toxic=np.asarray(p_toxic, dtype=np.float64),
        uncertainty=np.asarray(uncertainty, dtype=np.float64),
        total_evidence=np.asarray(total_evidence, dtype=np.float64),
        sequences=np.asarray(sequences, dtype=object),
        fused_embeddings=np.asarray(fused_embeddings, dtype=np.float32),
    )


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
    parser.add_argument("--plm-path", type=str, default="/root/group_data/qiuleyu/esm2_t12_35M_UR50D")
    parser.add_argument("--max-residues", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--chunk-batch-size", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--no-apc", action="store_true")

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

    parser.add_argument("--out-dir", type=str, default="artifacts/protein_plan_test_v1_toxdl2_plm_contact")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--graph-hidden-dim", type=int, default=256)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-dropout", type=float, default=0.1)
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

    extractor = EsmPseudoContactExtractor(
        EsmPseudoContactConfig(
            model_path=args.plm_path,
            max_residues=int(args.max_residues),
            stride=int(args.stride),
            chunk_batch_size=int(args.chunk_batch_size),
            use_amp=(not bool(args.no_amp)),
            device=str(device),
            top_k=int(args.top_k),
            use_apc=(not bool(args.no_apc)),
            attn_implementation="eager",
            freeze_plm=True,
        )
    )

    graph_encoder = MeanGraphEncoder(
        GraphEncoderConfig(
            in_dim=int(extractor.hidden_size),
            hidden_dim=int(args.graph_hidden_dim),
            num_layers=int(args.graph_layers),
            dropout=float(args.graph_dropout),
        )
    ).to(device)

    in_dim = int(extractor.hidden_size) + int(args.graph_hidden_dim)
    head = EvidentialMLPHead(in_dim=in_dim, hidden_dim=int(args.adapter_dim), dropout=float(args.dropout), num_classes=2).to(device)

    train_seqs, train_y = load_sequences_labels(args.train_csv)
    val_seqs, val_y = load_sequences_labels(args.val_csv)
    test_seqs, test_y = load_sequences_labels(args.test_csv)
    indep_seqs, indep_y = load_sequences_labels(args.independent_csv)

    train_ds = SequenceDataset(train_seqs, train_y)
    val_ds = SequenceDataset(val_seqs, val_y)
    test_ds = SequenceDataset(test_seqs, test_y)
    indep_ds = SequenceDataset(indep_seqs, indep_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_raw_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_raw_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_raw_batch)
    indep_loader = DataLoader(indep_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_raw_batch)

    class_weights = None
    if not args.no_class_weights:
        n0 = int(sum(1 for y in train_y if int(y) == 0))
        n1 = int(sum(1 for y in train_y if int(y) == 1))
        w0 = 1.0
        w1 = float(n0 / max(1, n1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    optim = torch.optim.AdamW(list(graph_encoder.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1e9
    best_epoch = 0
    patience_left = int(args.patience)
    best_path = out_dir / "evi_toxdl2_plm_contact.pt"
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        graph_encoder.train()
        head.train()
        lam = min(1.0, epoch / max(1, args.kl_warmup_epochs))
        running = 0.0
        n = 0

        for batch in train_loader:
            assert isinstance(batch, RawBatch)
            feats = fused_encode_batch(extractor, graph_encoder, batch.sequences, device=device)
            out = head(feats)
            y_one_hot = F.one_hot(batch.labels.to(device), num_classes=2).float()
            loss = dirichlet_evidence_loss(
                y_one_hot,
                out["alphas"],
                lam=lam,
                class_weights=class_weights,
                focal_gamma=float(args.focal_gamma),
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += float(loss.item()) * batch.labels.shape[0]
            n += batch.labels.shape[0]

        train_loss = running / max(1, n)
        val_out = predict_loader(extractor, graph_encoder, head, val_loader, device)
        val_rank = ranking_metrics(val_out.y_true, val_out.p_toxic)
        val_thr = select_threshold(val_out.y_true, val_out.p_toxic, metric=args.select_metric)
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
            best_epoch = epoch
            patience_left = int(args.patience)
            torch.save(
                {
                    "graph_state_dict": graph_encoder.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "graph_cfg": asdict(graph_encoder.cfg),
                    "head_cfg": {"in_dim": in_dim, "adapter_dim": int(args.adapter_dim), "dropout": float(args.dropout)},
                },
                best_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    best_graph = MeanGraphEncoder(GraphEncoderConfig(**payload["graph_cfg"])).to(device)
    best_graph.load_state_dict(payload["graph_state_dict"])
    best_graph.eval()
    best_head = EvidentialMLPHead(
        in_dim=int(payload["head_cfg"]["in_dim"]),
        hidden_dim=int(payload["head_cfg"]["adapter_dim"]),
        dropout=float(payload["head_cfg"]["dropout"]),
        num_classes=2,
    ).to(device)
    best_head.load_state_dict(payload["head_state_dict"])
    best_head.eval()

    val_out = predict_loader(extractor, best_graph, best_head, val_loader, device)
    val_rank = ranking_metrics(val_out.y_true, val_out.p_toxic)
    val_thr = select_threshold(val_out.y_true, val_out.p_toxic, metric=args.select_metric)
    best_thr = float(val_thr["best_thr"])

    def eval_split(name: str, loader: DataLoader) -> Tuple[Dict[str, object], pd.DataFrame, np.ndarray]:
        out = predict_loader(extractor, best_graph, best_head, loader, device)
        rank = ranking_metrics(out.y_true, out.p_toxic)
        m_05 = metrics_at_threshold(out.y_true, out.p_toxic, 0.5)
        m_best = metrics_at_threshold(out.y_true, out.p_toxic, best_thr)
        df_pred = pd.DataFrame(
            {
                "sequence": out.sequences,
                "label": out.y_true,
                "p_toxic": out.p_toxic,
                "uncertainty": out.uncertainty,
                "total_evidence": out.total_evidence,
                "pred_0.5": (out.p_toxic >= 0.5).astype(np.int64),
                "pred_best": (out.p_toxic >= best_thr).astype(np.int64),
            }
        )
        return (
            {
                "name": name,
                "n": int(len(out.y_true)),
                "pos": int(out.y_true.sum()),
                "neg": int(len(out.y_true) - int(out.y_true.sum())),
                "ranking": rank,
                "thr_0.5": m_05,
                "thr_best": m_best,
            },
            df_pred,
            out.fused_embeddings,
        )

    val_report, _, _ = eval_split("val", val_loader)
    test_report, test_df, _ = eval_split("test", test_loader)
    indep_report, indep_df, _ = eval_split("independent", indep_loader)

    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    indep_df.to_csv(out_dir / "independent_predictions.csv", index=False)

    retrieval_tfidf = build_retrieval_index(train_seqs, train_y)
    save_retrieval_index(retrieval_tfidf, str(out_dir / "retrieval_tfidf.joblib"))

    train_loader_noshuf = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_raw_batch)
    train_out = predict_loader(extractor, best_graph, best_head, train_loader_noshuf, device)
    emb_index = build_embedding_index(train_out.fused_embeddings, list(train_out.sequences.tolist()), list(train_out.y_true.tolist()))
    save_embedding_index(emb_index, str(out_dir / "retrieval_plm_contact.joblib"))

    summary = {
        "timestamp": now_timestamp(),
        "seed": int(args.seed),
        "device": str(device),
        "plm_contact": asdict(
            EsmPseudoContactConfig(
                model_path=args.plm_path,
                max_residues=int(args.max_residues),
                stride=int(args.stride),
                chunk_batch_size=int(args.chunk_batch_size),
                use_amp=(not bool(args.no_amp)),
                device=str(device),
                top_k=int(args.top_k),
                use_apc=(not bool(args.no_apc)),
                attn_implementation="eager",
                freeze_plm=True,
            )
        ),
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
            "arch": "ESM2-35M(last-layer-attn contact)+GNN+evidential",
            "hidden_size": int(extractor.hidden_size),
            "graph_cfg": asdict(graph_encoder.cfg),
            "head_in_dim": int(in_dim),
            "adapter_dim": int(args.adapter_dim),
            "dropout": float(args.dropout),
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
            "retrieval_tfidf": str(out_dir / "retrieval_tfidf.joblib"),
            "retrieval_plm_contact": str(out_dir / "retrieval_plm_contact.joblib"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("saved:", out_dir / "summary.json")
    print("best_thr:", best_thr)
    print("test auprc:", test_report["ranking"]["auprc"], "bacc(best):", test_report["thr_best"]["bacc"])
    print("independent auprc:", indep_report["ranking"]["auprc"], "bacc(best):", indep_report["thr_best"]["bacc"])

    extractor.close()


if __name__ == "__main__":
    main()

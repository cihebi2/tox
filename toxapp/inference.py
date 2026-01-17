from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .data import collate_batch, sanitize_sequence
from .evidential import alpha_to_probs, alpha_total_evidence, alpha_uncertainty
from .model import EvidentialToxModel, ModelConfig


def save_checkpoint(path: str | Path, model: EvidentialToxModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model.cfg),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> EvidentialToxModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = EvidentialToxModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_sequences(
    model: EvidentialToxModel,
    sequences: list[str],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> list[dict]:
    device = torch.device(device)
    cleaned = [sanitize_sequence(s) for s in sequences]
    outputs: list[dict] = []

    for start in range(0, len(cleaned), batch_size):
        batch_seqs = cleaned[start : start + batch_size]
        batch = collate_batch([(s, 0) for s in batch_seqs])
        out = model(batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device))
        alpha = out["alphas"].detach().cpu()
        probs = alpha_to_probs(alpha)
        unc = alpha_uncertainty(alpha)
        evidence = alpha_total_evidence(alpha)

        for i, seq in enumerate(batch_seqs):
            p_tox = float(probs[i, 1])
            outputs.append(
                {
                    "sequence": seq,
                    "p_toxic": p_tox,
                    "pred_label": int(p_tox >= 0.5),
                    "uncertainty": float(unc[i]),
                    "confidence": float(1.0 - unc[i]),
                    "total_evidence": float(evidence[i]),
                    "alphas": alpha[i].tolist(),
                    "probs": probs[i].tolist(),
                }
            )

    return outputs

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import PAD_ID, VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    emb_dim: int = 64
    conv_channels: int = 128
    kernel_sizes: tuple[int, ...] = (3, 5, 7)
    hidden_dim: int = 128
    dropout: float = 0.2
    num_classes: int = 2


class EvidentialToxModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(cfg.emb_dim, cfg.conv_channels, kernel_size=k, padding=k // 2) for k in cfg.kernel_sizes]
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.conv_channels * len(cfg.kernel_sizes), cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.embedding(input_ids)  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]

        pooled: list[torch.Tensor] = []
        for conv in self.convs:
            h = F.relu(conv(x))  # [B, C, L]
            if attention_mask is not None:
                h = h.masked_fill(~attention_mask.unsqueeze(1), torch.finfo(h.dtype).min)
            pooled.append(torch.max(h, dim=-1).values)  # [B, C]

        feats = torch.cat(pooled, dim=-1)
        logits = self.head(feats)  # [B, K]
        alphas = F.softplus(logits) + 1.0
        return {"logits": logits, "alphas": alphas}

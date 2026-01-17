from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tan_fusion import TAN


@dataclass(frozen=True)
class FusionV2Config:
    plm_dim: int = 480
    phys_dim: int = 33
    motif_dim: int = 512
    adapter_dim: int = 64
    fusion: str = "tan"  # tan | trilinear
    tan_heads: int = 2
    dropout: float = 0.2
    head_hidden: int = 256
    num_classes: int = 2


class EvidentialFusionV2(nn.Module):
    """
    v2 fusion model (sequence-level):
      - PLM pooled embedding (frozen, cached) + adapter
      - physchem global features + adapter
      - motif count/frequency features + adapter
      - fusion: TAN (tri-attention) or trilinear gating (ablation)
      - evidential Dirichlet head (alphas)
    """

    def __init__(self, cfg: FusionV2Config):
        super().__init__()
        self.cfg = cfg

        fusion = str(cfg.fusion).lower().strip()
        if fusion not in {"tan", "trilinear"}:
            raise ValueError("cfg.fusion must be one of: tan, trilinear")

        self.plm_norm = nn.LayerNorm(int(cfg.plm_dim))
        self.phys_norm = nn.LayerNorm(int(cfg.phys_dim))
        self.motif_norm = nn.LayerNorm(int(cfg.motif_dim))

        self.plm_adapter = nn.Sequential(
            nn.Linear(int(cfg.plm_dim), int(cfg.adapter_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
        )
        self.phys_adapter = nn.Sequential(
            nn.Linear(int(cfg.phys_dim), int(cfg.adapter_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
        )
        self.motif_adapter = nn.Sequential(
            nn.Linear(int(cfg.motif_dim), int(cfg.adapter_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
        )

        self.fusion = fusion
        if self.fusion == "tan":
            self.tan = TAN((1, 1, 1), int(cfg.adapter_dim), int(cfg.tan_heads), dropouts=(cfg.dropout, cfg.dropout, cfg.dropout))
            fused_dim = int(cfg.adapter_dim)
            self.tri_mlp = None
        else:
            self.tan = None
            fused_dim = int(cfg.adapter_dim)
            self.tri_mlp = nn.Sequential(
                nn.Linear(int(cfg.adapter_dim) * 4, fused_dim),
                nn.ReLU(),
                nn.Dropout(float(cfg.dropout)),
            )

        self.head = nn.Sequential(
            nn.Linear(fused_dim, int(cfg.head_hidden)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.head_hidden), int(cfg.num_classes)),
        )

    def forward(
        self,
        plm_emb: torch.Tensor,  # [B, plm_dim]
        phys_feat: torch.Tensor,  # [B, phys_dim]
        motif_feat: torch.Tensor,  # [B, motif_dim]
        *,
        return_att_maps: bool = False,
    ) -> dict[str, torch.Tensor]:
        x = self.plm_adapter(self.plm_norm(plm_emb))
        y = self.phys_adapter(self.phys_norm(phys_feat))
        z = self.motif_adapter(self.motif_norm(motif_feat))

        att_maps = None
        if self.fusion == "tan":
            assert self.tan is not None
            fused, att_maps = self.tan(x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1), softmax=True)
        else:
            assert self.tri_mlp is not None
            tri = x * y * z
            fused = self.tri_mlp(torch.cat([x, y, z, tri], dim=-1))

        logits = self.head(fused)
        alphas = F.softplus(logits) + 1.0

        out: dict[str, torch.Tensor] = {"logits": logits, "alphas": alphas, "fused": fused}
        if return_att_maps:
            out["att_maps"] = att_maps if att_maps is not None else torch.empty(0)
        return out


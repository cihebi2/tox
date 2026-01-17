from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def apc(scores: torch.Tensor) -> torch.Tensor:
    """
    Average product correction for a symmetric score matrix.
    scores: [L, L]
    """
    a1 = scores.sum(dim=-1, keepdim=True)  # [L, 1]
    a2 = scores.sum(dim=-2, keepdim=True)  # [1, L]
    a12 = scores.sum().clamp_min(1e-8)
    return scores - (a1 * a2) / a12


def topk_edge_index(scores: torch.Tensor, top_k: int, *, add_self_loops: bool = True) -> torch.Tensor:
    """
    Build a sparse directed edge_index from a dense score matrix by row-wise top-k.
    Returns edge_index: [2, E] with src->dst edges.
    """
    if scores.dim() != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be square [L, L].")
    L = int(scores.shape[0])
    if L == 0:
        return torch.empty((2, 0), dtype=torch.long, device=scores.device)

    k = int(top_k)
    if k <= 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=scores.device)
    else:
        k = min(k, max(1, L - 1))
        s = scores.clone()
        s.fill_diagonal_(float("-inf"))
        nbr = torch.topk(s, k=k, dim=-1, largest=True).indices  # [L, k]
        src = torch.arange(L, device=scores.device).unsqueeze(1).expand(L, k).reshape(-1)
        dst = nbr.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)

    if add_self_loops:
        self_idx = torch.arange(L, device=scores.device, dtype=torch.long)
        self_edges = torch.stack([self_idx, self_idx], dim=0)
        edge_index = torch.cat([edge_index, self_edges], dim=1)

    return edge_index


def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Mean aggregation over incoming edges.
    x: [L, D]
    edge_index: [2, E] with src->dst
    """
    if edge_index.numel() == 0:
        return x
    src = edge_index[0]
    dst = edge_index[1]
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    deg = torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    out = out / deg.clamp_min(1.0).unsqueeze(-1)
    return out


@dataclass(frozen=True)
class GraphEncoderConfig:
    in_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


class MeanGraphEncoder(nn.Module):
    def __init__(self, cfg: GraphEncoderConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        dims = [int(cfg.in_dim)] + [int(cfg.hidden_dim)] * int(cfg.num_layers)
        for i in range(int(cfg.num_layers)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(float(cfg.dropout))

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = node_feats
        for lin in self.layers:
            m = mean_aggregate(h, edge_index)
            h = lin(m)
            h = F.relu(h)
            h = self.dropout(h)
        return h.mean(dim=0)  # graph embedding


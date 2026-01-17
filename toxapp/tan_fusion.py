from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """
    Simple MLP used inside TAN.

    This is a lightweight re-implementation inspired by the style used in PROTAC-STAN,
    but kept dependency-free for this repo.
    """

    def __init__(self, dims: list[int], *, act: str = "ReLU", dropout: float = 0.0):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output dim.")

        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            layers.append(weight_norm(nn.Linear(int(dims[i]), int(dims[i + 1])), dim=None))
            if act:
                layers.append(getattr(nn, act)())
        if dropout and dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(weight_norm(nn.Linear(int(dims[-2]), int(dims[-1])), dim=None))
        if act:
            layers.append(getattr(nn, act)())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TAN(nn.Module):
    """
    Tensor Attention Network (TAN) for tri-modal fusion.

    Expected inputs:
      x: [B, X, Dx]
      y: [B, Y, Dy]
      z: [B, Z, Dz]

    For vector modalities, a practical lightweight usage is:
      x = x_vec.unsqueeze(-1)  # [B, D, 1]
      y = y_vec.unsqueeze(-1)
      z = z_vec.unsqueeze(-1)
    so X=Y=Z=D and Dx=Dy=Dz=1.
    """

    def __init__(
        self,
        in_dims: tuple[int, int, int],
        h_dim: int,
        h_out: int,
        *,
        dropouts: tuple[float, float, float] = (0.2, 0.2, 0.2),
        act: str = "ReLU",
        k: int = 1,
    ):
        super().__init__()
        self.k = int(k)
        self.x_dim, self.y_dim, self.z_dim = map(int, in_dims)
        self.h_dim = int(h_dim)
        self.h_out = int(h_out)

        K = int(h_dim) * int(k)
        self.x_net = FCNet([self.x_dim, K], act=act, dropout=float(dropouts[0]))
        self.y_net = FCNet([self.y_dim, K], act=act, dropout=float(dropouts[1]))
        self.z_net = FCNet([self.z_dim, K], act=act, dropout=float(dropouts[2]))

        self.p_net = nn.AvgPool1d(kernel_size=self.k, stride=self.k) if self.k > 1 else None
        self.h_net = weight_norm(nn.Linear(K, self.h_out), dim=None)
        self.bn = nn.BatchNorm1d(self.h_dim)

    def _attention_pooling(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, att_map: torch.Tensor) -> torch.Tensor:
        # x: [B, X, K], y: [B, Y, K], z: [B, Z, K], att_map: [B, X, Y, Z]
        xy = torch.einsum("bxk,byk->bxyk", (x, y))  # [B, X, Y, K]
        xy = xy.permute(0, 2, 1, 3).contiguous()  # [B, Y, X, K]
        logits = torch.einsum("byxk,bxyz,bzk->bk", (xy, att_map, z))  # [B, K]
        if self.p_net is not None:
            logits = self.p_net(logits.unsqueeze(1)).squeeze(1) * float(self.k)  # [B, h_dim]
        return logits

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, *, softmax: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        x_num = int(x.size(1))
        y_num = int(y.size(1))
        z_num = int(z.size(1))

        _x = self.x_net(x)  # [B, X, K]
        _y = self.y_net(y)  # [B, Y, K]
        _z = self.z_net(z)  # [B, Z, K]

        _xyz = torch.einsum("bxk,byk,bzk->bxyzk", (_x, _y, _z))  # [B, X, Y, Z, K]
        att_maps = self.h_net(_xyz)  # [B, X, Y, Z, H]
        att_maps = att_maps.permute(0, 4, 1, 2, 3).contiguous()  # [B, H, X, Y, Z]

        if softmax:
            p = F.softmax(att_maps.view(-1, self.h_out, x_num * y_num * z_num), dim=2)
            att_maps = p.view(-1, self.h_out, x_num, y_num, z_num)

        logits = self._attention_pooling(_x, _y, _z, att_maps[:, 0, :, :, :])
        for i in range(1, self.h_out):
            logits = logits + self._attention_pooling(_x, _y, _z, att_maps[:, i, :, :, :])

        logits = self.bn(logits)
        return logits, att_maps


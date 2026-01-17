from __future__ import annotations

import torch


def dirichlet_kl_divergence(alpha: torch.Tensor, beta: torch.Tensor | None = None) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(beta)) per sample.

    alpha: [B, K]
    beta:  [B, K] or [K] (defaults to uniform Dirichlet with all ones)
    returns: [B, 1]
    """
    if beta is None:
        beta = torch.ones_like(alpha)
    elif beta.dim() == 1:
        beta = beta.unsqueeze(0).expand_as(alpha)

    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_beta = torch.sum(beta, dim=-1, keepdim=True)

    ln_b_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    ln_b_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(sum_beta)

    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha)

    kl = ln_b_alpha + ln_b_beta + torch.sum((alpha - beta) * (digamma_alpha - digamma_sum_alpha), dim=-1, keepdim=True)
    return kl


def dirichlet_evidence_loss(
    y_one_hot: torch.Tensor,  # [B, K]
    alpha: torch.Tensor,  # [B, K]
    lam: float = 1.0,
    *,
    class_weights: torch.Tensor | None = None,  # [K]
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """
    Evidential Dirichlet classification loss (Sensoy-style), returning a scalar.

    y_one_hot is expected to be float in {0,1}.
    """
    y_one_hot = y_one_hot.to(dtype=alpha.dtype)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    probs = alpha / sum_alpha

    sos_a = torch.sum((y_one_hot - probs) ** 2, dim=-1, keepdim=True)
    sos_b = torch.sum((probs * (1.0 - probs)) / (sum_alpha + 1.0), dim=-1, keepdim=True)
    sos = sos_a + sos_b  # [B, 1]

    alpha_hat = y_one_hot + (1.0 - y_one_hot) * alpha
    kl = dirichlet_kl_divergence(alpha_hat)

    data_term = sos

    if class_weights is not None:
        if class_weights.dim() != 1 or class_weights.shape[0] != alpha.shape[-1]:
            raise ValueError("class_weights must be a 1D tensor with shape [K].")
        w = torch.sum(y_one_hot * class_weights.to(dtype=alpha.dtype), dim=-1, keepdim=True)  # [B, 1]
        data_term = data_term * w

    if focal_gamma and focal_gamma > 0.0:
        p_true = torch.sum(y_one_hot * probs, dim=-1, keepdim=True).clamp_min(1e-8)  # [B, 1]
        data_term = data_term * (1.0 - p_true) ** float(focal_gamma)

    loss = data_term + lam * kl
    return loss.mean()


def alpha_to_probs(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / torch.sum(alpha, dim=-1, keepdim=True)


def alpha_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[-1]
    return k / torch.sum(alpha, dim=-1)


def alpha_total_evidence(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[-1]
    return torch.sum(alpha, dim=-1) - k

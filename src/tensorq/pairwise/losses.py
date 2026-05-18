from __future__ import annotations

import torch


def dirichlet_loss(q_t: torch.Tensor, q_tau: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    reduce_dtype = torch.float32 if q_t.dtype in {torch.float16, torch.bfloat16} else q_t.dtype
    diff = q_t.to(dtype=reduce_dtype) - q_tau.to(dtype=reduce_dtype)
    diff2 = diff.square().mean(dim=1)
    if weights is None:
        return diff2.mean()
    w = weights.to(dtype=reduce_dtype, device=q_t.device)
    return torch.sum(w * diff2) / torch.clamp(torch.sum(w), min=1e-12)


def endpoint_loss(q: torch.Tensor, pair_labels: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    labels = pair_labels.to(dtype=q.dtype, device=q.device)
    mask = (labels >= 0).to(dtype=q.dtype)
    target = labels.clamp(min=0.0)
    per_dim = (q - target).square() * mask
    per_sample = per_dim.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    valid = (mask.sum(dim=1) > 0).to(dtype=q.dtype)
    if weights is None:
        return torch.sum(per_sample * valid) / valid.sum().clamp_min(1.0)
    w = weights.to(dtype=q.dtype, device=q.device) * valid
    return torch.sum(w * per_sample) / torch.clamp(torch.sum(w), min=1e-12)


def total_pairwise_committor_loss(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    pair_label_t: torch.Tensor,
    pair_label_tau: torch.Tensor,
    weights: torch.Tensor | None = None,
    lambda_dirichlet: float = 1.0,
    lambda_endpoint: float = 100.0,
    weighted_dirichlet: bool = True,
    weighted_endpoint: bool = False,
) -> dict[str, torch.Tensor]:
    dirichlet_weights = weights if weighted_dirichlet else None
    d = dirichlet_loss(q_t, q_tau, weights=dirichlet_weights)
    endpoint_weights = weights if weighted_endpoint else None
    b = 0.5 * (
        endpoint_loss(q_t, pair_label_t, weights=endpoint_weights)
        + endpoint_loss(q_tau, pair_label_tau, weights=endpoint_weights)
    )
    total = float(lambda_dirichlet) * d + float(lambda_endpoint) * b
    return {"total_loss": total, "dirichlet_loss": d, "endpoint_loss": b}

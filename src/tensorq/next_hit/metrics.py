from __future__ import annotations

import torch


def mean_entropy(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(q.clamp_min(eps) * q.clamp_min(eps).log()).sum(dim=1).mean()


def normalization_error(q: torch.Tensor) -> torch.Tensor:
    return (q.sum(dim=-1) - 1.0).abs().max()


def boundary_accuracy(q: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device=q.device, dtype=torch.long)
    mask = labels >= 0
    if not torch.any(mask):
        return q.sum() * 0.0
    pred = q.argmax(dim=1)
    return (pred[mask] == labels[mask]).to(q.dtype).mean()


def endpoint_boundary_accuracy(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    state_t: torch.Tensor,
    state_tau: torch.Tensor,
) -> torch.Tensor:
    q = torch.cat([q_t, q_tau], dim=0)
    labels = torch.cat([state_t, state_tau], dim=0)
    return boundary_accuracy(q, labels)


def as_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)

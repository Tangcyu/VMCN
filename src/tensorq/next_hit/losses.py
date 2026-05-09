from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from ..common.flux import flux_consistency_loss


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    w = weights.to(device=values.device, dtype=values.dtype).reshape(values.shape[0], *([1] * (values.ndim - 1)))
    return (values * w).sum() / (w.sum() + 1e-12)


def dirichlet_loss(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    weights: torch.Tensor | None = None,
    tau: float | None = None,
    scale_by_tau: bool = False,
) -> torch.Tensor:
    per_sample = (q_tau - q_t).square().sum(dim=1)
    loss = weighted_mean(per_sample, weights)
    if scale_by_tau:
        if tau is None or float(tau) <= 0:
            raise ValueError("tau must be positive when scale_by_tau=True.")
        loss = loss / (2.0 * float(tau))
    return loss


def boundary_loss(
    q: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    mode: str = "cross_entropy",
    eps: float = 1e-12,
) -> torch.Tensor:
    labels = labels.to(device=q.device, dtype=torch.long)
    mask = labels >= 0
    if not torch.any(mask):
        return q.sum() * 0.0

    q_valid = q[mask]
    y = labels[mask]
    w_valid = weights.to(device=q.device, dtype=q.dtype)[mask] if weights is not None else None

    mode = str(mode).lower()
    if mode in {"cross_entropy", "ce", "nll"}:
        per_sample = -torch.log(q_valid.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(eps))
    elif mode == "mse":
        target = F.one_hot(y, num_classes=q.shape[1]).to(dtype=q.dtype)
        per_sample = (q_valid - target).square().sum(dim=1)
    else:
        raise ValueError("boundary mode must be 'cross_entropy' or 'mse'.")
    return weighted_mean(per_sample, w_valid)


def endpoint_boundary_loss(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    state_t: torch.Tensor,
    state_tau: torch.Tensor,
    weights: torch.Tensor | None = None,
    mode: str = "cross_entropy",
) -> torch.Tensor:
    q = torch.cat([q_t, q_tau], dim=0)
    labels = torch.cat([state_t, state_tau], dim=0)
    w = torch.cat([weights, weights], dim=0) if weights is not None else None
    return boundary_loss(q, labels, weights=w, mode=mode)


def total_committor_loss(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    state_t: torch.Tensor,
    state_tau: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    thresholds: torch.Tensor,
    flux_eps: float,
    sample_weights: torch.Tensor | None,
    lambda_dir: float = 1.0,
    lambda_bc: float = 10.0,
    lambda_flux: float = 0.1,
    boundary_mode: str = "cross_entropy",
    weighted_dirichlet: bool = True,
    weighted_boundary: bool = False,
    weighted_flux: bool = True,
    tau: float | None = None,
    scale_dirichlet_by_tau: bool = False,
    scale_flux_by_tau: bool = False,
    flux_surface: str = "qi_decrease",
) -> dict[str, torch.Tensor]:
    w_dir = sample_weights if weighted_dirichlet else None
    w_bc = sample_weights if weighted_boundary else None
    w_flux = sample_weights if weighted_flux else None

    loss_dir = dirichlet_loss(q_t, q_tau, weights=w_dir, tau=tau, scale_by_tau=scale_dirichlet_by_tau)
    loss_bc = endpoint_boundary_loss(q_t, q_tau, state_t, state_tau, weights=w_bc, mode=boundary_mode)
    if float(lambda_flux) == 0.0:
        loss_flux = q_t.sum() * 0.0
        J = torch.zeros((len(pairs), thresholds.numel()), dtype=q_t.dtype, device=q_t.device)
        flux_var = torch.zeros((len(pairs),), dtype=q_t.dtype, device=q_t.device)
    else:
        loss_flux, J, flux_var = flux_consistency_loss(
            q_t=q_t,
            q_tau=q_tau,
            pairs=pairs,
            thresholds=thresholds,
            eps=flux_eps,
            weights=w_flux,
            tau=tau,
            scale_by_tau=scale_flux_by_tau,
            surface=flux_surface,
        )

    total = float(lambda_dir) * loss_dir + float(lambda_bc) * loss_bc + float(lambda_flux) * loss_flux
    return {
        "total_loss": total,
        "dirichlet_loss": loss_dir,
        "boundary_loss": loss_bc,
        "flux_loss": loss_flux,
        "J": J,
        "flux_variance": flux_var,
    }

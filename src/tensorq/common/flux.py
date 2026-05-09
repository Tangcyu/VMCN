from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


def all_ordered_pairs(n_states: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(int(n_states)) for j in range(int(n_states)) if i != j]


def unordered_pairs(n_states: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(int(n_states)) for j in range(i + 1, int(n_states))]


def resolve_ordered_pairs(
    n_states: int,
    adjacency: Sequence[Sequence[int]] | None = None,
    symmetric_adjacency: bool = True,
) -> list[tuple[int, int]]:
    if adjacency is None:
        return all_ordered_pairs(n_states)

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw in adjacency:
        if len(raw) != 2:
            raise ValueError("Each adjacency entry must have exactly two state indices.")
        i, j = int(raw[0]), int(raw[1])
        if i == j:
            continue
        if not (0 <= i < n_states and 0 <= j < n_states):
            raise ValueError(f"Adjacency pair {(i, j)} is outside n_states={n_states}.")
        for pair in ([(i, j), (j, i)] if symmetric_adjacency else [(i, j)]):
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    if not pairs:
        raise ValueError("No valid ordered pairs were produced from adjacency.")
    return pairs


def make_thresholds(
    values: Iterable[float] | None,
    n_thresholds: int = 9,
    start: float = 0.1,
    stop: float = 0.9,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if values is not None:
        thresholds = torch.tensor(list(values), dtype=dtype, device=device)
    else:
        thresholds = torch.linspace(float(start), float(stop), int(n_thresholds), device=device, dtype=dtype)
    if thresholds.ndim != 1 or thresholds.numel() < 1:
        raise ValueError("At least one flux threshold is required.")
    if torch.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Flux thresholds must lie strictly inside (0, 1).")
    return thresholds


def pair_index_tensors(pairs: Sequence[tuple[int, int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pair_i = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    pair_j = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    return pair_i, pair_j


def reactive_current(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """C_ij = q_j(t+tau) q_i(t) - q_j(t) q_i(t+tau), shape (batch, n_pairs)."""
    pair_i, pair_j = pair_index_tensors(pairs, q_t.device)
    return q_tau[:, pair_j] * q_t[:, pair_i] - q_t[:, pair_j] * q_tau[:, pair_i]


def crossing_weight(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    thresholds: torch.Tensor,
    eps: float,
    surface: str = "qi_decrease",
) -> torch.Tensor:
    """
    Smooth isocommittor crossing indicator, shape (batch, n_pairs, n_thresholds).

    qi_decrease uses sigmoid((q_i(t)-c)/eps) * sigmoid((c-q_i(t+tau))/eps).
    qj_increase uses sigmoid((c-q_j(t))/eps) * sigmoid((q_j(t+tau)-c)/eps).
    """
    if eps <= 0:
        raise ValueError("flux eps must be positive.")
    pair_i, pair_j = pair_index_tensors(pairs, q_t.device)
    c = thresholds.to(device=q_t.device, dtype=q_t.dtype).view(1, 1, -1)
    surface = str(surface).lower()
    if surface == "qi_decrease":
        left = q_t[:, pair_i].unsqueeze(-1)
        right = q_tau[:, pair_i].unsqueeze(-1)
        return torch.sigmoid((left - c) / eps) * torch.sigmoid((c - right) / eps)
    if surface == "qj_increase":
        left = q_t[:, pair_j].unsqueeze(-1)
        right = q_tau[:, pair_j].unsqueeze(-1)
        return torch.sigmoid((c - left) / eps) * torch.sigmoid((right - c) / eps)
    raise ValueError("surface must be either 'qi_decrease' or 'qj_increase'.")


def flux_profiles(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    thresholds: torch.Tensor,
    eps: float,
    weights: torch.Tensor | None = None,
    tau: float | None = None,
    scale_by_tau: bool = False,
    surface: str = "qi_decrease",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return J_ij(c_m) and C_ij(t,t+tau).

    J has shape (n_pairs, n_thresholds). C has shape (batch, n_pairs).
    """
    c_ij = reactive_current(q_t, q_tau, pairs)
    chi = crossing_weight(q_t, q_tau, pairs, thresholds, eps=eps, surface=surface)
    samples = chi * c_ij.unsqueeze(-1)

    if weights is None:
        J = samples.mean(dim=0)
    else:
        w = weights.to(device=q_t.device, dtype=q_t.dtype).view(-1, 1, 1)
        J = (samples * w).sum(dim=0) / (w.sum() + 1e-12)

    if scale_by_tau:
        if tau is None or float(tau) <= 0:
            raise ValueError("tau must be positive when scale_by_tau=True.")
        J = J / float(tau)
    return J, c_ij


def flux_consistency_loss(
    q_t: torch.Tensor,
    q_tau: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    thresholds: torch.Tensor,
    eps: float,
    weights: torch.Tensor | None = None,
    tau: float | None = None,
    scale_by_tau: bool = False,
    surface: str = "qi_decrease",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    J, _ = flux_profiles(
        q_t=q_t,
        q_tau=q_tau,
        pairs=pairs,
        thresholds=thresholds,
        eps=eps,
        weights=weights,
        tau=tau,
        scale_by_tau=scale_by_tau,
        surface=surface,
    )
    if J.shape[1] < 2:
        var = torch.zeros(J.shape[0], dtype=J.dtype, device=J.device)
        return J.sum() * 0.0, J, var
    jbar = J.mean(dim=1, keepdim=True)
    per_pair = (J - jbar).square().mean(dim=1)
    return per_pair.mean(), J, per_pair


def flux_profiles_numpy(
    q_t: np.ndarray,
    q_tau: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    thresholds: np.ndarray,
    eps: float,
    weights: np.ndarray | None = None,
    tau: float | None = None,
    scale_by_tau: bool = False,
    surface: str = "qi_decrease",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy implementation used by plotting and post-training rate estimates."""
    q_t = np.asarray(q_t, dtype=np.float64)
    q_tau = np.asarray(q_tau, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    pair_i = np.asarray([p[0] for p in pairs], dtype=np.int64)
    pair_j = np.asarray([p[1] for p in pairs], dtype=np.int64)

    C = q_tau[:, pair_j] * q_t[:, pair_i] - q_t[:, pair_j] * q_tau[:, pair_i]
    c = thresholds.reshape(1, 1, -1)
    if surface == "qi_decrease":
        left = q_t[:, pair_i][:, :, None]
        right = q_tau[:, pair_i][:, :, None]
        chi = 1.0 / (1.0 + np.exp(-(left - c) / eps))
        chi *= 1.0 / (1.0 + np.exp(-(c - right) / eps))
    elif surface == "qj_increase":
        left = q_t[:, pair_j][:, :, None]
        right = q_tau[:, pair_j][:, :, None]
        chi = 1.0 / (1.0 + np.exp(-(c - left) / eps))
        chi *= 1.0 / (1.0 + np.exp(-(right - c) / eps))
    else:
        raise ValueError("surface must be either 'qi_decrease' or 'qj_increase'.")

    samples = chi * C[:, :, None]
    if weights is None:
        J = np.mean(samples, axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1, 1)
        J = np.sum(w * samples, axis=0) / (np.sum(w) + 1e-300)
    if scale_by_tau:
        if tau is None or float(tau) <= 0:
            raise ValueError("tau must be positive when scale_by_tau=True.")
        J = J / float(tau)
    var = np.mean((J - J.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return J.astype(np.float64), var.astype(np.float64), C.astype(np.float64)

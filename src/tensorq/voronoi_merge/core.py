from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class VoronoiAssignment:
    labels: np.ndarray
    distances: np.ndarray
    probabilities: np.ndarray


def as_points(array: np.ndarray, *, name: str = "array") -> np.ndarray:
    out = np.asarray(array, dtype=np.float64)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError(f"{name} must have shape (n_points, n_dim).")
    if out.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values.")
    return out


def normalize_periods(periods: Sequence[float | None] | None, ndim: int) -> list[float | None]:
    if periods is None or periods is False:
        return [None] * int(ndim)
    out = [None if item is None else float(item) for item in periods]
    if len(out) != int(ndim):
        raise ValueError(f"periods must have length {int(ndim)}.")
    return [period if period is not None and period > 0.0 else None for period in out]


def minimum_image_delta(diff: np.ndarray, periods: Sequence[float | None] | None) -> np.ndarray:
    out = np.asarray(diff, dtype=np.float64).copy()
    period_list = normalize_periods(periods, out.shape[-1])
    for dim, period in enumerate(period_list):
        if period is not None:
            out[..., dim] = ((out[..., dim] + 0.5 * period) % period) - 0.5 * period
    return out


def _torch_dtype(dtype: str):
    import torch

    name = str(dtype).lower()
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"float32", "single"}:
        return torch.float32
    raise ValueError("dtype must be 'float32' or 'float64'.")


def _resolve_torch_device(device: str | None):
    if device is None or str(device).lower() in {"numpy", "np", "none", ""}:
        return None
    import torch

    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _use_sincos_geometry(
    periodic_geometry: str | None,
    periods: Sequence[float | None] | None,
    ndim: int,
) -> bool:
    mode = str(periodic_geometry or "minimum_image").lower()
    if mode in {"minimum_image", "periodic", "raw", "unwrap", "none"}:
        return False
    if mode not in {"sincos", "sin_cos", "embedding", "periodic_embedding", "circular"}:
        raise ValueError("periodic_geometry must be 'minimum_image' or 'sincos'.")
    return any(period is not None for period in normalize_periods(periods, ndim))


def _minimum_image_delta_torch(diff, periods: Sequence[float | None] | None):
    import torch

    period_list = normalize_periods(periods, diff.shape[-1])
    out = diff
    for dim, period in enumerate(period_list):
        if period is not None:
            out_dim = torch.remainder(out[..., dim] + 0.5 * period, period) - 0.5 * period
            out = out.clone()
            out[..., dim] = out_dim
    return out


def _assign_voronoi_cells_torch(
    points: np.ndarray,
    centers: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    chunk_size: int = 65536,
    device: str | None = "cuda:0",
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch_device = _resolve_torch_device(device)
    if torch_device is None:
        raise ValueError("torch device must not be None.")
    torch_dtype = _torch_dtype(dtype)
    points_arr = as_points(points, name="points")
    centers_arr = as_points(centers, name="centers")
    if points_arr.shape[1] != centers_arr.shape[1]:
        raise ValueError("points and centers must have the same dimension.")
    period_list = normalize_periods(periods, points_arr.shape[1])
    chunk = max(1, int(chunk_size))
    labels = np.empty(points_arr.shape[0], dtype=np.int64)
    distances = np.empty(points_arr.shape[0], dtype=np.float64)
    centers_t = torch.as_tensor(centers_arr, dtype=torch_dtype, device=torch_device)
    with torch.no_grad():
        for start in range(0, points_arr.shape[0], chunk):
            stop = min(start + chunk, points_arr.shape[0])
            points_t = torch.as_tensor(points_arr[start:stop], dtype=torch_dtype, device=torch_device)
            diff = points_t[:, None, :] - centers_t[None, :, :]
            if any(period is not None for period in period_list):
                diff = _minimum_image_delta_torch(diff, period_list)
            dist2 = torch.sum(diff * diff, dim=2)
            values, local_labels = torch.min(dist2, dim=1)
            labels[start:stop] = local_labels.detach().cpu().numpy().astype(np.int64, copy=False)
            distances[start:stop] = torch.sqrt(values).detach().cpu().numpy().astype(np.float64, copy=False)
    return labels, distances


def normalize_weights(weights: np.ndarray | None, n_items: int, *, name: str = "weights") -> np.ndarray:
    if weights is None:
        return np.ones(int(n_items), dtype=np.float64)
    out = np.asarray(weights, dtype=np.float64).reshape(-1)
    if out.shape[0] != int(n_items):
        raise ValueError(f"{name} length must match n_items={int(n_items)}.")
    if np.any(out < 0.0) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite nonnegative values.")
    return out


def assign_voronoi_cells(
    points: np.ndarray,
    centers: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    periodic_geometry: str | None = "minimum_image",
    chunk_size: int = 65536,
    device: str | None = None,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point to its nearest Voronoi image."""

    points_arr = as_points(points, name="points")
    centers_arr = as_points(centers, name="centers")
    if points_arr.shape[1] != centers_arr.shape[1]:
        raise ValueError("points and centers must have the same dimension.")
    if _use_sincos_geometry(periodic_geometry, periods, points_arr.shape[1]):
        points_arr = periodic_sincos_embed(points_arr, periods=periods)
        centers_arr = periodic_sincos_embed(centers_arr, periods=periods)
        periods = None

    if _resolve_torch_device(device) is not None:
        return _assign_voronoi_cells_torch(
            points_arr,
            centers_arr,
            periods=periods,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
        )

    period_list = normalize_periods(periods, points_arr.shape[1])
    chunk = max(1, int(chunk_size))
    labels = np.empty(points_arr.shape[0], dtype=np.int64)
    distances = np.empty(points_arr.shape[0], dtype=np.float64)
    for start in range(0, points_arr.shape[0], chunk):
        stop = min(start + chunk, points_arr.shape[0])
        diff = points_arr[start:stop, None, :] - centers_arr[None, :, :]
        diff = minimum_image_delta(diff, period_list)
        dist2 = np.sum(diff * diff, axis=2)
        local_labels = np.argmin(dist2, axis=1)
        labels[start:stop] = local_labels
        distances[start:stop] = np.sqrt(dist2[np.arange(stop - start), local_labels])
    return labels, distances


def cell_probabilities(
    labels: np.ndarray,
    n_cells: int,
    *,
    weights: np.ndarray | None = None,
    pseudocount: float = 0.0,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(n_cells)
    if n < 1:
        raise ValueError("n_cells must be positive.")
    if labels_arr.size and (labels_arr.min() < 0 or labels_arr.max() >= n):
        raise ValueError("labels contain values outside [0, n_cells).")
    weights_arr = normalize_weights(weights, labels_arr.size)
    counts = np.bincount(labels_arr, weights=weights_arr, minlength=n).astype(np.float64)
    if pseudocount < 0.0:
        raise ValueError("pseudocount must be nonnegative.")
    counts += float(pseudocount)
    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("Cannot normalize zero total probability mass.")
    return counts / total


def voronoi_assignment(
    points: np.ndarray,
    centers: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    periods: Sequence[float | None] | None = None,
    periodic_geometry: str | None = "minimum_image",
    pseudocount: float = 0.0,
    chunk_size: int = 65536,
    device: str | None = None,
    dtype: str = "float32",
) -> VoronoiAssignment:
    labels, distances = assign_voronoi_cells(
        points,
        centers,
        periods=periods,
        periodic_geometry=periodic_geometry,
        chunk_size=chunk_size,
        device=device,
        dtype=dtype,
    )
    probabilities = cell_probabilities(labels, len(centers), weights=weights, pseudocount=pseudocount)
    return VoronoiAssignment(labels=labels, distances=distances, probabilities=probabilities)


def kl_divergence(p_current: np.ndarray, p_previous: np.ndarray, *, eps: float = 1.0e-12) -> float:
    p = np.asarray(p_current, dtype=np.float64).reshape(-1)
    q = np.asarray(p_previous, dtype=np.float64).reshape(-1)
    if p.shape != q.shape:
        raise ValueError("p_current and p_previous must have the same shape.")
    if np.any(p < 0.0) or np.any(q < 0.0):
        raise ValueError("probabilities must be nonnegative.")
    if p.sum() <= 0.0 or q.sum() <= 0.0:
        raise ValueError("probabilities must have positive total mass.")
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0.0
    q_safe = np.maximum(q[mask], float(eps))
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe)))


def periodic_weighted_mean(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    points_arr = as_points(points, name="points")
    weights_arr = normalize_weights(weights, points_arr.shape[0])
    if weights_arr.sum() <= 0.0:
        weights_arr = np.ones_like(weights_arr)
    period_list = normalize_periods(periods, points_arr.shape[1])
    ref = points_arr[0] if reference is None else np.asarray(reference, dtype=np.float64).reshape(-1)
    if ref.shape[0] != points_arr.shape[1]:
        raise ValueError("reference dimension must match points.")
    aligned = ref[None, :] + minimum_image_delta(points_arr - ref[None, :], period_list)
    return np.average(aligned, axis=0, weights=weights_arr)


def periodic_sincos_embed(
    points: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
) -> np.ndarray:
    """Embed periodic coordinates as scaled sin/cos pairs.

    The scale factor period / 2pi makes small angular displacements comparable
    to the original coordinate units.
    """

    points_arr = np.asarray(points, dtype=np.float64)
    if points_arr.ndim < 1:
        raise ValueError("points must have at least one dimension.")
    period_list = normalize_periods(periods, points_arr.shape[-1])
    pieces = []
    for dim, period in enumerate(period_list):
        coord = points_arr[..., dim]
        if period is None:
            pieces.append(coord[..., None])
            continue
        angle = 2.0 * np.pi * coord / float(period)
        scale = float(period) / (2.0 * np.pi)
        pieces.append((scale * np.sin(angle))[..., None])
        pieces.append((scale * np.cos(angle))[..., None])
    return np.concatenate(pieces, axis=-1)


def periodic_sincos_project(
    embedded: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
) -> np.ndarray:
    """Project scaled sin/cos periodic coordinates back to raw coordinates."""

    emb = np.asarray(embedded, dtype=np.float64)
    if emb.ndim < 1:
        raise ValueError("embedded points must have at least one dimension.")
    if periods is None:
        return emb.copy()
    out = []
    cursor = 0
    for period in normalize_periods(periods, len(periods)):
        if period is None:
            if cursor >= emb.shape[-1]:
                raise ValueError("embedded dimension is too small for periods.")
            out.append(emb[..., cursor][..., None])
            cursor += 1
            continue
        if cursor + 1 >= emb.shape[-1]:
            raise ValueError("embedded dimension is too small for periodic sin/cos pairs.")
        sin_value = emb[..., cursor]
        cos_value = emb[..., cursor + 1]
        angle = np.arctan2(sin_value, cos_value)
        out.append((angle * float(period) / (2.0 * np.pi))[..., None])
        cursor += 2
    if cursor != emb.shape[-1]:
        raise ValueError("embedded dimension does not match periods.")
    return np.concatenate(out, axis=-1)


def periodic_sincos_weighted_mean(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
) -> np.ndarray:
    """Average periodic coordinates in sin/cos space and reproject."""

    points_arr = as_points(points, name="points")
    weights_arr = normalize_weights(weights, points_arr.shape[0])
    if weights_arr.sum() <= 0.0:
        weights_arr = np.ones_like(weights_arr)
    embedded = periodic_sincos_embed(points_arr, periods=periods)
    center = np.average(embedded, axis=0, weights=weights_arr)
    return periodic_sincos_project(center[None, :], periods=periods)[0]

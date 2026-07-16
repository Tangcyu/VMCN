from __future__ import annotations

import math
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def resolve_device(name: str | None):
    if torch is None:
        return None
    if name is None or name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; using CPU.")
        name = "cpu"
    return torch.device(name)


def _chunked_mean_std(
    X: np.ndarray,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute column-wise mean and std in chunks, avoiding a full-array copy."""
    n, d = X.shape
    if n == 0:
        return np.zeros(d, dtype=np.float64), np.ones(d, dtype=np.float64)

    # Pass 1: mean
    sum_x = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        chunk = np.asarray(X[start:stop], dtype=np.float64)
        sum_x += chunk.sum(axis=0)
    mean = sum_x / n

    # Pass 2: std (two-pass for numerical stability)
    sum_sq = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        chunk = np.asarray(X[start:stop], dtype=np.float64)
        diff = chunk - mean
        sum_sq += (diff * diff).sum(axis=0)
    var = np.maximum(sum_sq / n, 0.0)
    std = np.sqrt(var)
    return mean, std


def compute_standardization_params(
    X: np.ndarray,
    method: str = "standard",
    eps: float = 1e-8,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute standardization center and scale in chunks.

    Returns (center, scale) as float64 arrays that can be applied later,
    e.g. per-batch inside a training loop.
    """
    if method == "none":
        return np.zeros(X.shape[1], dtype=np.float64), np.ones(X.shape[1], dtype=np.float64)

    if method == "robust":
        # robust stats are harder to chunk; fall back to single-pass load
        X64 = np.asarray(X, dtype=np.float64)
        center = np.nanmedian(X64, axis=0)
        q25, q75 = np.nanpercentile(X64, [25, 75], axis=0)
        scale = q75 - q25
    elif method == "range":
        X64 = np.asarray(X, dtype=np.float64)
        center = np.nanmin(X64, axis=0)
        scale = np.nanmax(X64, axis=0) - center
    else:
        center, scale = _chunked_mean_std(X, chunk_size=chunk_size)

    scale = np.where(np.abs(scale) < eps, 1.0, scale)
    return center.astype(np.float64), scale.astype(np.float64)


def standardize(
    X: np.ndarray,
    method: str = "standard",
    eps: float = 1e-8,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize X to zero mean, unit variance (or other methods).

    Computes statistics and applies the transform in chunks so that the
    full dataset is never upcast to float64 at once.
    """
    n, d = X.shape

    if method == "none":
        center = np.zeros(d, dtype=np.float64)
        scale = np.ones(d, dtype=np.float64)
        return X.astype(np.float32, copy=False), center, scale

    # Compute statistics
    center, scale = compute_standardization_params(X, method=method, eps=eps, chunk_size=chunk_size)

    # Apply transformation in chunks to a pre-allocated float32 output
    out = np.empty((n, d), dtype=np.float32)
    scale_safe = np.where(np.abs(scale) < eps, 1.0, scale)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        chunk = np.asarray(X[start:stop], dtype=np.float64)
        out[start:stop] = ((chunk - center) / scale_safe).astype(np.float32)

    return out, center.astype(np.float64), scale.astype(np.float64)

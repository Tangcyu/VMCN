from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .checkpoint import exists, load_npz, save_npz
from .config import ensure_dir, stage_path
from .data import modeling_coordinate_matrix
from .tensor import compute_standardization_params, resolve_device


def _normalized_sampling_weights(weights: np.ndarray) -> np.ndarray | None:
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf)
    total = float(w.sum())
    if total <= 0.0:
        return None
    return w / total


def _initial_indices(rng: np.random.Generator, n_frames: int, n_clusters: int, weights: np.ndarray | None) -> np.ndarray:
    p = _normalized_sampling_weights(weights) if weights is not None else None
    if p is None or int(np.count_nonzero(p > 0.0)) < n_clusters:
        return rng.choice(n_frames, size=n_clusters, replace=False)
    return rng.choice(n_frames, size=n_clusters, replace=False, p=p)


def _apply_norm(
    chunk: np.ndarray,
    center: np.ndarray | None,
    scale: np.ndarray | None,
) -> np.ndarray:
    """Apply (x - center) / scale; returns float32."""
    if center is None and scale is None:
        return np.asarray(chunk, dtype=np.float32)
    out = np.asarray(chunk, dtype=np.float32).copy()
    if center is not None:
        out -= np.asarray(center, dtype=np.float32)
    if scale is not None:
        scale32 = np.asarray(np.where(np.abs(scale) < 1e-8, 1.0, scale), dtype=np.float32)
        out /= scale32
    return out.astype(np.float32, copy=False)


def torch_kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    weights: np.ndarray | None,
    device_name: str,
    seed: int,
    max_iter: int,
    batch_size: int,
    norm_center: np.ndarray | None = None,
    norm_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None:
        return numpy_kmeans(
            X, n_clusters,
            weights=weights, seed=seed, max_iter=max_iter, batch_size=batch_size,
            norm_center=norm_center, norm_scale=norm_scale,
        )
    rng = np.random.default_rng(seed)
    if n_clusters >= X.shape[0]:
        raise SystemExit(f"n_microstates ({n_clusters}) must be smaller than frame count ({X.shape[0]}).")
    init_idx = _initial_indices(rng, X.shape[0], n_clusters, weights)
    device = resolve_device(device_name)

    # Initialise centres in *standardised* space
    centers_np = _apply_norm(X[init_idx], norm_center, norm_scale)
    centers = torch.as_tensor(centers_np, dtype=torch.float32, device=device)

    wt_cpu = None if weights is None else torch.as_tensor(weights, dtype=torch.float32)
    do_norm = norm_center is not None or norm_scale is not None

    labels = np.zeros(X.shape[0], dtype=np.int64)
    with torch.no_grad():
        for _ in range(max_iter):
            counts = torch.zeros(n_clusters, dtype=torch.float32, device=device)
            sums = torch.zeros((n_clusters, X.shape[1]), dtype=torch.float32, device=device)
            centers_norm = torch.sum(centers * centers, dim=1, keepdim=True).T
            changed = 0
            for start in range(0, X.shape[0], batch_size):
                stop = min(X.shape[0], start + batch_size)
                xb_np = X[start:stop]
                if do_norm:
                    xb_np = _apply_norm(xb_np, norm_center, norm_scale)
                xb = torch.as_tensor(xb_np, dtype=torch.float32, device=device)
                d2 = torch.sum(xb * xb, dim=1, keepdim=True) + centers_norm - 2.0 * (xb @ centers.T)
                d2.clamp_min_(0.0)
                lab = torch.argmin(d2, dim=1)
                old = torch.as_tensor(labels[start:stop], dtype=torch.long, device=device)
                changed += int((lab != old).sum().item())
                labels[start:stop] = lab.detach().cpu().numpy()
                wb = torch.ones_like(lab, dtype=torch.float32) if wt_cpu is None else wt_cpu[start:stop].to(device, non_blocking=True)
                counts.scatter_add_(0, lab, wb)
                sums.scatter_add_(0, lab[:, None].expand(-1, X.shape[1]), xb * wb[:, None])
            nonempty = counts > 0
            centers[nonempty] = sums[nonempty] / counts[nonempty, None]
            if changed == 0:
                break
    return labels, centers.detach().cpu().numpy()


def assign_to_centers_torch(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    device_name: str,
    batch_size: int,
    norm_center: np.ndarray | None = None,
    norm_scale: np.ndarray | None = None,
) -> np.ndarray:
    if torch is None:
        return assign_to_centers_numpy(
            X,
            centers,
            batch_size=batch_size,
            norm_center=norm_center,
            norm_scale=norm_scale,
        )
    device = resolve_device(device_name)
    centers_t = torch.as_tensor(np.asarray(centers, dtype=np.float32), dtype=torch.float32, device=device)
    labels = np.empty(X.shape[0], dtype=np.int64)
    do_norm = norm_center is not None or norm_scale is not None
    with torch.no_grad():
        centers_norm = torch.sum(centers_t * centers_t, dim=1, keepdim=True).T
        for start in range(0, X.shape[0], int(batch_size)):
            stop = min(X.shape[0], start + int(batch_size))
            xb_np = X[start:stop]
            if do_norm:
                xb_np = _apply_norm(xb_np, norm_center, norm_scale)
            xb = torch.as_tensor(xb_np, dtype=torch.float32, device=device)
            d2 = torch.sum(xb * xb, dim=1, keepdim=True) + centers_norm - 2.0 * (xb @ centers_t.T)
            d2.clamp_min_(0.0)
            labels[start:stop] = torch.argmin(d2, dim=1).detach().cpu().numpy()
    return labels


def assign_to_centers_numpy(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    batch_size: int,
    norm_center: np.ndarray | None = None,
    norm_scale: np.ndarray | None = None,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    center_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    labels = np.empty(X.shape[0], dtype=np.int64)
    do_norm = norm_center is not None or norm_scale is not None
    for start in range(0, X.shape[0], int(batch_size)):
        stop = min(X.shape[0], start + int(batch_size))
        xb = X[start:stop]
        if do_norm:
            xb = _apply_norm(xb, norm_center, norm_scale)
        xb = np.asarray(xb, dtype=np.float32)
        d2 = np.sum(xb * xb, axis=1, keepdims=True) + center_norm - 2.0 * (xb @ centers.T)
        np.maximum(d2, 0.0, out=d2)
        labels[start:stop] = np.argmin(d2, axis=1)
    return labels


def sampled_kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    weights: np.ndarray | None,
    device_name: str,
    seed: int,
    max_iter: int,
    batch_size: int,
    train_size: int,
    assign_batch_size: int,
    norm_center: np.ndarray | None = None,
    norm_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if n_clusters >= X.shape[0]:
        raise SystemExit(f"n_microstates ({n_clusters}) must be smaller than frame count ({X.shape[0]}).")
    train_size = int(max(n_clusters + 1, min(int(train_size), X.shape[0])))
    train_idx = _initial_indices(rng, X.shape[0], train_size, weights)
    train_idx.sort()
    print(
        f"[info] sampled k-means training on {train_size}/{X.shape[0]} frames, then assigning all frames.",
        flush=True,
    )
    X_train = np.asarray(X[train_idx], dtype=np.float32)
    train_weights = None if weights is None else np.asarray(weights[train_idx], dtype=np.float64)
    _, centers = torch_kmeans(
        X_train,
        n_clusters,
        weights=train_weights,
        device_name=device_name,
        seed=seed,
        max_iter=max_iter,
        batch_size=batch_size,
        norm_center=norm_center,
        norm_scale=norm_scale,
    )
    labels = assign_to_centers_torch(
        X,
        centers,
        device_name=device_name,
        batch_size=assign_batch_size,
        norm_center=norm_center,
        norm_scale=norm_scale,
    )
    return labels, centers, train_idx


def numpy_kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    weights: np.ndarray | None,
    seed: int,
    max_iter: int,
    batch_size: int,
    norm_center: np.ndarray | None = None,
    norm_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if n_clusters >= X.shape[0]:
        raise SystemExit(f"n_microstates ({n_clusters}) must be smaller than frame count ({X.shape[0]}).")
    init_idx = _initial_indices(rng, X.shape[0], n_clusters, weights)
    centers = _apply_norm(X[init_idx], norm_center, norm_scale)
    labels = np.zeros(X.shape[0], dtype=np.int64)
    print("[warn] torch is not installed; using NumPy CPU k-means. Install torch for CUDA acceleration.")
    w_all = None if weights is None else np.asarray(weights, dtype=np.float64)
    do_norm = norm_center is not None or norm_scale is not None
    n_features = X.shape[1]
    for _ in range(max_iter):
        changed = 0
        sums = np.zeros_like(centers, dtype=np.float64)
        counts = np.zeros(n_clusters, dtype=np.float64)
        for start in range(0, X.shape[0], batch_size):
            stop = min(X.shape[0], start + batch_size)
            xb = X[start:stop]
            if do_norm:
                xb = _apply_norm(xb, norm_center, norm_scale)
            d2 = np.sum((xb[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            lab = np.argmin(d2, axis=1)
            changed += int(np.count_nonzero(lab != labels[start:stop]))
            labels[start:stop] = lab
            wb = np.ones(len(lab), dtype=np.float64) if w_all is None else w_all[start:stop]
            np.add.at(sums, lab, xb.astype(np.float64) * wb[:, None])
            np.add.at(counts, lab, wb)
        nonempty = counts > 0
        centers[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
        if changed == 0:
            break
    return labels, centers


def cluster_microstates(cfg: Dict, table: pd.DataFrame) -> Dict[str, np.ndarray]:
    force = bool(cfg["project"].get("force", False))
    out_npz = stage_path(cfg, "02_microstates", "microstates.npz")
    if exists(out_npz, force=force):
        print(f"[reuse] microstates: {out_npz}")
        return load_npz(out_npz)

    ensure_dir(stage_path(cfg, "02_microstates"))
    c_cfg = cfg["clustering"]
    X_raw, feature_names, feature_meta = modeling_coordinate_matrix(cfg, table)
    use_weights = bool(c_cfg.get("use_weights", cfg.get("msm", {}).get("use_weights", True)))
    sample_weights = None
    if use_weights:
        weight_column = cfg["data"].get("weight_column", "weight")
        if weight_column in table.columns:
            sample_weights = pd.to_numeric(table[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
        else:
            sample_weights = np.ones(len(table), dtype=np.float64)
        if float(sample_weights.sum()) <= 0.0:
            sample_weights = np.ones(len(table), dtype=np.float64)

    # Compute standardisation parameters in chunks without creating a full copy.
    norm_method = cfg.get("normalization", {}).get("method", "standard")
    norm_eps = float(cfg.get("normalization", {}).get("eps", 1e-8))
    center, scale = compute_standardization_params(X_raw, method=norm_method, eps=norm_eps)

    method = str(c_cfg.get("method", c_cfg.get("algorithm", "exact"))).lower()
    device_name = str(c_cfg.get("device", cfg["project"].get("device", "cuda")))
    seed = int(cfg["project"].get("seed", 2026))
    max_iter = int(c_cfg.get("max_iter", 100))
    batch_size = int(c_cfg.get("batch_size", 65536))
    train_indices = np.asarray([], dtype=np.int64)
    if method in {"sampled", "sample", "minibatch", "mini_batch"}:
        labels, centers, train_indices = sampled_kmeans(
            X_raw,
            int(c_cfg["n_microstates"]),
            weights=sample_weights,
            device_name=device_name,
            seed=seed,
            max_iter=max_iter,
            batch_size=batch_size,
            train_size=int(c_cfg.get("train_size", c_cfg.get("sample_size", 50000))),
            assign_batch_size=int(c_cfg.get("assign_batch_size", batch_size)),
            norm_center=center,
            norm_scale=scale,
        )
    elif method == "exact":
        labels, centers = torch_kmeans(
            X_raw,
            int(c_cfg["n_microstates"]),
            weights=sample_weights,
            device_name=device_name,
            seed=seed,
            max_iter=max_iter,
            batch_size=batch_size,
            norm_center=center,
            norm_scale=scale,
        )
    else:
        raise SystemExit("clustering.method must be 'exact' or 'sampled'.")
    result = {
        "labels": labels,
        "centers": centers,
        "norm_center": center,
        "norm_scale": scale,
        "feature_names": np.asarray(feature_names, dtype=object),
        "train_indices": train_indices,
    }
    save_npz(
        out_npz,
        manifest={
            "stage": "microstates",
            "n_microstates": int(c_cfg["n_microstates"]),
            "feature_names": feature_names,
            "feature_source": feature_meta.get("source", "cvs"),
            "feature_meta": feature_meta,
            "projection_cvs": cfg["data"]["cvs"],
            "use_weights": use_weights,
            "weight_column": cfg["data"].get("weight_column", "weight") if use_weights else None,
            "method": method,
            "train_size": int(train_indices.size) if train_indices.size else None,
        },
        **result,
    )
    print(f"[ok] microstates: {out_npz}")
    return result

from __future__ import annotations

import os
from itertools import combinations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .checkpoint import exists, save_npz
from .config import ensure_dir, stage_path
from .data import core_coordinate_matrix, output_feature_coordinate_matrix
from .tensor import resolve_device


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf)
    total = float(w.sum())
    if total <= 0.0:
        return np.full(w.shape[0], 1.0 / max(1, w.shape[0]), dtype=np.float64)
    return w / total


def _standardize(X: np.ndarray, eps: float = 1e-8, fit_mask: np.ndarray | None = None, chunk_size: int = 65536) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunked standardisation – never upcasts the full array to float64 at once."""
    mean, scale = _standardization_params(X, eps=eps, fit_mask=fit_mask, chunk_size=chunk_size)
    out = np.empty(X.shape, dtype=np.float32)
    for start in range(0, X.shape[0], chunk_size):
        stop = min(X.shape[0], start + chunk_size)
        out[start:stop] = _standardize_block(X[start:stop], mean, scale)
    return out, mean, scale


def _standardization_params(
    X: np.ndarray,
    eps: float = 1e-8,
    fit_mask: np.ndarray | None = None,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit standardisation parameters without materialising a transformed copy."""
    n, d = X.shape
    mask = None if fit_mask is None else np.asarray(fit_mask, dtype=bool)

    if mask is None:
        # ----- no mask: two-pass chunked mean/std -----
        sum_x = np.zeros(d, dtype=np.float64)
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            sum_x += np.asarray(X[start:stop], dtype=np.float64).sum(axis=0)
        mean = sum_x / n

        sum_sq = np.zeros(d, dtype=np.float64)
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            diff = np.asarray(X[start:stop], dtype=np.float64) - mean
            sum_sq += (diff * diff).sum(axis=0)
        var = np.maximum(sum_sq / n, 0.0)
        std = np.sqrt(var)
    else:
        # ----- masked fit: only rows where mask is True -----
        n_fit = int(np.sum(mask))
        if n_fit == 0:
            return np.zeros(d, dtype=np.float64), np.ones(d, dtype=np.float64)
        sum_x = np.zeros(d, dtype=np.float64)
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            m = mask[start:stop]
            if not np.any(m):
                continue
            chunk = np.asarray(X[start:stop][m], dtype=np.float64)
            sum_x += chunk.sum(axis=0)
        mean = sum_x / n_fit

        sum_sq = np.zeros(d, dtype=np.float64)
        for start in range(0, n, chunk_size):
            stop = min(n, start + chunk_size)
            m = mask[start:stop]
            if not np.any(m):
                continue
            diff = np.asarray(X[start:stop][m], dtype=np.float64) - mean
            sum_sq += (diff * diff).sum(axis=0)
        var = np.maximum(sum_sq / n_fit, 0.0)
        std = np.sqrt(var)

    scale = np.where(np.abs(std) < eps, 1.0, std)
    return mean.astype(np.float64), scale.astype(np.float64)


def _standardize_block(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    mean32 = np.asarray(mean, dtype=np.float32)
    scale32 = np.asarray(scale, dtype=np.float32)
    return ((np.asarray(X, dtype=np.float32) - mean32) / scale32).astype(np.float32, copy=False)


def _standardize_rows(X: np.ndarray, rows: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return _standardize_block(X[np.asarray(rows, dtype=np.int64)], mean, scale)


def _weighted_sample_indices(
    candidate_idx: np.ndarray,
    weights: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if candidate_idx.size <= max_points:
        return candidate_idx
    p = _normalize_weights(weights[candidate_idx])
    return np.sort(rng.choice(candidate_idx, size=max_points, replace=False, p=p))


def weighted_knn_density(
    query: np.ndarray,
    reference: np.ndarray,
    reference_weights: np.ndarray,
    k: int,
    batch_size: int,
    *,
    device_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is not None and device_name is not None:
        device = resolve_device(device_name)
        if device is not None and str(device).startswith("cuda"):
            return weighted_knn_density_torch(query, reference, reference_weights, k, batch_size, device)
    return weighted_knn_density_numpy(query, reference, reference_weights, k, batch_size)


def _density_from_radius(local_weight: np.ndarray, radius: np.ndarray, dim: int) -> np.ndarray:
    log_density = np.log(np.maximum(local_weight, 1e-300)) - float(dim) * np.log(np.maximum(radius, 1e-12))
    log_density = np.clip(log_density, -745.0, np.log(np.finfo(np.float64).max))
    return np.exp(log_density)


def weighted_knn_density_numpy(
    query: np.ndarray,
    reference: np.ndarray,
    reference_weights: np.ndarray,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_ref = reference.shape[0]
    if n_ref == 0:
        return np.zeros(query.shape[0], dtype=np.float64), np.full(query.shape[0], np.inf, dtype=np.float64)
    k_eff = max(1, min(int(k), n_ref))
    dim = max(1, int(query.shape[1]))
    density = np.zeros(query.shape[0], dtype=np.float64)
    kth_dist = np.zeros(query.shape[0], dtype=np.float64)
    ref_w = _normalize_weights(reference_weights)

    for start in range(0, query.shape[0], int(batch_size)):
        stop = min(query.shape[0], start + int(batch_size))
        q = np.asarray(query[start:stop], dtype=np.float32)
        r = np.asarray(reference, dtype=np.float32)
        d2 = q @ r.T
        d2 *= -2.0
        d2 += np.sum(q * q, axis=1, keepdims=True)
        d2 += np.sum(r * r, axis=1, keepdims=True).T
        np.maximum(d2, 0.0, out=d2)
        nn = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        nn_d2 = np.take_along_axis(d2, nn, axis=1)
        radius = np.sqrt(np.max(nn_d2, axis=1))
        local_weight = ref_w[nn].sum(axis=1)
        density[start:stop] = _density_from_radius(local_weight, radius, dim)
        kth_dist[start:stop] = radius
    return density, kth_dist


def weighted_knn_density_torch(
    query: np.ndarray,
    reference: np.ndarray,
    reference_weights: np.ndarray,
    k: int,
    batch_size: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    n_ref = reference.shape[0]
    if n_ref == 0:
        return np.zeros(query.shape[0], dtype=np.float64), np.full(query.shape[0], np.inf, dtype=np.float64)
    k_eff = max(1, min(int(k), n_ref))
    dim = max(1, int(query.shape[1]))
    density = np.zeros(query.shape[0], dtype=np.float64)
    kth_dist = np.zeros(query.shape[0], dtype=np.float64)
    ref_w = _normalize_weights(reference_weights)

    ref_t = torch.as_tensor(reference, dtype=torch.float32, device=device)
    ref_norm = torch.sum(ref_t * ref_t, dim=1, keepdim=True).T
    ref_w_t = torch.as_tensor(ref_w, dtype=torch.float64, device=device)
    log_max = float(np.log(np.finfo(np.float64).max))
    with torch.no_grad():
        for start in range(0, query.shape[0], int(batch_size)):
            stop = min(query.shape[0], start + int(batch_size))
            q_t = torch.as_tensor(query[start:stop], dtype=torch.float32, device=device)
            d2 = torch.sum(q_t * q_t, dim=1, keepdim=True) + ref_norm - 2.0 * (q_t @ ref_t.T)
            d2.clamp_min_(0.0)
            nn_d2, nn = torch.topk(d2, k=k_eff, dim=1, largest=False, sorted=False)
            radius = torch.sqrt(torch.amax(nn_d2, dim=1)).clamp_min(1e-12)
            local_weight = ref_w_t[nn].sum(dim=1).clamp_min(1e-300)
            log_density = torch.log(local_weight.double()) - float(dim) * torch.log(radius.double())
            log_density = torch.clamp(log_density, min=-745.0, max=log_max)
            density[start:stop] = torch.exp(log_density).detach().cpu().numpy()
            kth_dist[start:stop] = radius.detach().cpu().numpy().astype(np.float64)
    return density, kth_dist


def select_core_by_density(
    idx: np.ndarray,
    density: np.ndarray,
    weights: np.ndarray,
    fraction: float,
    mode: str,
) -> np.ndarray:
    if idx.size == 0:
        return idx
    order = idx[np.argsort(density[idx])[::-1]]
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return np.asarray([], dtype=np.int64)
    if mode == "weight":
        local_w = _normalize_weights(weights[order])
        keep = np.cumsum(local_w) <= fraction
        if not np.any(keep):
            keep[0] = True
        return np.sort(order[keep])
    n_keep = max(1, int(np.ceil(fraction * idx.size)))
    return np.sort(order[:n_keep])


def _nearest_neighbors(X: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float64)
    n_neighbors = max(1, min(int(n_neighbors), n - 1))
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto").fit(X)
        dist, ind = nn.kneighbors(X, return_distance=True)
        return ind[:, 1:].astype(np.int64), dist[:, 1:].astype(np.float64)
    except Exception:
        d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        ind = np.argpartition(d2, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
        dist = np.sqrt(np.take_along_axis(d2, ind, axis=1))
        return ind.astype(np.int64), dist.astype(np.float64)


def connected_components_from_candidates(
    X: np.ndarray,
    candidate_idx: np.ndarray,
    kdist: np.ndarray,
    *,
    graph_k_neighbors: int,
    max_edge_distance: float | None,
    edge_distance_scale: float,
    edge_distance_quantile: float,
) -> List[np.ndarray]:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    n = candidate_idx.size
    if n == 0:
        return []
    if n == 1:
        return [candidate_idx.copy()]

    local = X[candidate_idx]
    return connected_components_from_candidate_points(
        local,
        candidate_idx,
        kdist,
        graph_k_neighbors=graph_k_neighbors,
        max_edge_distance=max_edge_distance,
        edge_distance_scale=edge_distance_scale,
        edge_distance_quantile=edge_distance_quantile,
    )


def connected_components_from_candidate_points(
    local: np.ndarray,
    candidate_idx: np.ndarray,
    kdist: np.ndarray,
    *,
    graph_k_neighbors: int,
    max_edge_distance: float | None,
    edge_distance_scale: float,
    edge_distance_quantile: float,
) -> List[np.ndarray]:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    n = candidate_idx.size
    if n == 0:
        return []
    if n == 1:
        return [candidate_idx.copy()]

    neigh, dist = _nearest_neighbors(local, graph_k_neighbors)
    if max_edge_distance is None:
        finite_kdist = np.asarray(kdist[candidate_idx], dtype=np.float64)
        finite_kdist = finite_kdist[np.isfinite(finite_kdist) & (finite_kdist > 0.0)]
        if finite_kdist.size:
            cutoff = float(edge_distance_scale) * float(np.quantile(finite_kdist, float(edge_distance_quantile)))
        else:
            cutoff = np.inf
    else:
        cutoff = float(max_edge_distance)

    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j, d in zip(neigh[i], dist[i]):
            if d <= cutoff:
                jj = int(j)
                adjacency[i].append(jj)
                adjacency[jj].append(i)

    seen = np.zeros(n, dtype=bool)
    components: List[np.ndarray] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            item = stack.pop()
            comp.append(item)
            for nxt in adjacency[item]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        components.append(np.sort(candidate_idx[np.asarray(comp, dtype=np.int64)]))
    return components


def select_connected_core_components(
    X: np.ndarray,
    core_idx: np.ndarray,
    center_idx: int,
    density: np.ndarray,
    kdist: np.ndarray,
    connected_cfg: Dict[str, Any],
) -> List[np.ndarray]:
    if core_idx.size == 0:
        return []
    components = connected_components_from_candidates(
        X,
        core_idx,
        kdist,
        graph_k_neighbors=int(connected_cfg.get("graph_k_neighbors", 10)),
        max_edge_distance=connected_cfg.get("max_edge_distance", None),
        edge_distance_scale=float(connected_cfg.get("edge_distance_scale", 2.5)),
        edge_distance_quantile=float(connected_cfg.get("edge_distance_quantile", 0.90)),
    )
    min_size = int(connected_cfg.get("min_pocket_size", 1))
    components = [comp for comp in components if comp.size >= min_size]
    if not components:
        return [np.asarray([center_idx], dtype=np.int64)] if center_idx >= 0 else []

    split = bool(connected_cfg.get("split_disconnected_pockets", False))
    if not split:
        for comp in components:
            if np.any(comp == center_idx):
                return [comp]
        best = max(components, key=lambda comp: float(np.max(density[comp])))
        return [best]

    components.sort(key=lambda comp: float(np.max(density[comp])), reverse=True)
    max_pockets = connected_cfg.get("max_pockets_per_macrostate", None)
    if max_pockets is not None:
        components = components[: int(max_pockets)]
    return components


def select_connected_core_components_from_points(
    candidate_points: np.ndarray,
    core_idx: np.ndarray,
    center_idx: int,
    density: np.ndarray,
    kdist: np.ndarray,
    connected_cfg: Dict[str, Any],
) -> List[np.ndarray]:
    if core_idx.size == 0:
        return []
    components = connected_components_from_candidate_points(
        candidate_points,
        core_idx,
        kdist,
        graph_k_neighbors=int(connected_cfg.get("graph_k_neighbors", 10)),
        max_edge_distance=connected_cfg.get("max_edge_distance", None),
        edge_distance_scale=float(connected_cfg.get("edge_distance_scale", 2.5)),
        edge_distance_quantile=float(connected_cfg.get("edge_distance_quantile", 0.90)),
    )
    min_size = int(connected_cfg.get("min_pocket_size", 1))
    components = [comp for comp in components if comp.size >= min_size]
    if not components:
        return [np.asarray([center_idx], dtype=np.int64)] if center_idx >= 0 else []

    split = bool(connected_cfg.get("split_disconnected_pockets", False))
    if not split:
        for comp in components:
            if np.any(comp == center_idx):
                return [comp]
        best = max(components, key=lambda comp: float(np.max(density[comp])))
        return [best]

    components.sort(key=lambda comp: float(np.max(density[comp])), reverse=True)
    max_pockets = connected_cfg.get("max_pockets_per_macrostate", None)
    if max_pockets is not None:
        components = components[: int(max_pockets)]
    return components


def save_tensorq_dataset(
    path: str,
    save_format: str,
    features: np.ndarray,
    weights: np.ndarray,
    meta_state: np.ndarray,
    dist_to_centroid: np.ndarray,
    thresholds: np.ndarray,
    meta: Dict[str, Any],
    cv_data: np.ndarray,
    traj_id: np.ndarray,
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    features32 = np.asarray(features, dtype=np.float32)
    weights32 = np.asarray(weights, dtype=np.float32)
    meta_state64 = np.asarray(meta_state, dtype=np.int64)
    dist32 = np.asarray(dist_to_centroid, dtype=np.float32)
    thresholds32 = np.asarray(thresholds, dtype=np.float32)
    cv32 = np.asarray(cv_data, dtype=np.float32)
    traj64 = np.asarray(traj_id, dtype=np.int64)
    if str(save_format).lower() == "pt":
        try:
            import torch
        except Exception as exc:
            raise SystemExit("core_labeling.save_format='pt' requires torch. Use 'npz' or install torch.") from exc
        torch.save(
            {
                "features": torch.as_tensor(features32, dtype=torch.float32),
                "weights": torch.as_tensor(weights32, dtype=torch.float32),
                "meta_state": torch.as_tensor(meta_state64, dtype=torch.int64),
                "dist_to_centroid": torch.as_tensor(dist32, dtype=torch.float32),
                "thresholds": torch.as_tensor(thresholds32, dtype=torch.float32),
                "cv": torch.as_tensor(cv32, dtype=torch.float32),
                "traj_id": torch.as_tensor(traj64, dtype=torch.int64),
                "meta": meta,
            },
            path,
        )
        return

    try:
        import yaml
        meta_yaml = yaml.safe_dump(meta, sort_keys=False)
    except Exception:
        meta_yaml = str(meta)
    np.savez_compressed(
        path,
        features=features32,
        weights=weights32,
        meta_state=meta_state64,
        dist_to_centroid=dist32,
        thresholds=thresholds32,
        cv=cv32,
        traj_id=traj64,
        meta_yaml=np.array([meta_yaml], dtype=object),
    )


def save_weights_and_labels_csv(
    path: str,
    table: pd.DataFrame,
    cv_headers: List[str],
    weights: np.ndarray,
    meta_state: np.ndarray,
    dist_to_centroid: np.ndarray,
    density: np.ndarray,
    free_energy: np.ndarray,
) -> None:
    cols = ["global_frame", "traj_id", "frame_in_traj"] + [c for c in cv_headers if c in table.columns]
    cols = [c for c in cols if c in table.columns]
    df = table[cols].copy()
    df["weight"] = weights
    df["meta_state"] = meta_state
    df["is_intermediate"] = (meta_state == -1).astype(np.int8)
    df["dist_to_centroid"] = dist_to_centroid
    df["core_density"] = density
    df["core_free_energy"] = free_energy
    df.to_csv(path, index=False)


def _plot_core_labels(
    out_dir: str,
    cfg: Dict,
    table: pd.DataFrame,
    meta_state: np.ndarray,
    center_frame: np.ndarray,
    m: int,
    lag: int,
) -> List[str]:
    plotting = cfg.get("plotting", {})
    cv_headers = list(cfg["data"]["cvs"])
    pairs = plotting.get("cv_pairs")
    if not pairs:
        pairs = [list(pair) for pair in combinations(cv_headers, 2)]
    max_points = int(plotting.get("max_points", 200000))
    frame_idx = np.arange(len(table))
    if len(frame_idx) > max_points:
        rng = np.random.default_rng(int(cfg["project"].get("seed", 2026)))
        frame_idx = np.sort(rng.choice(frame_idx, size=max_points, replace=False))

    import matplotlib.pyplot as plt

    paths: List[str] = []
    labels = meta_state[frame_idx]
    for pair in pairs:
        if len(pair) != 2 or pair[0] not in table.columns or pair[1] not in table.columns:
            continue
        pair_name = f"{pair[0]}_vs_{pair[1]}".replace("/", "_")
        path = os.path.join(out_dir, f"core_labels_lag_{lag}_m_{m}_{pair_name}.png")
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        intermediate = labels == -1
        if np.any(intermediate):
            ax.scatter(
                table.iloc[frame_idx[intermediate]][pair[0]],
                table.iloc[frame_idx[intermediate]][pair[1]],
                c="lightgray",
                s=4,
                alpha=0.22,
                linewidths=0,
                rasterized=True,
                label="intermediate",
            )
        labeled = labels >= 0
        if np.any(labeled):
            sc = ax.scatter(
                table.iloc[frame_idx[labeled]][pair[0]],
                table.iloc[frame_idx[labeled]][pair[1]],
                c=labels[labeled],
                s=7,
                cmap="tab10",
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )
            fig.colorbar(sc, ax=ax, label="core state")
        center_rows = center_frame[(center_frame >= 0) & (center_frame < len(table))]
        ax.scatter(
            table.iloc[center_rows][pair[0]],
            table.iloc[center_rows][pair[1]],
            c=np.arange(center_rows.shape[0]),
            cmap="tab10",
            s=170,
            marker="X",
            edgecolors="black",
            linewidths=0.8,
        )
        for state, row in enumerate(center_rows):
            ax.annotate(
                str(state),
                (float(table.iloc[row][pair[0]]), float(table.iloc[row][pair[1]])),
                xytext=(5, 5),
                textcoords="offset points",
            )
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
        ax.set_title(f"Core labels: lag={lag}, m={m}")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def build_core_label_datasets(
    cfg: Dict,
    table: pd.DataFrame,
    micro: Dict[str, np.ndarray],
    pcca: Dict[int, Dict[int, Dict[str, np.ndarray]]],
) -> List[str]:
    label_cfg = cfg.get("core_labeling", {})
    if not bool(label_cfg.get("enabled", True)):
        return []

    force = bool(label_cfg.get("force", cfg["project"].get("force", False)))
    out_root = ensure_dir(stage_path(cfg, "06_core_labels"))
    save_format = str(label_cfg.get("save_format", "npz")).lower()
    outputs: List[str] = []

    single_m = label_cfg.get("single_m", label_cfg.get("only_m", None))
    if single_m is not None:
        requested_m = {int(single_m)}
    elif "m_values" in label_cfg:
        requested_m = {int(x) for x in label_cfg["m_values"]}
    else:
        requested_m = None

    if not force:
        missing_any = False
        reusable: List[tuple[int, int, str]] = []
        for lag, by_m in pcca.items():
            for m in by_m:
                if requested_m is not None and m not in requested_m:
                    continue
                dataset_name = f"dataset.{save_format}" if save_format != "pt" else "dataset.pt"
                dataset_path = os.path.join(out_root, f"lag_{lag}", f"m_{m}", dataset_name)
                if exists(dataset_path, force=False):
                    reusable.append((int(lag), int(m), dataset_path))
                else:
                    missing_any = True
        if reusable and not missing_any:
            for lag, m, dataset_path in reusable:
                print(f"[reuse] core labels lag={lag} m={m}: {dataset_path}")
                outputs.append(dataset_path)
            return outputs

    label_feature_headers: List[str]
    label_X_raw, label_feature_headers, label_feature_meta = core_coordinate_matrix(cfg, table)
    output_features, output_feature_headers, output_feature_meta = output_feature_coordinate_matrix(
        cfg,
        table,
        label_X_raw,
        label_feature_headers,
        label_feature_meta,
    )
    projection_cvs = list(cfg["data"]["cvs"])

    weight_column = cfg["data"].get("weight_column", "weight")
    raw_weights = pd.to_numeric(table[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    origin_valid = raw_weights > 0.0
    if not np.any(origin_valid):
        raise SystemExit("core_labeling found no positive-weight frames. Cannot choose FES cores.")

    standardize_chunk_size = int(label_cfg.get("standardize_chunk_size", label_cfg.get("chunk_size", 65536)))
    norm_eps = float(label_cfg.get("standardize_eps", cfg.get("normalization", {}).get("eps", 1e-8)))
    if bool(label_cfg.get("standardize", True)):
        norm_center, norm_scale = _standardization_params(
            label_X_raw,
            eps=norm_eps,
            fit_mask=origin_valid,
            chunk_size=standardize_chunk_size,
        )
    else:
        norm_center = np.zeros(label_X_raw.shape[1], dtype=np.float64)
        norm_scale = np.ones(label_X_raw.shape[1], dtype=np.float64)
    norm_center_work = norm_center.astype(np.float32)
    norm_scale_work = norm_scale.astype(np.float32)

    weights = _normalize_weights(raw_weights)
    traj_codes, _ = pd.factorize(table["traj_id"], sort=False)
    micro_labels = np.asarray(micro["labels"], dtype=np.int64)
    rng = np.random.default_rng(int(cfg["project"].get("seed", 2026)))
    k = int(label_cfg.get("k_neighbors", 25))
    max_ref = int(label_cfg.get("density_reference_points", 5000))
    batch_size = int(label_cfg.get("density_batch_size", 4096))
    device_name = str(label_cfg.get("device", cfg["project"].get("device", "auto")))
    density_backend = str(label_cfg.get("density_backend", "auto")).lower()
    density_device = device_name if density_backend in {"auto", "cuda", "torch"} else None
    if density_device is not None and torch is None:
        print("[warn] torch is not installed; using NumPy CPU kNN density.")
    core_fraction = float(label_cfg.get("core_fraction", 0.05))
    core_by = str(label_cfg.get("core_by", "count")).lower()

    for lag, by_m in pcca.items():
        for m, result in by_m.items():
            if requested_m is not None and m not in requested_m:
                continue
            out_dir = ensure_dir(os.path.join(out_root, f"lag_{lag}", f"m_{m}"))
            dataset_name = f"dataset.{save_format}" if save_format != "pt" else "dataset.pt"
            dataset_path = os.path.join(out_dir, dataset_name)
            if exists(dataset_path, force=force):
                print(f"[reuse] core labels lag={lag} m={m}: {dataset_path}")
                outputs.append(dataset_path)
                continue

            macro_by_micro = np.asarray(result["macro_by_micro"], dtype=np.int64)
            pcca_macro = macro_by_micro[micro_labels]
            meta_state = np.full(len(table), -1, dtype=np.int64)
            density = np.zeros(len(table), dtype=np.float64)
            kdist = np.full(len(table), np.inf, dtype=np.float64)
            selection_mode = str(label_cfg.get("selection_mode", "density_connected")).lower()
            connected_cfg = label_cfg.get("connected_core", {})
            label_units: List[Dict[str, Any]] = []

            for state in range(int(m)):
                state_idx = np.where((pcca_macro == state) & origin_valid)[0]
                if state_idx.size == 0:
                    continue
                ref_idx = _weighted_sample_indices(state_idx, weights, max_ref, rng)
                state_X = _standardize_rows(label_X_raw, state_idx, norm_center_work, norm_scale_work)
                ref_pos = np.searchsorted(state_idx, ref_idx)
                local_density, local_kdist = weighted_knn_density(
                    state_X,
                    state_X[ref_pos],
                    weights[ref_idx],
                    k=k,
                    batch_size=batch_size,
                    device_name=density_device,
                )
                density[state_idx] = local_density
                kdist[state_idx] = local_kdist
                best_local = int(np.argmax(local_density))
                center_idx = int(state_idx[best_local])

                core_idx = select_core_by_density(state_idx, density, weights, core_fraction, core_by)
                if core_idx.size == 0:
                    continue
                if selection_mode in ("top_density", "density"):
                    components = [core_idx]
                elif selection_mode in ("density_connected", "connected", "nearest_core"):
                    core_pos = np.searchsorted(state_idx, core_idx)
                    components = select_connected_core_components_from_points(
                        state_X[core_pos],
                        core_idx,
                        center_idx,
                        density,
                        kdist,
                        connected_cfg,
                    )
                else:
                    raise SystemExit("core_labeling.selection_mode must be top_density or density_connected.")

                for comp in components:
                    if comp.size == 0:
                        continue
                    comp_center_idx = int(comp[np.argmax(density[comp])])
                    label_units.append(
                        {
                            "macrostate": int(state),
                            "indices": comp,
                            "center_idx": comp_center_idx,
                            "density_threshold": float(np.min(density[comp])),
                        }
                    )

            if not label_units:
                raise SystemExit(f"core_labeling found no connected core labels for lag={lag}, m={m}.")

            centers = np.stack([label_X_raw[item["center_idx"]] for item in label_units], axis=0).astype(np.float64)
            center_frame = np.asarray([item["center_idx"] for item in label_units], dtype=np.int64)
            label_to_macrostate = np.asarray([item["macrostate"] for item in label_units], dtype=np.int64)
            density_thresholds = np.asarray([item["density_threshold"] for item in label_units], dtype=np.float64)
            for label_id, item in enumerate(label_units):
                meta_state[np.asarray(item["indices"], dtype=np.int64)] = label_id

            center_scaled = ((centers - norm_center) / norm_scale).astype(np.float32)
            dist_to_centroid = np.full(len(table), np.inf, dtype=np.float32)
            valid_center_mask = np.isfinite(center_scaled).all(axis=1)
            if np.any(valid_center_mask):
                valid_centers_scaled = center_scaled[valid_center_mask]
                for start in range(0, len(table), batch_size):
                    stop = min(len(table), start + batch_size)
                    x_chunk = _standardize_block(label_X_raw[start:stop], norm_center_work, norm_scale_work)
                    diff = x_chunk[:, None, :] - valid_centers_scaled[None, :, :]
                    dist_to_centroid[start:stop] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)).astype(np.float32)
            thresholds = np.full(len(label_units), np.inf, dtype=np.float32)
            for state in range(len(label_units)):
                core_idx = np.where(meta_state == state)[0]
                if core_idx.size > 0:
                    thresholds[state] = float(np.max(dist_to_centroid[core_idx]))

            positive = density > 0.0
            free_energy = np.full(len(table), np.inf, dtype=np.float64)
            if np.any(positive):
                free_energy[positive] = -np.log(np.maximum(density[positive], 1e-300))
                free_energy[positive] -= np.min(free_energy[positive])

            centers_df = pd.DataFrame(centers, columns=label_feature_headers)
            centers_df.insert(0, "core_state", np.arange(len(label_units), dtype=np.int64))
            centers_df.insert(1, "pcca_macrostate", label_to_macrostate)
            centers_df["center_global_frame"] = center_frame
            for cv in projection_cvs:
                if cv in table.columns:
                    centers_df[f"cv_{cv}"] = table.iloc[center_frame][cv].to_numpy(dtype=np.float64)
            center_density = np.full(len(label_units), np.nan, dtype=np.float64)
            center_free_energy = np.full(len(label_units), np.nan, dtype=np.float64)
            valid_centers = center_frame >= 0
            center_density[valid_centers] = density[center_frame[valid_centers]]
            center_free_energy[valid_centers] = free_energy[center_frame[valid_centers]]
            centers_df["center_density"] = center_density
            centers_df["center_free_energy"] = center_free_energy
            centers_df["density_threshold"] = density_thresholds
            centers_df["n_labeled_frames"] = [int(np.sum(meta_state == i)) for i in range(len(label_units))]
            centers_df.to_csv(os.path.join(out_dir, "macrostate_fes_centers.csv"), index=False)

            save_weights_and_labels_csv(
                os.path.join(out_dir, "weights_and_labels.csv"),
                table,
                projection_cvs,
                weights,
                meta_state,
                dist_to_centroid,
                density,
                free_energy,
            )

            meta = {
                "n_frames": int(len(table)),
                "feature_dim": int(output_features.shape[1]),
                "k_selected": int(len(label_units)),
                "pcca_m": int(m),
                "feature_headers": output_feature_headers,
                "cv_headers": projection_cvs,
                "config": cfg,
                "weighting_method": "input_frame_weights",
                "stored_feature_space": output_feature_meta.get("source", "model"),
                "feature_meta": output_feature_meta,
                "label_feature_dim": int(label_X_raw.shape[1]),
                "label_feature_headers": label_feature_headers,
                "label_feature_meta": label_feature_meta,
                "labeling_method": "pcca_weighted_fes_core",
                "label_details": {
                    "lag": int(lag),
                    "m": int(m),
                    "n_core_states": int(len(label_units)),
                    "selection_mode": selection_mode,
                    "connected_core": connected_cfg,
                    "label_to_pcca_macrostate": label_to_macrostate.tolist(),
                    "core_fraction": core_fraction,
                    "core_by": core_by,
                    "k_neighbors": k,
                    "density_reference_points": max_ref,
                    "density_backend": "torch_cuda" if density_device is not None and torch is not None else "numpy",
                    "density_device": density_device,
                    "centers": centers.tolist(),
                    "center_feature_headers": label_feature_headers,
                    "center_feature_space": label_feature_meta.get("source", "model"),
                    "center_global_frame": center_frame.tolist(),
                    "density_thresholds": density_thresholds.tolist(),
                    "threshold_metric": "dist_to_nearest_fes_center_for_core_frames",
                    "intermediate_label": -1,
                    "zero_weight_frames_are_intermediate": True,
                    "n_positive_weight_frames": int(np.sum(origin_valid)),
                },
                "notes": "TensorQ dataset: PCCA defines candidate macrostates; only weighted high-density FES cores are labeled, all other frames are intermediate.",
            }
            cv_data = table[[cv for cv in projection_cvs if cv in table.columns]].to_numpy(dtype=np.float64)
            save_tensorq_dataset(
                dataset_path,
                save_format,
                output_features,
                weights,
                meta_state,
                dist_to_centroid,
                thresholds,
                meta,
                cv_data,
                traj_codes.astype(np.int64),
            )
            save_npz(
                os.path.join(out_dir, "core_labeling.npz"),
                meta_state=meta_state,
                density=density,
                free_energy=free_energy,
                dist_to_centroid=dist_to_centroid,
                thresholds=thresholds,
                density_thresholds=density_thresholds,
                centers=centers,
                center_global_frame=center_frame,
                label_to_macrostate=label_to_macrostate,
                pcca_macro=pcca_macro,
                manifest={
                    "stage": "core_labeling",
                    "lag": int(lag),
                    "m": int(m),
                    "n_core_states": int(len(label_units)),
                    "selection_mode": selection_mode,
                },
            )
            _plot_core_labels(out_dir, cfg, table, meta_state, center_frame, int(m), int(lag))
            counts = pd.Series(meta_state).value_counts().sort_index().to_dict()
            print(f"[ok] core labels lag={lag} m={m}: {dataset_path} counts={counts}")
            outputs.append(dataset_path)
    return outputs

from __future__ import annotations

import numpy as np

from .knn import _torch_knn_distances

def _component_weights(frame_idx, weights):
    if weights is None:
        return np.ones(frame_idx.size, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)[frame_idx]
    return np.clip(sample_weights, 0.0, np.inf)

def _weighted_local_density(z, frame_idx, weights, cfg):
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    sample_weights = _component_weights(frame_idx, weights)
    if frame_idx.size == 0:
        return np.zeros(0, dtype=np.float64), {
            "density_k_neighbors": 0,
            "density_radius_power": np.nan,
            "knn_backend": "none",
            "knn_device": "",
        }
    if frame_idx.size == 1:
        return sample_weights.copy(), {
            "density_k_neighbors": 0,
            "density_radius_power": 0.0,
            "knn_backend": "none",
            "knn_device": "",
        }

    k_default = int(cfg.get("k_neighbors", 20))
    k = int(cfg.get("density_k_neighbors", k_default))
    k = min(max(1, k), frame_idx.size - 1)
    radius_power = float(cfg.get("density_radius_power", min(z.shape[1], 6)))
    radius_power = max(0.0, radius_power)

    n_neighbors = min(frame_idx.size, k + 1)
    points = z[frame_idx]
    torch_knn = _torch_knn_distances(points, points, n_neighbors, cfg, exclude_self=False)
    if torch_knn is None:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(points)
        distances, neighbor_local = nn.kneighbors(points)
        knn_meta = {"knn_backend": "sklearn", "knn_device": ""}
    else:
        distances, neighbor_local, knn_meta = torch_knn
    neighbor_weights = np.sum(sample_weights[neighbor_local], axis=1)
    radius = np.maximum(distances[:, -1], 1e-12)
    density = neighbor_weights / np.power(radius, radius_power)
    return density, {
        "density_k_neighbors": int(k),
        "density_radius_power": radius_power,
        "knn_backend": knn_meta["knn_backend"],
        "knn_device": knn_meta["knn_device"],
    }

def _select_weighted_density_core(z, frame_idx, weights, cfg):
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    sample_weights = _component_weights(frame_idx, weights)
    total_weight = float(np.sum(sample_weights)) if frame_idx.size else 0.0
    if frame_idx.size == 0:
        return frame_idx, frame_idx, {}

    if not bool(cfg.get("density_core_enabled", True)):
        density = np.ones(frame_idx.size, dtype=np.float64)
        return frame_idx, np.zeros(0, dtype=np.int64), {
            "dense_core_enabled": False,
            "dense_core_fraction": 1.0,
            "dense_core_weight_fraction": 1.0,
            "dense_core_density_cutoff": float(np.min(density)),
        }

    fraction = float(cfg.get("density_core_fraction", 0.5))
    fraction = min(max(fraction, 1.0 / max(1, frame_idx.size)), 1.0)
    min_size_default = max(1, int(np.ceil(float(cfg.get("min_new_core_size", 100)) * fraction)))
    min_size = int(cfg.get("density_core_min_size", min_size_default))
    min_size = min(max(1, min_size), frame_idx.size)
    min_weight = float(cfg.get("density_core_min_weight", 0.0))
    max_size = cfg.get("density_core_max_size", None)
    max_size = frame_idx.size if max_size is None else min(max(min_size, int(max_size)), frame_idx.size)

    density, density_meta = _weighted_local_density(z, frame_idx, weights, cfg)
    order = np.lexsort((frame_idx, -density))
    if total_weight > 0.0:
        cumulative_weight = np.cumsum(sample_weights[order])
        target_weight = max(fraction * total_weight, min_weight)
        selected_count = int(np.searchsorted(cumulative_weight, target_weight, side="left") + 1)
        selected_count = max(selected_count, min_size)
    else:
        selected_count = max(int(np.ceil(fraction * frame_idx.size)), min_size)
    selected_count = min(selected_count, max_size, frame_idx.size)

    core_local = order[:selected_count]
    shell_local = order[selected_count:]
    core_idx = np.sort(frame_idx[core_local])
    shell_idx = np.sort(frame_idx[shell_local])
    core_weight = float(np.sum(sample_weights[core_local])) if core_local.size else 0.0
    shell_weight = float(np.sum(sample_weights[shell_local])) if shell_local.size else 0.0
    cutoff = float(np.min(density[core_local])) if core_local.size else np.nan
    meta = {
        "dense_core_enabled": True,
        "dense_core_fraction": fraction,
        "dense_core_frames": int(core_idx.size),
        "dense_core_shell_frames": int(shell_idx.size),
        "dense_core_weight": core_weight,
        "dense_core_shell_weight": shell_weight,
        "dense_core_weight_fraction": float(core_weight / total_weight) if total_weight > 0.0 else np.nan,
        "dense_core_density_cutoff": cutoff,
        "dense_core_density_min": float(np.min(density)) if density.size else np.nan,
        "dense_core_density_median": float(np.median(density)) if density.size else np.nan,
        "dense_core_density_max": float(np.max(density)) if density.size else np.nan,
    }
    meta.update(density_meta)
    return core_idx, shell_idx, meta

def _knn_missing_components(z, candidate_idx, q_values, q_max, q_argmax, entropy_norm, weights, cfg):
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    rows = []
    assignments = []
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if candidate_idx.size == 0:
        return rows, assignments

    k = int(cfg.get("k_neighbors", 20))
    min_size = int(cfg.get("min_new_core_size", 100))
    min_weight = float(cfg.get("min_new_core_weight", 0.0))
    if candidate_idx.size < min_size:
        return rows, assignments

    cluster_idx = candidate_idx
    cluster_shell_frames = 0
    cluster_core_fraction = float(cfg.get("candidate_cluster_core_fraction", cfg.get("density_core_fraction", 0.5)))
    cluster_core_fraction = min(max(cluster_core_fraction, 1.0 / max(1, candidate_idx.size)), 1.0)
    if candidate_idx.size > min_size and cluster_core_fraction < 1.0:
        density, density_meta = _weighted_local_density(z, candidate_idx, weights, cfg)
        sample_weights = _component_weights(candidate_idx, weights)
        order = np.lexsort((candidate_idx, -density))
        if np.sum(sample_weights) > 0.0:
            cumulative = np.cumsum(sample_weights[order])
            target = cluster_core_fraction * float(np.sum(sample_weights))
            keep_count = int(np.searchsorted(cumulative, target, side="left") + 1)
        else:
            keep_count = int(np.ceil(cluster_core_fraction * candidate_idx.size))
        keep_count = min(max(keep_count, min_size), candidate_idx.size)
        cluster_idx = np.sort(candidate_idx[order[:keep_count]])
        cluster_shell_frames = int(candidate_idx.size - cluster_idx.size)
    else:
        density_meta = {
            "density_k_neighbors": 0,
            "density_radius_power": np.nan,
            "knn_backend": "none",
            "knn_device": "",
        }

    if cluster_idx.size == 1:
        labels = np.zeros(1, dtype=np.int64)
        n_components = 1
        knn_meta = {"knn_backend": "none", "knn_device": ""}
    else:
        graph_neighbors = min(max(1, k), cluster_idx.size - 1)
        points = z[cluster_idx]
        torch_knn = _torch_knn_distances(points, points, graph_neighbors, cfg, exclude_self=True)
        if torch_knn is None:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=graph_neighbors + 1)
            nn.fit(points)
            distances, neighbor_idx = nn.kneighbors(points)
            distances = distances[:, 1:]
            neighbor_idx = neighbor_idx[:, 1:]
            knn_meta = {"knn_backend": "sklearn", "knn_device": ""}
        else:
            distances, neighbor_idx, knn_meta = torch_knn

        row_idx = np.repeat(np.arange(cluster_idx.size, dtype=np.int64), neighbor_idx.shape[1])
        col_idx = neighbor_idx.reshape(-1)
        data = np.ones(row_idx.size, dtype=np.int8)
        directed = csr_matrix((data, (row_idx, col_idx)), shape=(cluster_idx.size, cluster_idx.size))
        graph_mode = str(cfg.get("candidate_graph_mode", "mutual_knn")).lower()
        graph = directed.multiply(directed.T) if graph_mode in {"mutual", "mutual_knn", "auto"} else directed.maximum(directed.T)
        if graph.nnz == 0:
            graph = directed.maximum(directed.T)
        n_components, labels = connected_components(graph, directed=False)

    for component in range(n_components):
        local = np.flatnonzero(labels == component)
        frame_idx = cluster_idx[local]
        weighted_population = float(np.sum(weights[frame_idx])) if weights is not None else float(frame_idx.size)
        if frame_idx.size < min_size or weighted_population < min_weight:
            continue

        argmax = q_argmax[frame_idx]
        if argmax.size:
            states, counts = np.unique(argmax, return_counts=True)
            dominant_pos = int(np.argmax(counts))
            dominant_state = int(states[dominant_pos])
            dominant_fraction = float(counts[dominant_pos] / argmax.size)
            mean_q_dominant = float(np.mean(q_values[frame_idx, dominant_state]))
        else:
            dominant_state = -1
            dominant_fraction = np.nan
            mean_q_dominant = np.nan

        rows.append({
            "component": int(component),
            "component_type": "missing_metastate",
            "state_a": -1,
            "state_b": -1,
            "n_frames": int(frame_idx.size),
            "weighted_population": weighted_population,
            "dominant_q_argmax": dominant_state,
            "fraction_dominant_q_argmax": dominant_fraction,
            "mean_q_dominant_argmax": mean_q_dominant,
            "mean_q_max": float(np.mean(q_max[frame_idx])) if frame_idx.size else np.nan,
            "mean_entropy_norm": float(np.mean(entropy_norm[frame_idx])) if frame_idx.size else np.nan,
            "candidate_cluster_core_fraction": cluster_core_fraction,
            "candidate_cluster_core_frames": int(cluster_idx.size),
            "candidate_cluster_shell_frames": int(cluster_shell_frames),
            "candidate_graph_mode": str(cfg.get("candidate_graph_mode", "mutual_knn")),
            "preselected_density_core": True,
            "knn_backend": knn_meta["knn_backend"],
            "knn_device": knn_meta["knn_device"],
        })
        rows[-1].update({
            "candidate_density_k_neighbors": int(density_meta.get("density_k_neighbors", 0)),
            "candidate_density_radius_power": float(density_meta.get("density_radius_power", np.nan)),
        })
        assignments.append(frame_idx)

    return rows, assignments


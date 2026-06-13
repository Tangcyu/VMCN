from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, infer_n_states, load_dataset, select_model_inputs
from ..next_hit.predict import infer_probabilities, load_committor_model
from .kinetic_groups import analyze_basin_kinetic_groups
from .utils import (
    _entropy_confidence,
    _rows_for_frames,
    _save_dataset_like_input,
    _write_csv,
    plot_relabel_cv,
)
from .lag_pair_utils import build_lag_pairs
from .settings import analysis_settings


def _relabel_cfg(config):
    if "relabel" in config and isinstance(config["relabel"], dict):
        return config["relabel"]
    return config.get("radical", {})


def _standardize_features(x):
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std


def _sample_indices(indices, scores, max_count, seed):
    indices = np.asarray(indices, dtype=np.int64)
    max_count = int(max_count)
    if max_count <= 0 or indices.size <= max_count:
        return indices

    scores = np.asarray(scores, dtype=np.float64)
    scores = np.clip(scores, 0.0, np.inf)
    if np.sum(scores) <= 0.0:
        prob = None
    else:
        prob = scores / np.sum(scores)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, size=max_count, replace=False, p=prob))


def _torch_knn_distances(query, reference, k, cfg, *, exclude_self=False):
    backend = str(cfg.get("knn_backend", "auto")).lower()
    if backend in {"sklearn", "cpu", "none"}:
        return None
    if backend not in {"auto", "torch", "cuda"}:
        raise ValueError("relabel.knn_backend must be one of: auto, torch, cuda, sklearn.")

    try:
        import torch
    except Exception:
        if backend == "auto":
            return None
        raise

    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    n_query = int(query.shape[0])
    n_reference = int(reference.shape[0])
    k = int(k)
    if n_query == 0 or n_reference == 0 or k <= 0:
        return (
            np.empty((n_query, 0), dtype=np.float32),
            np.empty((n_query, 0), dtype=np.int64),
            {"knn_backend": "torch", "knn_device": str(cfg.get("knn_device", "cuda:0"))},
        )

    max_pairs = int(cfg.get("torch_knn_auto_max_pairs", 1_000_000_000))
    if backend == "auto" and (n_query * n_reference) > max_pairs:
        return None

    device_str = str(cfg.get("knn_device", cfg.get("device", "cuda:0")))
    device = torch.device(device_str)
    if backend == "auto" and device.type != "cuda":
        return None
    if device.type == "cuda" and not torch.cuda.is_available():
        if backend == "auto":
            return None
        raise RuntimeError(f"Requested relabel.knn_device={device_str!r}, but CUDA is not available.")

    query_batch = max(1, int(cfg.get("torch_knn_query_batch", 4096)))
    reference_batch = max(1, int(cfg.get("torch_knn_reference_batch", 32768)))
    dtype_name = str(cfg.get("torch_knn_dtype", "float32")).lower()
    dtype = torch.float64 if dtype_name in {"float64", "double"} else torch.float32
    k = min(k, n_reference - (1 if exclude_self and n_query == n_reference else 0))
    if k <= 0:
        return (
            np.empty((n_query, 0), dtype=np.float32),
            np.empty((n_query, 0), dtype=np.int64),
            {"knn_backend": "torch", "knn_device": str(device)},
        )

    try:
        reference_t = torch.as_tensor(reference, dtype=dtype, device=device)
        distances_out = np.empty((n_query, k), dtype=np.float32)
        indices_out = np.empty((n_query, k), dtype=np.int64)
        with torch.no_grad():
            for start in range(0, n_query, query_batch):
                stop = min(start + query_batch, n_query)
                query_t = torch.as_tensor(query[start:stop], dtype=dtype, device=device)
                best_dist = torch.full((stop - start, k), float("inf"), dtype=dtype, device=device)
                best_idx = torch.full((stop - start, k), -1, dtype=torch.long, device=device)
                for ref_start in range(0, n_reference, reference_batch):
                    ref_stop = min(ref_start + reference_batch, n_reference)
                    dist = torch.cdist(query_t, reference_t[ref_start:ref_stop])
                    if exclude_self and n_query == n_reference:
                        overlap_start = max(start, ref_start)
                        overlap_stop = min(stop, ref_stop)
                        if overlap_start < overlap_stop:
                            local_q = torch.arange(
                                overlap_start - start,
                                overlap_stop - start,
                                device=device,
                            )
                            local_r = torch.arange(
                                overlap_start - ref_start,
                                overlap_stop - ref_start,
                                device=device,
                            )
                            dist[local_q, local_r] = float("inf")

                    local_k = min(k, dist.shape[1])
                    local_dist, local_idx = torch.topk(dist, k=local_k, dim=1, largest=False)
                    local_idx = local_idx + ref_start
                    merged_dist = torch.cat([best_dist, local_dist], dim=1)
                    merged_idx = torch.cat([best_idx, local_idx], dim=1)
                    best_dist, order = torch.topk(merged_dist, k=k, dim=1, largest=False)
                    best_idx = torch.gather(merged_idx, 1, order)

                distances_out[start:stop] = best_dist.detach().cpu().numpy().astype(np.float32, copy=False)
                indices_out[start:stop] = best_idx.detach().cpu().numpy().astype(np.int64, copy=False)
    except RuntimeError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if backend == "auto":
            return None
        raise

    return distances_out, indices_out, {
        "knn_backend": "torch",
        "knn_device": str(device),
        "torch_knn_query_batch": query_batch,
        "torch_knn_reference_batch": reference_batch,
        "torch_knn_auto_max_pairs": max_pairs,
    }


def _select_graph_features(pack, model_features, config):
    relabel_cfg = _relabel_cfg(config)
    space = str(relabel_cfg.get("graph_space", "auto")).lower()
    if space == "auto":
        model_space = str(config.get("model_input_space", "")).lower()
        space = "features" if model_space in {"features", "feature", "model_features", "model"} else "cv"
    if space in {"cv", "cvs", "colvars"}:
        if pack.cv is None:
            raise RuntimeError("relabel.graph_space='cv' requires saved CV data.")
        return pack.cv.detach().cpu().numpy(), "cv"
    if space in {"features", "feature", "model_features", "model"}:
        return model_features.detach().cpu().numpy(), "model_features"
    raise ValueError("relabel.graph_space must be 'cv' or 'model_features'.")


def _surviving_top2(q_values, surviving_states):
    top1_state = np.full(q_values.shape[0], -1, dtype=np.int64)
    top2_state = np.full(q_values.shape[0], -1, dtype=np.int64)
    top1_q = np.full(q_values.shape[0], np.nan, dtype=np.float64)
    top2_q = np.full(q_values.shape[0], np.nan, dtype=np.float64)

    surviving_states = np.asarray(surviving_states, dtype=np.int64)
    if surviving_states.size < 2:
        return top1_state, top2_state, top1_q, top2_q

    q_surv = q_values[:, surviving_states]
    order = np.argsort(q_surv, axis=1)
    top1_local = order[:, -1]
    top2_local = order[:, -2]
    row = np.arange(q_values.shape[0], dtype=np.int64)
    top1_state = surviving_states[top1_local]
    top2_state = surviving_states[top2_local]
    top1_q = q_surv[row, top1_local]
    top2_q = q_surv[row, top2_local]
    return top1_state, top2_state, top1_q, top2_q


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


def _filter_existing_core_overlap(
    z,
    candidate_mask,
    state_labels,
    q_values,
    q_argmax,
    entropy_norm,
    weights,
    removed_state_ids,
    cfg,
    q_label_cutoff,
    entropy_cutoff,
):
    if not bool(cfg.get("filter_existing_core_overlap", False)):
        return candidate_mask, np.zeros_like(candidate_mask, dtype=bool), [], {
            "enabled": False,
            "reason": "filter_existing_core_overlap is false",
        }

    candidate_idx = np.flatnonzero(candidate_mask)
    overlap_mask = np.zeros_like(candidate_mask, dtype=bool)
    if candidate_idx.size == 0:
        return candidate_mask, overlap_mask, [], {
            "enabled": True,
            "reason": "no missing-metastate candidates",
            "n_overlap_frames": 0,
        }

    n_states = int(q_values.shape[1])
    removed = set(int(state) for state in removed_state_ids)
    core_q_cutoff = float(cfg.get("existing_core_q_cutoff", max(float(q_label_cutoff), 0.8)))
    core_entropy_cutoff = float(cfg.get("existing_core_entropy_cutoff", min(float(entropy_cutoff), 0.35)))
    min_core_size = int(cfg.get("existing_core_min_size", max(10, int(cfg.get("min_new_core_size", 100)) // 5)))
    max_core_frames = int(cfg.get("existing_core_max_frames", 10000))
    n_neighbors = int(cfg.get("existing_core_k_neighbors", 5))
    radius_quantile = float(cfg.get("existing_core_radius_quantile", 0.95))
    radius_scale = float(cfg.get("existing_core_radius_scale", 1.25))
    candidate_min_q = float(cfg.get("existing_overlap_min_candidate_q", 0.0))
    seed = int(cfg.get("random_seed", 0))

    rows = []
    candidate_points = z[candidate_idx]
    for state in range(n_states):
        if state in removed:
            continue
        core_mask = (
            (state_labels == state)
            & (q_argmax == state)
            & (q_values[:, state] >= core_q_cutoff)
            & (entropy_norm <= core_entropy_cutoff)
        )
        core_idx = np.flatnonzero(core_mask)
        if core_idx.size < min_core_size:
            continue

        sample_scores = None if weights is None else np.asarray(weights, dtype=np.float64)[core_idx]
        core_sample = _sample_indices(
            core_idx,
            np.ones(core_idx.size) if sample_scores is None else sample_scores,
            max_core_frames,
            seed + state,
        )
        graph_neighbors = min(max(1, n_neighbors), core_sample.size)
        core_points = z[core_sample]
        torch_core = _torch_knn_distances(core_points, core_points, graph_neighbors, cfg, exclude_self=False)
        torch_candidate = None
        if torch_core is not None:
            torch_candidate = _torch_knn_distances(candidate_points, core_points, 1, cfg, exclude_self=False)

        if torch_core is None or torch_candidate is None:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=graph_neighbors)
            nn.fit(core_points)
            core_dist, _ = nn.kneighbors(core_points)
            candidate_dist, _ = nn.kneighbors(candidate_points, n_neighbors=1)
            knn_meta = {"knn_backend": "sklearn", "knn_device": ""}
        else:
            core_dist, _, knn_meta = torch_core
            candidate_dist, _, _ = torch_candidate

        radius = float(np.quantile(core_dist[:, -1], radius_quantile) * radius_scale)
        min_radius = float(cfg.get("existing_core_min_radius", 0.0))
        radius = max(radius, min_radius)

        state_overlap = candidate_dist[:, 0] <= radius
        if candidate_min_q > 0.0:
            state_overlap &= q_values[candidate_idx, state] >= candidate_min_q
        frame_idx = candidate_idx[state_overlap]
        overlap_mask[frame_idx] = True
        rows.append({
            "state": int(state),
            "core_frames": int(core_idx.size),
            "sampled_core_frames": int(core_sample.size),
            "candidate_overlap_frames": int(frame_idx.size),
            "core_q_cutoff": core_q_cutoff,
            "core_entropy_cutoff": core_entropy_cutoff,
            "core_radius": radius,
            "radius_quantile": radius_quantile,
            "radius_scale": radius_scale,
            "knn_backend": knn_meta["knn_backend"],
            "knn_device": knn_meta["knn_device"],
        })

    filtered = candidate_mask & ~overlap_mask
    return filtered, overlap_mask, rows, {
        "enabled": True,
        "core_q_cutoff": core_q_cutoff,
        "core_entropy_cutoff": core_entropy_cutoff,
        "min_core_size": min_core_size,
        "max_core_frames": max_core_frames,
        "k_neighbors": n_neighbors,
        "radius_quantile": radius_quantile,
        "radius_scale": radius_scale,
        "candidate_min_q": candidate_min_q,
        "n_overlap_frames": int(np.sum(overlap_mask)),
        "n_candidate_frames_before_filter": int(candidate_idx.size),
        "n_candidate_frames_after_filter": int(np.sum(filtered)),
    }


def _pair_mixing_decision_from_matrices(label_i, label_j, active_labels, matrices, count_matrices, lag_list, cfg):
    min_valid = int(cfg.get("merge_min_valid_pairs", cfg.get("min_valid_pairs", 50)))
    transition_cutoff = float(cfg.get("merge_transition_cutoff", cfg.get("mixing_transition_cutoff", 0.05)))
    require_bidirectional = bool(cfg.get("merge_require_bidirectional", True))
    label_to_local = {int(label): idx for idx, label in enumerate(active_labels)}
    local_i = label_to_local[int(label_i)]
    local_j = label_to_local[int(label_j)]

    best = None
    best_score = -np.inf
    for lag in lag_list:
        lag = int(lag)
        probs = matrices[lag]
        counts = count_matrices[lag]
        n_i = int(np.sum(counts[local_i]))
        n_j = int(np.sum(counts[local_j]))
        p_ij = float(probs[local_i, local_j]) if n_i >= min_valid and np.isfinite(probs[local_i, local_j]) else np.nan
        p_ji = float(probs[local_j, local_i]) if n_j >= min_valid and np.isfinite(probs[local_j, local_i]) else np.nan
        finite = [value for value in (p_ij, p_ji) if np.isfinite(value)]
        if not finite:
            continue

        if require_bidirectional:
            if not (np.isfinite(p_ij) and np.isfinite(p_ji)):
                continue
            score = min(p_ij, p_ji)
        else:
            score = max(finite)

        if score > best_score:
            best_score = score
            best = {
                "lag": lag,
                "n_pairs_i": n_i,
                "n_pairs_j": n_j,
                "p_i_to_i": float(probs[local_i, local_i]) if n_i >= min_valid and np.isfinite(probs[local_i, local_i]) else np.nan,
                "p_i_to_j": p_ij,
                "p_j_to_i": p_ji,
                "p_j_to_j": float(probs[local_j, local_j]) if n_j >= min_valid and np.isfinite(probs[local_j, local_j]) else np.nan,
            }

    if best is None or best_score < transition_cutoff:
        return None

    best.update({
        "label_i": int(label_i),
        "label_j": int(label_j),
        "mixing_score": float(best_score),
        "transition_cutoff": transition_cutoff,
        "require_bidirectional": require_bidirectional,
    })
    return best


def _merge_label_into(labels, source, target):
    labels[labels == source] = target


def _positive_lag_list(value, fallback):
    if value is None:
        value = fallback
    if isinstance(value, (int, float)):
        value = [value]
    if value is None:
        return []
    return [int(lag) for lag in value if int(lag) > 0]


def _classify_lagged_entropy_candidates(
    candidate_mask,
    entropy_norm,
    trajectory_index,
    frame_index,
    config,
):
    relabel_cfg = _relabel_cfg(config)
    empty = np.zeros_like(candidate_mask, dtype=bool)
    nan_values = np.full(candidate_mask.shape[0], np.nan, dtype=np.float64)
    if trajectory_index is None or frame_index is None:
        return empty, empty, candidate_mask.copy(), nan_values, nan_values, {
            "enabled": False,
            "reason": "trajectory_index/frame_index unavailable",
        }

    settings = analysis_settings(config)
    lag_list = _positive_lag_list(
        relabel_cfg.get("candidate_lag_list", None),
        settings["lag_list"],
    )
    if not lag_list:
        return empty, empty, candidate_mask.copy(), nan_values, nan_values, {
            "enabled": False,
            "reason": "no candidate lag list configured",
        }

    high_cutoff = float(
        relabel_cfg.get(
            "lagged_entropy_high_cutoff",
            settings["lagged_entropy_cutoff"],
        )
    )
    low_cutoff = float(
        relabel_cfg.get(
            "lagged_entropy_low_cutoff",
            high_cutoff,
        )
    )
    min_valid_lags = int(relabel_cfg.get("candidate_lagged_entropy_min_valid_lags", 1))
    high_fraction_cutoff = float(
        relabel_cfg.get(
            "missing_metastate_lagged_high_fraction",
            settings["persistent_fraction"],
        )
    )
    low_fraction_cutoff = float(
        relabel_cfg.get(
            "transition_lagged_low_fraction",
            settings["persistent_fraction"],
        )
    )
    no_valid_policy = str(relabel_cfg.get("candidate_no_valid_lag_policy", "review")).lower()
    candidate_idx = np.flatnonzero(candidate_mask)
    high_hits = np.zeros(candidate_mask.shape[0], dtype=np.int64)
    low_hits = np.zeros(candidate_mask.shape[0], dtype=np.int64)
    valid_hits = np.zeros(candidate_mask.shape[0], dtype=np.int64)
    entropy_sum = np.zeros(candidate_mask.shape[0], dtype=np.float64)
    lag_pairs = build_lag_pairs(trajectory_index, frame_index, lag_list)

    for lag in lag_list:
        idx_t, idx_tau = lag_pairs[int(lag)]
        start_candidate = candidate_mask[idx_t]
        if not np.any(start_candidate):
            continue
        starts = idx_t[start_candidate]
        lagged_entropy = entropy_norm[idx_tau[start_candidate]]
        finite = np.isfinite(lagged_entropy)
        if not np.any(finite):
            continue
        starts = starts[finite]
        lagged_entropy = lagged_entropy[finite]
        valid_hits[starts] += 1
        high_hits[starts] += (lagged_entropy >= high_cutoff).astype(np.int64)
        low_hits[starts] += (lagged_entropy <= low_cutoff).astype(np.int64)
        entropy_sum[starts] += lagged_entropy

    high_fraction = np.full(candidate_mask.shape[0], np.nan, dtype=np.float64)
    low_fraction = np.full(candidate_mask.shape[0], np.nan, dtype=np.float64)
    mean_lagged_entropy = np.full(candidate_mask.shape[0], np.nan, dtype=np.float64)
    valid = valid_hits > 0
    high_fraction[valid] = high_hits[valid] / valid_hits[valid]
    low_fraction[valid] = low_hits[valid] / valid_hits[valid]
    mean_lagged_entropy[valid] = entropy_sum[valid] / valid_hits[valid]

    enough_lags = valid_hits >= min_valid_lags
    missing_metastate = (
        candidate_mask
        & enough_lags
        & (high_fraction >= high_fraction_cutoff)
    )
    transition_like = (
        candidate_mask
        & ~missing_metastate
        & enough_lags
        & (low_fraction >= low_fraction_cutoff)
    )
    unresolved = candidate_mask & ~(missing_metastate | transition_like)
    if no_valid_policy in {"missing", "promote"}:
        no_valid = candidate_mask & (valid_hits == 0)
        missing_metastate |= no_valid
        unresolved &= ~no_valid
    elif no_valid_policy == "transition":
        no_valid = candidate_mask & (valid_hits == 0)
        transition_like |= no_valid
        unresolved &= ~no_valid

    meta = {
        "enabled": True,
        "lag_list": lag_list,
        "lagged_entropy_high_cutoff": high_cutoff,
        "lagged_entropy_low_cutoff": low_cutoff,
        "min_valid_lags": min_valid_lags,
        "missing_metastate_lagged_high_fraction": high_fraction_cutoff,
        "transition_lagged_low_fraction": low_fraction_cutoff,
        "no_valid_lag_policy": no_valid_policy,
        "n_raw_candidates": int(candidate_idx.size),
        "n_missing_metastate_candidates": int(np.sum(missing_metastate)),
        "n_transition_like_candidates": int(np.sum(transition_like)),
        "n_unresolved_candidates": int(np.sum(unresolved)),
    }
    return missing_metastate, transition_like, unresolved, mean_lagged_entropy, high_fraction, meta


def _new_label_transition_matrices(labels, new_labels, lag_pairs, lag_list, weights):
    labels = np.asarray(labels, dtype=np.int64)
    new_labels = np.sort(np.asarray(new_labels, dtype=np.int64))
    n_labels = int(new_labels.size)
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    matrices = {}
    count_matrices = {}

    for lag in lag_list:
        idx_t, idx_tau = lag_pairs[int(lag)]
        start_labels = labels[idx_t]
        end_labels = labels[idx_tau]
        valid_start = np.isin(start_labels, new_labels)
        valid_end = np.isin(end_labels, new_labels)
        valid = valid_start & valid_end

        matrix = np.zeros((n_labels, n_labels), dtype=np.float64)
        counts = np.zeros((n_labels, n_labels), dtype=np.int64)
        if np.any(valid):
            starts = np.searchsorted(new_labels, start_labels[valid])
            ends = np.searchsorted(new_labels, end_labels[valid])
            pair_weights = (
                np.ones(starts.size, dtype=np.float64)
                if weights_arr is None
                else weights_arr[idx_t[valid]]
            )
            np.add.at(matrix, (starts, ends), pair_weights)
            np.add.at(counts, (starts, ends), 1)

        row_sums = np.sum(matrix, axis=1)
        nonzero = row_sums > 0.0
        probs = np.full((n_labels, n_labels), np.nan, dtype=np.float64)
        probs[nonzero] = matrix[nonzero] / row_sums[nonzero, None]
        matrices[int(lag)] = probs
        count_matrices[int(lag)] = counts

    return matrices, count_matrices


def _iteratively_merge_mixed_new_labels(labels, new_labels, trajectory_index, frame_index, weights, config):
    relabel_cfg = _relabel_cfg(config)
    if not bool(relabel_cfg.get("merge_mixed_new_states", True)):
        return labels, []

    new_labels = sorted(int(label) for label in new_labels)
    if len(new_labels) < 2 or trajectory_index is None or frame_index is None:
        return labels, []

    kinetics_cfg = config.get("kinetics", {})
    lag_list = relabel_cfg.get("merge_lag_list", kinetics_cfg.get("lag_list", [1, 2, 5, 10, 20]))
    if lag_list is None:
        lag_list = kinetics_cfg.get("lag_list", [1, 2, 5, 10, 20])
    if isinstance(lag_list, (int, float)):
        lag_list = [lag_list]
    lag_list = [int(lag) for lag in lag_list if int(lag) > 0]
    if not lag_list:
        return labels, []

    merge_cfg = dict(relabel_cfg)
    merge_cfg["min_valid_pairs"] = int(kinetics_cfg.get("min_valid_pairs", 50))
    lag_pairs = build_lag_pairs(trajectory_index, frame_index, lag_list)

    max_iterations = int(relabel_cfg.get("merge_max_iterations", max(1, len(new_labels) * len(new_labels))))
    merge_rows = []
    labels = labels.copy()

    for iteration in range(max_iterations):
        active = sorted(label for label in new_labels if np.any(labels == label))
        if len(active) < 2:
            break
        matrices, count_matrices = _new_label_transition_matrices(
            labels,
            active,
            lag_pairs,
            lag_list,
            weights,
        )
        best = None
        for idx, label_i in enumerate(active):
            for label_j in active[idx + 1:]:
                decision = _pair_mixing_decision_from_matrices(
                    label_i,
                    label_j,
                    active,
                    matrices,
                    count_matrices,
                    lag_list,
                    merge_cfg,
                )
                if decision is None:
                    continue
                if best is None or decision["mixing_score"] > best["mixing_score"]:
                    best = decision

        if best is None:
            break

        source = max(best["label_i"], best["label_j"])
        target = min(best["label_i"], best["label_j"])
        _merge_label_into(labels, source, target)
        best["iteration"] = int(iteration)
        best["merged_from"] = int(source)
        best["merged_into"] = int(target)
        best["reason"] = "new labels have high indicator time-correlation exchange"
        merge_rows.append(best)

    return labels, merge_rows


def _iteratively_merge_kinetically_duplicate_labels(
    labels,
    original_labels,
    trajectory_index,
    frame_index,
    weights,
    config,
):
    relabel_cfg = _relabel_cfg(config)
    if not bool(relabel_cfg.get("final_kinetic_check_enabled", True)):
        return labels, []
    if not bool(relabel_cfg.get("final_check_merge_enabled", True)):
        return labels, []
    if trajectory_index is None or frame_index is None:
        return labels, []

    kinetics_cfg = config.get("kinetics", {})
    lag_list = _positive_lag_list(
        relabel_cfg.get("final_check_lag_list", relabel_cfg.get("merge_lag_list", None)),
        kinetics_cfg.get("lag_list", [1, 2, 5, 10, 20]),
    )
    if not lag_list:
        return labels, []

    original_labels = set(int(label) for label in original_labels if int(label) >= 0)
    merge_new_existing = bool(relabel_cfg.get("final_check_merge_new_existing", True))
    merge_new_new = bool(relabel_cfg.get("final_check_merge_new_new", True))
    merge_all = bool(relabel_cfg.get("final_check_merge_all_labels", False))
    if not (merge_new_existing or merge_new_new or merge_all):
        return labels, []

    merge_cfg = dict(relabel_cfg)
    merge_cfg["merge_min_valid_pairs"] = int(
        relabel_cfg.get(
            "final_check_merge_min_valid_pairs",
            relabel_cfg.get("merge_min_valid_pairs", kinetics_cfg.get("min_valid_pairs", 50)),
        )
    )
    merge_cfg["merge_transition_cutoff"] = float(
        relabel_cfg.get(
            "final_check_merge_transition_cutoff",
            relabel_cfg.get("merge_transition_cutoff", 0.05),
        )
    )
    merge_cfg["merge_require_bidirectional"] = bool(
        relabel_cfg.get(
            "final_check_merge_require_bidirectional",
            relabel_cfg.get("merge_require_bidirectional", True),
        )
    )

    lag_pairs = build_lag_pairs(trajectory_index, frame_index, lag_list)
    max_iterations = int(relabel_cfg.get("final_check_merge_max_iterations", 100))
    labels = labels.copy()
    merge_rows = []

    for iteration in range(max_iterations):
        active = sorted(int(label) for label in np.unique(labels[labels >= 0]))
        if len(active) < 2:
            break

        matrices, count_matrices = _new_label_transition_matrices(
            labels,
            active,
            lag_pairs,
            lag_list,
            weights,
        )
        best = None
        for idx, label_i in enumerate(active):
            i_original = int(label_i) in original_labels
            for label_j in active[idx + 1:]:
                j_original = int(label_j) in original_labels
                allowed = merge_all
                allowed = allowed or (merge_new_existing and (i_original != j_original))
                allowed = allowed or (merge_new_new and not i_original and not j_original)
                if not allowed:
                    continue

                decision = _pair_mixing_decision_from_matrices(
                    label_i,
                    label_j,
                    active,
                    matrices,
                    count_matrices,
                    lag_list,
                    merge_cfg,
                )
                if decision is None:
                    continue
                if best is None or decision["mixing_score"] > best["mixing_score"]:
                    best = decision

        if best is None:
            break

        label_i = int(best["label_i"])
        label_j = int(best["label_j"])
        i_original = label_i in original_labels
        j_original = label_j in original_labels
        if i_original and not j_original:
            target, source = label_i, label_j
            reason = "new label kinetically duplicates an existing label"
        elif j_original and not i_original:
            target, source = label_j, label_i
            reason = "new label kinetically duplicates an existing label"
        else:
            target, source = min(label_i, label_j), max(label_i, label_j)
            reason = "labels have high lagged exchange and are kinetically duplicate"

        _merge_label_into(labels, source, target)
        best["iteration"] = int(iteration)
        best["merged_from"] = int(source)
        best["merged_into"] = int(target)
        best["merged_from_was_original"] = bool(source in original_labels)
        best["merged_into_was_original"] = bool(target in original_labels)
        best["reason"] = reason
        merge_rows.append(best)

    return labels, merge_rows


def _knn_component_labels(z, frame_idx, k, cfg):
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    if frame_idx.size == 0:
        return 0, np.zeros(0, dtype=np.int64), {"knn_backend": "none", "knn_device": ""}
    if frame_idx.size == 1:
        return 1, np.zeros(1, dtype=np.int64), {"knn_backend": "none", "knn_device": ""}

    graph_neighbors = min(max(1, int(k)), frame_idx.size - 1)
    points = z[frame_idx]
    torch_knn = _torch_knn_distances(points, points, graph_neighbors, cfg, exclude_self=True)
    if torch_knn is None:
        from sklearn.neighbors import kneighbors_graph

        graph = kneighbors_graph(
            points,
            n_neighbors=graph_neighbors,
            mode="connectivity",
            include_self=False,
        )
        knn_meta = {"knn_backend": "sklearn", "knn_device": ""}
    else:
        _, neighbor_idx, knn_meta = torch_knn
        row_idx = np.repeat(np.arange(frame_idx.size, dtype=np.int64), neighbor_idx.shape[1])
        col_idx = neighbor_idx.reshape(-1)
        data = np.ones(row_idx.size, dtype=np.int8)
        graph = csr_matrix((data, (row_idx, col_idx)), shape=(frame_idx.size, frame_idx.size))

    graph = graph.maximum(graph.T)
    n_components, component_labels = connected_components(graph, directed=False)
    return int(n_components), np.asarray(component_labels, dtype=np.int64), knn_meta


def _component_exchange_decision(labels, label, component_by_frame, n_components, lag_pairs, lag_list, weights, cfg):
    min_valid_pairs = int(
        cfg.get(
            "final_check_split_min_valid_pairs",
            cfg.get("merge_min_valid_pairs", cfg.get("min_valid_pairs", 50)),
        )
    )
    min_component_pairs = int(cfg.get("final_check_split_min_component_valid_pairs", 1))
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    best = None
    best_score = np.inf

    for lag in lag_list:
        idx_t, idx_tau = lag_pairs[int(lag)]
        valid = (labels[idx_t] == label) & (labels[idx_tau] == label)
        if not np.any(valid):
            continue
        starts = component_by_frame[idx_t[valid]]
        ends = component_by_frame[idx_tau[valid]]
        valid_components = (starts >= 0) & (ends >= 0)
        if not np.any(valid_components):
            continue
        starts = starts[valid_components]
        ends = ends[valid_components]
        start_frames = idx_t[valid][valid_components]
        pair_weights = (
            np.ones(starts.size, dtype=np.float64)
            if weights_arr is None
            else np.clip(weights_arr[start_frames], 0.0, np.inf)
        )

        weighted_counts = np.zeros((n_components, n_components), dtype=np.float64)
        raw_counts = np.zeros((n_components, n_components), dtype=np.int64)
        np.add.at(weighted_counts, (starts, ends), pair_weights)
        np.add.at(raw_counts, (starts, ends), 1)

        row_weights = np.sum(weighted_counts, axis=1)
        row_raw_counts = np.sum(raw_counts, axis=1)
        active_rows = np.flatnonzero((row_raw_counts >= min_component_pairs) & (row_weights > 0.0))
        n_valid_pairs = int(np.sum(raw_counts))
        if n_valid_pairs < min_valid_pairs or active_rows.size < 2:
            continue

        probs = np.full_like(weighted_counts, np.nan, dtype=np.float64)
        nonzero_rows = row_weights > 0.0
        probs[nonzero_rows] = weighted_counts[nonzero_rows] / row_weights[nonzero_rows, None]
        sub_probs = probs[np.ix_(active_rows, active_rows)]
        offdiag = sub_probs[~np.eye(active_rows.size, dtype=bool)]
        finite_offdiag = offdiag[np.isfinite(offdiag)]
        max_exchange = float(np.max(finite_offdiag)) if finite_offdiag.size else 0.0
        diagonal = np.diag(sub_probs)
        finite_diagonal = diagonal[np.isfinite(diagonal)]
        mean_self = float(np.mean(finite_diagonal)) if finite_diagonal.size else np.nan

        if max_exchange < best_score:
            best_score = max_exchange
            best = {
                "lag": int(lag),
                "n_valid_pairs": n_valid_pairs,
                "n_valid_component_rows": int(active_rows.size),
                "max_exchange_probability": max_exchange,
                "mean_self_probability": mean_self,
                "min_valid_pairs": min_valid_pairs,
                "min_component_valid_pairs": min_component_pairs,
            }

    return best


def _split_labels_by_final_knn_kinetics(
    labels,
    z,
    original_labels,
    trajectory_index,
    frame_index,
    weights,
    config,
    next_label,
):
    relabel_cfg = _relabel_cfg(config)
    split_mask = np.zeros_like(labels, dtype=bool)
    if not bool(relabel_cfg.get("final_kinetic_check_enabled", True)):
        return labels, [], split_mask, next_label
    if not bool(relabel_cfg.get("final_check_split_enabled", True)):
        return labels, [], split_mask, next_label
    if trajectory_index is None or frame_index is None:
        return labels, [], split_mask, next_label

    kinetics_cfg = config.get("kinetics", {})
    lag_list = _positive_lag_list(
        relabel_cfg.get("final_check_lag_list", relabel_cfg.get("merge_lag_list", None)),
        kinetics_cfg.get("lag_list", [1, 2, 5, 10, 20]),
    )
    if not lag_list:
        return labels, [], split_mask, next_label

    original_labels = set(int(label) for label in original_labels if int(label) >= 0)
    scope = str(relabel_cfg.get("final_check_split_labels", "all")).lower()
    k = int(relabel_cfg.get("final_check_split_k_neighbors", relabel_cfg.get("k_neighbors", 20)))
    min_size = int(relabel_cfg.get("final_check_split_min_component_size", relabel_cfg.get("min_new_core_size", 100)))
    min_weight = float(relabel_cfg.get("final_check_split_min_component_weight", 0.0))
    min_components = int(relabel_cfg.get("final_check_split_min_components", 2))
    max_label_frames = int(relabel_cfg.get("final_check_split_max_label_frames", 200000))
    require_low_exchange = bool(relabel_cfg.get("final_check_split_require_low_exchange", True))
    exchange_cutoff = float(relabel_cfg.get("final_check_split_max_exchange_probability", 0.05))
    report_unsplit = bool(relabel_cfg.get("final_check_report_unsplit_labels", False))
    lag_pairs = build_lag_pairs(trajectory_index, frame_index, lag_list)

    labels = labels.copy()
    rows = []
    active_labels = sorted(int(label) for label in np.unique(labels[labels >= 0]))
    for label in active_labels:
        label_is_original = label in original_labels
        if scope in {"existing", "old", "original"} and not label_is_original:
            continue
        if scope in {"new", "added"} and label_is_original:
            continue

        frame_idx = np.flatnonzero(labels == label)
        if frame_idx.size < max(1, min_components) * max(1, min_size):
            continue
        if max_label_frames > 0 and frame_idx.size > max_label_frames:
            rows.append({
                "label": int(label),
                "label_was_original": bool(label_is_original),
                "action": "skipped_label_too_large",
                "n_frames": int(frame_idx.size),
                "max_label_frames": int(max_label_frames),
            })
            continue

        n_components, component_labels, knn_meta = _knn_component_labels(z, frame_idx, k, relabel_cfg)
        if n_components < min_components:
            continue

        component_stats = []
        for component in range(n_components):
            local = np.flatnonzero(component_labels == component)
            component_frame_idx = frame_idx[local]
            weighted_population = (
                float(np.sum(weights[component_frame_idx]))
                if weights is not None
                else float(component_frame_idx.size)
            )
            if component_frame_idx.size < min_size or weighted_population < min_weight:
                continue
            component_stats.append({
                "component": int(component),
                "n_frames": int(component_frame_idx.size),
                "weighted_population": weighted_population,
            })

        if len(component_stats) < min_components:
            continue

        component_stats = sorted(component_stats, key=lambda row: int(row["component"]))
        component_by_frame = np.full(labels.shape[0], -1, dtype=np.int64)
        for local_component, row in enumerate(component_stats):
            component = int(row["component"])
            component_by_frame[frame_idx[component_labels == component]] = local_component

        decision = _component_exchange_decision(
            labels,
            label,
            component_by_frame,
            len(component_stats),
            lag_pairs,
            lag_list,
            weights,
            relabel_cfg,
        )
        should_split = True
        if require_low_exchange:
            should_split = (
                decision is not None
                and decision["max_exchange_probability"] <= exchange_cutoff
            )

        if not should_split:
            if report_unsplit:
                rows.append({
                    "label": int(label),
                    "label_was_original": bool(label_is_original),
                    "action": "kept_label_mixed_by_lagged_exchange",
                    "n_frames": int(frame_idx.size),
                    "n_components": int(n_components),
                    "n_eligible_components": int(len(component_stats)),
                    "lag": -1 if decision is None else int(decision["lag"]),
                    "n_valid_pairs": 0 if decision is None else int(decision["n_valid_pairs"]),
                    "max_exchange_probability": np.nan if decision is None else float(decision["max_exchange_probability"]),
                    "split_exchange_cutoff": exchange_cutoff,
                    "knn_backend": knn_meta["knn_backend"],
                    "knn_device": knn_meta["knn_device"],
                })
            continue

        keep_component = max(
            component_stats,
            key=lambda row: (float(row["weighted_population"]), int(row["n_frames"]), -int(row["component"])),
        )["component"]
        for row in component_stats:
            component = int(row["component"])
            component_frame_idx = frame_idx[component_labels == component]
            if component == keep_component:
                assigned_label = label
                action = "keep_largest_component_as_label"
            else:
                assigned_label = int(next_label)
                next_label += 1
                labels[component_frame_idx] = assigned_label
                split_mask[component_frame_idx] = True
                action = "split_component_to_new_label"

            rows.append({
                "label": int(label),
                "label_was_original": bool(label_is_original),
                "component": component,
                "assigned_label": int(assigned_label),
                "action": action,
                "n_frames": int(component_frame_idx.size),
                "weighted_population": float(row["weighted_population"]),
                "n_components": int(n_components),
                "n_eligible_components": int(len(component_stats)),
                "lag": -1 if decision is None else int(decision["lag"]),
                "n_valid_pairs": 0 if decision is None else int(decision["n_valid_pairs"]),
                "n_valid_component_rows": 0 if decision is None else int(decision["n_valid_component_rows"]),
                "max_exchange_probability": np.nan if decision is None else float(decision["max_exchange_probability"]),
                "mean_self_probability": np.nan if decision is None else float(decision["mean_self_probability"]),
                "split_exchange_cutoff": exchange_cutoff,
                "require_low_exchange": bool(require_low_exchange),
                "knn_backend": knn_meta["knn_backend"],
                "knn_device": knn_meta["knn_device"],
            })

    return labels, rows, split_mask, next_label


def _reshape_existing_basins_from_kinetic_groups(
    new_state,
    state_labels,
    q_values,
    graph_features,
    trajectory_index,
    frame_index,
    weights,
    config,
    next_label,
):
    relabel_cfg = _relabel_cfg(config)
    if not bool(relabel_cfg.get("reshape_existing_basins", True)):
        return new_state, [], [], np.full_like(new_state, -1, dtype=np.int64), np.zeros_like(new_state, dtype=bool), np.zeros_like(new_state, dtype=bool), next_label
    if trajectory_index is None or frame_index is None:
        return new_state, [], [], np.full_like(new_state, -1, dtype=np.int64), np.zeros_like(new_state, dtype=bool), np.zeros_like(new_state, dtype=bool), next_label

    settings = analysis_settings(config)
    kinetics_cfg = config.get("kinetics", {})
    kg_cfg = config.get("basin_kinetic_groups", {})
    lag_list = kg_cfg.get("lag_list", settings["lag_list"])
    if lag_list is None:
        lag_list = kinetics_cfg.get("lag_list", [1, 2, 5, 10, 20])
    if isinstance(lag_list, (int, float)):
        lag_list = [lag_list]
    lag_list = [int(lag) for lag in lag_list if int(lag) > 0]
    if not lag_list:
        return new_state, [], [], np.full_like(new_state, -1, dtype=np.int64), np.zeros_like(new_state, dtype=bool), np.zeros_like(new_state, dtype=bool), next_label

    lag_pairs = build_lag_pairs(trajectory_index, frame_index, lag_list)
    state_rows, group_rows, group_labels = analyze_basin_kinetic_groups(
        state_labels,
        q_values,
        lag_pairs,
        config,
        weights=weights,
        n_states=int(q_values.shape[1]),
        features=graph_features,
    )

    trim_shell = bool(relabel_cfg.get("trim_existing_to_high_confidence_core", True))
    split_groups = bool(relabel_cfg.get("split_existing_kinetic_groups", True))
    min_groups_to_split = int(relabel_cfg.get("min_kinetic_groups_to_split", 2))
    min_split_size = int(relabel_cfg.get("min_new_core_size", 100))
    min_split_weight = float(relabel_cfg.get("min_new_core_weight", 0.0))
    min_split_weight_fraction = float(
        kg_cfg.get(
            "min_split_core_weight_fraction",
            relabel_cfg.get("min_split_core_weight_fraction", 0.1),
        )
    )
    reshape_core_mask = np.zeros_like(new_state, dtype=bool)
    reshape_shell_mask = np.zeros_like(new_state, dtype=bool)
    assignment_rows = []

    groups_by_state = {}
    for row in group_rows:
        groups_by_state.setdefault(int(row["state"]), []).append(row)

    for state, rows in groups_by_state.items():
        rows = sorted(rows, key=lambda row: (int(row["rank_within_state"]), int(row["kinetic_group"])))
        if not rows:
            continue
        state_mask = state_labels == state
        state_current_mask = new_state == state
        group_core_mask = np.isin(group_labels, [int(row["kinetic_group"]) for row in rows])

        if trim_shell:
            shell = state_current_mask & ~group_core_mask
            if np.any(shell):
                new_state[shell] = -1
                reshape_shell_mask[shell] = True

        split_recommended = any(bool(row.get("split_recommended", False)) for row in rows)
        do_split = split_groups and split_recommended and len(rows) >= min_groups_to_split
        split_group_total_weight = float(
            np.sum([float(row.get("weighted_population", 0.0)) for row in rows])
        )
        for row_idx, row in enumerate(rows):
            group_id = int(row["kinetic_group"])
            group_mask = (group_labels == group_id) & state_mask
            if not np.any(group_mask):
                continue
            group_frames = int(np.sum(group_mask))
            group_weight = float(row["weighted_population"])
            weighted_fraction_of_state = float(row.get("weighted_fraction_of_state", np.nan))
            weighted_fraction_of_split = (
                float(group_weight / split_group_total_weight)
                if split_group_total_weight > 0.0
                else np.nan
            )
            split_to_new_label = do_split and row_idx > 0
            small_split_group = split_to_new_label and (
                group_frames < min_split_size
                or group_weight < min_split_weight
                or (
                    np.isfinite(weighted_fraction_of_split)
                    and weighted_fraction_of_split < min_split_weight_fraction
                )
            )

            if small_split_group:
                target_label = -1
                action = "drop_small_split_group"
            elif split_to_new_label:
                target_label = int(next_label)
                next_label += 1
                action = "split_to_new_label"
            else:
                target_label = int(state)
                action = "keep_as_existing_label"

            new_state[group_mask] = target_label
            if small_split_group:
                reshape_shell_mask[group_mask] = True
            else:
                reshape_core_mask[group_mask] = True
            assignment_rows.append({
                "old_state": int(state),
                "kinetic_group": group_id,
                "rank_within_state": int(row["rank_within_state"]),
                "assigned_state": int(target_label),
                "n_frames": group_frames,
                "weighted_population": group_weight,
                "weighted_fraction_of_state": weighted_fraction_of_state,
                "weighted_fraction_of_split": weighted_fraction_of_split,
                "split_group_total_weight": split_group_total_weight,
                "fraction_of_state": float(row["fraction_of_state"]),
                "mean_q_own": float(row["mean_q_own"]),
                "mean_q_max": float(row["mean_q_max"]),
                "mean_entropy_norm": float(row["mean_entropy_norm"]),
                "min_new_core_size": int(min_split_size),
                "min_new_core_weight": float(min_split_weight),
                "min_split_core_weight_fraction": float(min_split_weight_fraction),
                "action": action,
            })

    return new_state, state_rows, assignment_rows, group_labels, reshape_core_mask, reshape_shell_mask, next_label


def propose_relabeling(
    q_values,
    state_labels,
    graph_features,
    weights,
    config,
    trajectory_index=None,
    frame_index=None,
):
    settings = analysis_settings(config)
    relabel_cfg = _relabel_cfg(config)
    if "knn_device" not in relabel_cfg and "device" in config:
        relabel_cfg["knn_device"] = str(config["device"])

    q_label_cutoff = float(settings["q_cutoff"])
    entropy_cutoff = float(settings["entropy_cutoff"])
    remove_cutoff = float(relabel_cfg.get("remove_problem_fraction_cutoff", 0.9))
    remove_min_stable_fraction = float(relabel_cfg.get("remove_min_stable_fraction", settings["persistent_fraction"]))
    candidate_qmax_cutoff = float(relabel_cfg.get("candidate_current_qmax_cutoff", q_label_cutoff))
    candidate_require_low_qmax = bool(relabel_cfg.get("candidate_require_current_low_qmax", False))
    top2_min_probability = float(relabel_cfg.get("top2_min_probability", 0.2))
    top2_margin_cutoff = float(relabel_cfg.get("top2_margin_cutoff", 0.2))

    state_labels = np.asarray(state_labels, dtype=np.int64)
    new_state = state_labels.copy()
    n_states = int(q_values.shape[1])
    q_max, q_argmax, _, entropy_norm = _entropy_confidence(q_values)
    weights = None if weights is None else np.asarray(weights, dtype=np.float64)

    label_consistency = np.full(state_labels.shape[0], np.nan, dtype=np.float64)
    valid = (state_labels >= 0) & (state_labels < n_states)
    valid_idx = np.flatnonzero(valid)
    label_consistency[valid] = q_values[valid_idx, state_labels[valid]]

    removed_states = []
    removed_state_ids = []
    removed_mask = np.zeros(state_labels.shape[0], dtype=bool)
    for state in range(n_states):
        mask = state_labels == state
        n_state = int(np.sum(mask))
        if n_state == 0:
            continue
        problem = mask & (
            (label_consistency < q_label_cutoff)
            | (entropy_norm >= entropy_cutoff)
        )
        problem_fraction = float(np.sum(problem) / max(1, n_state))
        stable = mask & (
            (label_consistency >= q_label_cutoff)
            & (entropy_norm < entropy_cutoff)
        )
        stable_fraction = float(np.sum(stable) / max(1, n_state))
        should_remove = problem_fraction >= remove_cutoff or stable_fraction < remove_min_stable_fraction
        if should_remove:
            new_state[mask] = -1
            removed_mask |= mask
            removed_state_ids.append(state)
            if problem_fraction >= remove_cutoff:
                reason = "state removed because most frames are low-consistency or high-entropy"
            else:
                reason = "state removed because too little of the label is stable high-confidence core"
            removed_states.append({
                "removed_state": int(state),
                "n_frames": n_state,
                "n_problem_frames": int(np.sum(problem)),
                "n_stable_frames": int(np.sum(stable)),
                "problem_fraction": problem_fraction,
                "stable_fraction": stable_fraction,
                "min_stable_fraction": remove_min_stable_fraction,
                "mean_q_own": float(np.mean(q_values[mask, state])),
                "mean_entropy_norm": float(np.mean(entropy_norm[mask])),
                "reason": reason,
            })

    current_entropy_candidate_mask = entropy_norm >= entropy_cutoff
    if candidate_require_low_qmax:
        current_entropy_candidate_mask &= q_max <= candidate_qmax_cutoff

    ambiguous_to_unlabeled_mask = current_entropy_candidate_mask & (new_state >= 0)
    new_state[ambiguous_to_unlabeled_mask] = -1

    current_entropy_unlabeled = current_entropy_candidate_mask & (new_state == -1)
    (
        missing_metastate_h_tau_mask,
        transition_like_candidate_mask,
        unresolved_lagged_candidate_mask,
        mean_lagged_entropy,
        lagged_high_entropy_fraction,
        lagged_entropy_classification,
    ) = _classify_lagged_entropy_candidates(
        current_entropy_unlabeled,
        entropy_norm,
        None if trajectory_index is None else np.asarray(trajectory_index, dtype=np.int64),
        None if frame_index is None else np.asarray(frame_index, dtype=np.int64),
        config,
    )
    missing_metastate_raw_mask = missing_metastate_h_tau_mask
    z = _standardize_features(graph_features)
    raw_candidate_idx = np.flatnonzero(missing_metastate_raw_mask)
    seed = int(relabel_cfg.get("random_seed", 0))
    max_graph = int(relabel_cfg.get("max_graph_frames", 20000))
    raw_lagged_score = np.nan_to_num(
        lagged_high_entropy_fraction[raw_candidate_idx],
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    raw_candidate_scores = (
        entropy_norm[raw_candidate_idx]
        * np.clip(raw_lagged_score, 0.0, 1.0)
        * (1.0 - np.clip(q_max[raw_candidate_idx], 0.0, 1.0))
        * (weights[raw_candidate_idx] if weights is not None else 1.0)
    )
    sampled_candidate_idx = _sample_indices(raw_candidate_idx, raw_candidate_scores, max_graph, seed)
    sampled_candidate_mask = np.zeros_like(missing_metastate_raw_mask, dtype=bool)
    sampled_candidate_mask[sampled_candidate_idx] = True
    missing_metastate_seed_mask, existing_core_overlap_mask, existing_core_overlap_rows, existing_core_overlap_filter = (
        _filter_existing_core_overlap(
            z,
            sampled_candidate_mask,
            state_labels,
            q_values,
            q_argmax,
            entropy_norm,
            weights,
            removed_state_ids,
            relabel_cfg,
            q_label_cutoff,
            entropy_cutoff,
        )
    )

    candidate_mask = missing_metastate_seed_mask
    candidate_persistence = lagged_entropy_classification
    candidate_idx = np.flatnonzero(candidate_mask)

    surviving_states = np.array(
        [state for state in range(n_states) if state not in set(removed_state_ids)],
        dtype=np.int64,
    )
    top1_state, top2_state, top1_q, top2_q = _surviving_top2(q_values, surviving_states)
    two_state_ambiguous_review_mask = (
        (new_state == -1)
        & current_entropy_candidate_mask
        & (top1_state >= 0)
        & (top2_state >= 0)
        & (top2_q >= top2_min_probability)
        & ((top1_q - top2_q) <= top2_margin_cutoff)
    )

    graph_idx = candidate_idx
    persistent_candidate_review_mask = np.zeros_like(candidate_mask, dtype=bool)
    if not bool(relabel_cfg.get("promote_missing_metastate_candidates", True)):
        persistent_candidate_review_mask = candidate_mask.copy()
        graph_idx = np.zeros(0, dtype=np.int64)

    component_rows = []
    component_frames = []
    if graph_idx.size:
        component_rows, component_frames = _knn_missing_components(
            z,
            graph_idx,
            q_values,
            q_max,
            q_argmax,
            entropy_norm,
            weights,
            relabel_cfg,
        )

    next_label = int(np.max(state_labels[state_labels >= 0]) + 1) if np.any(state_labels >= 0) else 0
    new_core_mask = np.zeros(state_labels.shape[0], dtype=bool)
    density_shell_mask = np.zeros(state_labels.shape[0], dtype=bool)
    provisional_new_labels = []
    for row, frame_idx in zip(component_rows, component_frames):
        row["new_state"] = int(next_label)
        row["provisional_new_state"] = int(next_label)
        row["component_candidate_frames"] = int(frame_idx.size)
        row["component_candidate_weight"] = float(np.sum(weights[frame_idx])) if weights is not None else float(frame_idx.size)
        new_state[frame_idx] = next_label
        provisional_new_labels.append(next_label)
        next_label += 1

    new_state, merged_new_state_rows = _iteratively_merge_mixed_new_labels(
        new_state,
        provisional_new_labels,
        None if trajectory_index is None else np.asarray(trajectory_index, dtype=np.int64),
        None if frame_index is None else np.asarray(frame_index, dtype=np.int64),
        weights,
        config,
    )

    component_final_labels = {
        int(row["provisional_new_state"]): int(new_state[component_frames[idx][0]])
        for idx, row in enumerate(component_rows)
        if len(component_frames[idx]) > 0
    }
    preselected_core_by_label = {}
    for idx, row in enumerate(component_rows):
        if len(component_frames[idx]) == 0:
            continue
        provisional = int(row["provisional_new_state"])
        final_label = component_final_labels.get(provisional, provisional)
        preselected_core_by_label[int(final_label)] = (
            preselected_core_by_label.get(int(final_label), False)
            or bool(row.get("preselected_density_core", False))
        )

    final_density = {}
    active_final_labels = sorted({label for label in component_final_labels.values() if label >= 0})
    for final_label in active_final_labels:
        final_idx = np.flatnonzero(new_state == final_label)
        if preselected_core_by_label.get(int(final_label), False):
            core_idx = final_idx
            shell_idx = np.zeros(0, dtype=np.int64)
            density_meta = {
                "dense_core_enabled": False,
                "dense_core_fraction": 1.0,
                "dense_core_frames": int(core_idx.size),
                "dense_core_shell_frames": 0,
                "dense_core_weight": float(np.sum(weights[core_idx])) if weights is not None else float(core_idx.size),
                "dense_core_shell_weight": 0.0,
                "dense_core_weight_fraction": 1.0,
                "dense_core_density_cutoff": np.nan,
                "dense_core_density_min": np.nan,
                "dense_core_density_median": np.nan,
                "dense_core_density_max": np.nan,
                "density_k_neighbors": 0,
                "density_radius_power": np.nan,
                "density_reason": "candidate components were already built from the dense core",
            }
        else:
            core_idx, shell_idx, density_meta = _select_weighted_density_core(
                z, final_idx, weights, relabel_cfg
            )
        if shell_idx.size:
            new_state[shell_idx] = -1
            density_shell_mask[shell_idx] = True
        if core_idx.size:
            new_core_mask[core_idx] = True
            new_state[core_idx] = final_label
        final_density[int(final_label)] = {
            **density_meta,
            "final_basin_candidate_frames": int(final_idx.size),
            "final_basin_candidate_weight": float(np.sum(weights[final_idx])) if weights is not None else float(final_idx.size),
            "mean_q_max": float(np.mean(q_max[core_idx])) if core_idx.size else np.nan,
            "mean_entropy_norm": float(np.mean(entropy_norm[core_idx])) if core_idx.size else np.nan,
        }

    for idx, row in enumerate(component_rows):
        provisional = int(row["provisional_new_state"])
        final_label = component_final_labels.get(provisional, provisional)
        row["new_state"] = int(final_label)
        component_frame_idx = component_frames[idx]
        row["component_dense_core_frames"] = int(np.sum(new_state[component_frame_idx] == final_label))
        row["component_density_shell_frames"] = int(np.sum(density_shell_mask[component_frame_idx]))
        row.update(final_density.get(int(final_label), {
            "dense_core_enabled": bool(relabel_cfg.get("density_core_enabled", True)),
            "dense_core_fraction": np.nan,
            "dense_core_frames": 0,
            "dense_core_shell_frames": 0,
            "dense_core_weight": 0.0,
            "dense_core_shell_weight": 0.0,
            "dense_core_weight_fraction": np.nan,
            "dense_core_density_cutoff": np.nan,
            "dense_core_density_min": np.nan,
            "dense_core_density_median": np.nan,
            "dense_core_density_max": np.nan,
            "density_k_neighbors": 0,
            "density_radius_power": np.nan,
            "final_basin_candidate_frames": 0,
            "final_basin_candidate_weight": 0.0,
            "mean_q_max": np.nan,
            "mean_entropy_norm": np.nan,
        }))

    next_label = int(np.max(new_state[new_state >= 0]) + 1) if np.any(new_state >= 0) else int(next_label)
    (
        new_state,
        basin_kinetic_state_stats,
        reshaped_basin_groups,
        basin_kinetic_group_labels,
        reshaped_basin_core_mask,
        reshaped_basin_shell_mask,
        next_label,
    ) = _reshape_existing_basins_from_kinetic_groups(
        new_state,
        state_labels,
        q_values,
        graph_features,
        None if trajectory_index is None else np.asarray(trajectory_index, dtype=np.int64),
        None if frame_index is None else np.asarray(frame_index, dtype=np.int64),
        weights,
        config,
        next_label,
    )

    original_label_set = set(int(label) for label in np.unique(state_labels[state_labels >= 0]))
    (
        new_state,
        final_kinetic_merge_rows,
    ) = _iteratively_merge_kinetically_duplicate_labels(
        new_state,
        original_label_set,
        None if trajectory_index is None else np.asarray(trajectory_index, dtype=np.int64),
        None if frame_index is None else np.asarray(frame_index, dtype=np.int64),
        weights,
        config,
    )
    for row in final_kinetic_merge_rows:
        row["phase"] = "before_final_split"

    next_label = int(np.max(new_state[new_state >= 0]) + 1) if np.any(new_state >= 0) else int(next_label)
    (
        new_state,
        final_kinetic_split_rows,
        final_kinetic_split_mask,
        next_label,
    ) = _split_labels_by_final_knn_kinetics(
        new_state,
        z,
        original_label_set,
        None if trajectory_index is None else np.asarray(trajectory_index, dtype=np.int64),
        None if frame_index is None else np.asarray(frame_index, dtype=np.int64),
        weights,
        config,
        next_label,
    )

    (
        new_state,
        final_post_split_merge_rows,
    ) = _iteratively_merge_kinetically_duplicate_labels(
        new_state,
        original_label_set,
        None if trajectory_index is None else np.asarray(trajectory_index, dtype=np.int64),
        None if frame_index is None else np.asarray(frame_index, dtype=np.int64),
        weights,
        config,
    )
    for row in final_post_split_merge_rows:
        row["phase"] = "after_final_split"
    final_kinetic_merge_rows.extend(final_post_split_merge_rows)

    for idx, row in enumerate(component_rows):
        component_frame_idx = component_frames[idx]
        if len(component_frame_idx) == 0:
            row["final_checked_state"] = -1
            row["component_final_checked_frames"] = 0
            row["component_final_unlabeled_frames"] = 0
            continue
        assigned = new_state[component_frame_idx]
        nonnegative = assigned[assigned >= 0]
        if nonnegative.size:
            labels, counts = np.unique(nonnegative, return_counts=True)
            dominant = int(labels[int(np.argmax(counts))])
            dominant_frames = int(np.max(counts))
        else:
            dominant = -1
            dominant_frames = 0
        row["new_state"] = dominant
        row["final_checked_state"] = dominant
        row["component_final_checked_frames"] = dominant_frames
        row["component_final_unlabeled_frames"] = int(np.sum(assigned < 0))

    changed_mask = new_state != state_labels
    unassigned_pair_ambiguous = (
        (two_state_ambiguous_review_mask & ~new_core_mask)
        | transition_like_candidate_mask
        | existing_core_overlap_mask
        | unresolved_lagged_candidate_mask
        | persistent_candidate_review_mask
        | (missing_metastate_raw_mask & ~new_core_mask)
    )

    return {
        "proposed_labels": new_state,
        "changed_mask": changed_mask,
        "removed_mask": removed_mask,
        "current_entropy_candidate_mask": current_entropy_candidate_mask,
        "ambiguous_to_unlabeled_mask": ambiguous_to_unlabeled_mask,
        "new_core_mask": new_core_mask,
        "density_shell_mask": density_shell_mask,
        "reshaped_basin_core_mask": reshaped_basin_core_mask,
        "reshaped_basin_shell_mask": reshaped_basin_shell_mask,
        "final_kinetic_split_mask": final_kinetic_split_mask,
        "review_mixed_mask": np.zeros(state_labels.shape[0], dtype=bool),
        "review_missing_mask": unassigned_pair_ambiguous,
        "transition_like_candidate_mask": transition_like_candidate_mask,
        "missing_metastate_h_tau_mask": missing_metastate_h_tau_mask,
        "unresolved_lagged_candidate_mask": unresolved_lagged_candidate_mask,
        "missing_metastate_raw_candidate_mask": missing_metastate_raw_mask,
        "missing_metastate_seed_candidate_mask": missing_metastate_seed_mask,
        "missing_metastate_candidate_mask": candidate_mask,
        "existing_core_overlap_mask": existing_core_overlap_mask,
        "two_state_ambiguous_review_mask": two_state_ambiguous_review_mask,
        "persistent_candidate_review_mask": persistent_candidate_review_mask,
        "removed_states": removed_states,
        "removed_state_ids": removed_state_ids,
        "new_core_components": component_rows,
        "merged_new_states": merged_new_state_rows,
        "knn_backend_config": {
            "knn_backend": str(relabel_cfg.get("knn_backend", "auto")),
            "knn_device": str(relabel_cfg.get("knn_device", relabel_cfg.get("device", "cuda:0"))),
            "torch_knn_auto_max_pairs": int(relabel_cfg.get("torch_knn_auto_max_pairs", 1_000_000_000)),
            "torch_knn_query_batch": int(relabel_cfg.get("torch_knn_query_batch", 4096)),
            "torch_knn_reference_batch": int(relabel_cfg.get("torch_knn_reference_batch", 32768)),
        },
        "basin_kinetic_state_stats": basin_kinetic_state_stats,
        "reshaped_basin_groups": reshaped_basin_groups,
        "basin_kinetic_group_labels": basin_kinetic_group_labels,
        "final_kinetic_merges": final_kinetic_merge_rows,
        "final_kinetic_splits": final_kinetic_split_rows,
        "candidate_frames": int(candidate_idx.size),
        "lagged_entropy_classification": lagged_entropy_classification,
        "existing_core_overlap_filter": existing_core_overlap_filter,
        "existing_core_overlap_rows": existing_core_overlap_rows,
        "existing_core_overlap_frames": int(np.sum(existing_core_overlap_mask)),
        "current_entropy_candidate_frames": int(np.sum(current_entropy_candidate_mask)),
        "missing_metastate_h_tau_frames": int(np.sum(missing_metastate_h_tau_mask)),
        "unresolved_lagged_candidate_frames": int(np.sum(unresolved_lagged_candidate_mask)),
        "missing_metastate_raw_candidate_frames": int(np.sum(missing_metastate_raw_mask)),
        "missing_metastate_seed_candidate_frames": int(np.sum(missing_metastate_seed_mask)),
        "missing_metastate_candidate_frames": int(candidate_idx.size),
        "two_state_ambiguous_review_frames": int(np.sum(two_state_ambiguous_review_mask)),
        "candidate_persistence": candidate_persistence,
        "transition_like_candidate_frames": int(np.sum(transition_like_candidate_mask)),
        "persistent_candidate_review_frames": int(np.sum(persistent_candidate_review_mask)),
        "graph_frames": int(graph_idx.size),
        "pair_ambiguous_graph_frames": 0,
        "missing_metastate_graph_frames": int(graph_idx.size),
        "q_max": q_max,
        "q_argmax": q_argmax,
        "surviving_top1_state": top1_state,
        "surviving_top2_state": top2_state,
        "surviving_top1_q": top1_q,
        "surviving_top2_q": top2_q,
        "entropy_norm": entropy_norm,
        "mean_lagged_entropy_norm": mean_lagged_entropy,
        "lagged_high_entropy_fraction": lagged_high_entropy_fraction,
        "label_consistency": label_consistency,
    }


def run_relabel(dataset_path, model_path, config, device="cuda:0", batch_size=65536, dataset_stride=1):
    from .label_diagnostics import _compute_frame_index

    device_obj = setup_device(device)
    stride = int(dataset_stride)
    pack = apply_stride(load_dataset(dataset_path), stride)
    n_states = infer_n_states(pack, config.get("n_states", None))
    model_features, _ = select_model_inputs(pack, config)

    model = load_committor_model(model_path, device_obj)
    q_values = infer_probabilities(model, model_features.float(), device_obj, batch_size=int(batch_size))
    if q_values.ndim != 2 or q_values.shape[1] != n_states:
        raise RuntimeError(f"Model returned q shape {q_values.shape}, expected (_, {n_states}).")

    state = pack.state.detach().cpu().numpy().astype(np.int64)
    weights = pack.weights.detach().cpu().numpy().astype(np.float64)
    traj_id = (
        pack.traj_id.detach().cpu().numpy().astype(np.int64)
        if pack.traj_id is not None
        else np.zeros(state.shape[0], dtype=np.int64)
    )
    frame_index = _compute_frame_index(traj_id)
    graph_features, graph_space = _select_graph_features(pack, model_features, config)

    proposal = propose_relabeling(
        q_values,
        state,
        graph_features,
        weights,
        config,
        trajectory_index=traj_id,
        frame_index=frame_index,
    )
    new_state = proposal["proposed_labels"]
    changed = np.flatnonzero(proposal["changed_mask"])
    removed = np.flatnonzero(proposal["removed_mask"])
    current_entropy_candidates = np.flatnonzero(proposal["current_entropy_candidate_mask"])
    ambiguous_unlabeled = np.flatnonzero(proposal["ambiguous_to_unlabeled_mask"])
    new_core = np.flatnonzero(proposal["new_core_mask"])
    density_shell = np.flatnonzero(proposal["density_shell_mask"])
    reshaped_basin_core = np.flatnonzero(proposal["reshaped_basin_core_mask"])
    reshaped_basin_shell = np.flatnonzero(proposal["reshaped_basin_shell_mask"])
    final_kinetic_split = np.flatnonzero(proposal["final_kinetic_split_mask"])
    transition_like_candidates = np.flatnonzero(proposal["transition_like_candidate_mask"])
    missing_metastate_h_tau = np.flatnonzero(proposal["missing_metastate_h_tau_mask"])
    unresolved_lagged_candidates = np.flatnonzero(proposal["unresolved_lagged_candidate_mask"])
    missing_metastate_raw_candidates = np.flatnonzero(proposal["missing_metastate_raw_candidate_mask"])
    missing_metastate_seed_candidates = np.flatnonzero(proposal["missing_metastate_seed_candidate_mask"])
    missing_metastate_candidates = np.flatnonzero(proposal["missing_metastate_candidate_mask"])
    existing_core_overlap = np.flatnonzero(proposal["existing_core_overlap_mask"])
    persistent_candidate_review = np.flatnonzero(proposal["persistent_candidate_review_mask"])
    review_pair_ambiguous = np.flatnonzero(proposal["review_missing_mask"] & ~proposal["changed_mask"])

    output_dir = ensure_dir(config.get("output_dir", "relabel_out"))
    relabel_cfg = _relabel_cfg(config)
    default_output = os.path.join(output_dir, f"relabeled_dataset{Path(str(dataset_path)).suffix or '.pt'}")
    output_dataset = relabel_cfg.get("output_dataset", default_output)
    saved_dataset = None
    if bool(relabel_cfg.get("write_relabel_dataset", True)):
        saved_dataset = _save_dataset_like_input(dataset_path, output_dataset, pack, new_state, config, stride)

    max_review = relabel_cfg.get("max_review_frames", 20000)
    _write_csv(os.path.join(output_dir, "relabel_removed_states.csv"), proposal["removed_states"])
    _write_csv(os.path.join(output_dir, "relabel_new_core_components.csv"), proposal["new_core_components"])
    _write_csv(os.path.join(output_dir, "relabel_merged_new_states.csv"), proposal["merged_new_states"])
    _write_csv(os.path.join(output_dir, "relabel_existing_core_overlap_summary.csv"), proposal["existing_core_overlap_rows"])
    _write_csv(os.path.join(output_dir, "relabel_basin_kinetic_state_summary.csv"), proposal["basin_kinetic_state_stats"])
    _write_csv(os.path.join(output_dir, "relabel_reshaped_basin_groups.csv"), proposal["reshaped_basin_groups"])
    _write_csv(os.path.join(output_dir, "relabel_final_kinetic_merges.csv"), proposal["final_kinetic_merges"])
    _write_csv(os.path.join(output_dir, "relabel_final_kinetic_splits.csv"), proposal["final_kinetic_splits"])
    _write_csv(
        os.path.join(output_dir, "relabel_changed_frames.csv"),
        _rows_for_frames(changed, state, new_state, proposal, traj_id, frame_index),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_removed_frames.csv"),
        _rows_for_frames(removed, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_current_entropy_candidate_frames.csv"),
        _rows_for_frames(current_entropy_candidates, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_ambiguous_unlabeled_frames.csv"),
        _rows_for_frames(ambiguous_unlabeled, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_new_core_frames.csv"),
        _rows_for_frames(new_core, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_density_shell_frames.csv"),
        _rows_for_frames(density_shell, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_reshaped_basin_core_frames.csv"),
        _rows_for_frames(reshaped_basin_core, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_reshaped_basin_shell_frames.csv"),
        _rows_for_frames(reshaped_basin_shell, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_final_kinetic_split_frames.csv"),
        _rows_for_frames(final_kinetic_split, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_pair_ambiguous_review_frames.csv"),
        _rows_for_frames(review_pair_ambiguous, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_transition_like_candidate_frames.csv"),
        _rows_for_frames(transition_like_candidates, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_missing_metastate_h_tau_frames.csv"),
        _rows_for_frames(missing_metastate_h_tau, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_unresolved_lagged_candidate_frames.csv"),
        _rows_for_frames(unresolved_lagged_candidates, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_missing_metastate_raw_candidate_frames.csv"),
        _rows_for_frames(missing_metastate_raw_candidates, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_missing_metastate_seed_candidate_frames.csv"),
        _rows_for_frames(missing_metastate_seed_candidates, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_missing_metastate_candidate_frames.csv"),
        _rows_for_frames(missing_metastate_candidates, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_existing_core_overlap_frames.csv"),
        _rows_for_frames(existing_core_overlap, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_persistent_candidate_review_frames.csv"),
        _rows_for_frames(persistent_candidate_review, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    for stale in [
        "relabel_rejected_missing_components.csv",
        "relabel_nonmetastable_missing_component_frames.csv",
        "relabel_removed_q_argmax_frames.csv",
        "relabel_removed_q_argmax_candidate_exclusion_frames.csv",
        "relabel_nonpersistent_missing_candidate_frames.csv",
        "radical_far_uncertain_review_frames.csv",
        "radical_removed_states.csv",
        "radical_new_core_components.csv",
        "radical_merged_new_states.csv",
        "radical_changed_frames.csv",
        "radical_removed_frames.csv",
        "radical_ambiguous_unlabeled_frames.csv",
        "radical_new_core_frames.csv",
        "radical_density_shell_frames.csv",
        "radical_pair_ambiguous_review_frames.csv",
        "radical_relabel_summary.yaml",
    ]:
        stale_path = os.path.join(output_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    saved_plots = []
    if bool(config.get("make_plots", True)) and bool(relabel_cfg.get("make_relabel_plots", True)):
        saved_plots = plot_relabel_cv(pack, state, new_state, proposal, config, output_dir)
        try:
            from .plot import plot_basin_kinetic_eigenvalues, plot_basin_kinetic_groups_cv

            group_results = {
                "basin_kinetic_state_stats": proposal["basin_kinetic_state_stats"],
                "basin_kinetic_groups": proposal["reshaped_basin_groups"],
                "_per_frame": {
                    "basin_kinetic_group": proposal["basin_kinetic_group_labels"],
                },
            }
            saved_plots.extend(plot_basin_kinetic_groups_cv(pack, q_values, group_results, config, output_dir))
            saved_plots.extend(plot_basin_kinetic_eigenvalues(group_results, config, output_dir))
        except Exception as exc:
            print(f"[RELABEL] Basin kinetic group plots skipped: {exc}")

    summary = {
        "dataset": os.path.abspath(str(dataset_path)),
        "model": os.path.abspath(str(model_path)),
        "output_dataset": None if saved_dataset is None else os.path.abspath(saved_dataset),
        "dataset_stride": stride,
        "graph_space": graph_space,
        "n_frames": int(state.size),
        "n_changed_frames": int(changed.size),
        "n_removed_frames": int(removed.size),
        "n_current_entropy_candidate_frames": int(current_entropy_candidates.size),
        "n_ambiguous_unlabeled_frames": int(ambiguous_unlabeled.size),
        "n_new_core_frames": int(new_core.size),
        "n_density_shell_frames": int(density_shell.size),
        "n_reshaped_basin_core_frames": int(reshaped_basin_core.size),
        "n_reshaped_basin_shell_frames": int(reshaped_basin_shell.size),
        "n_final_kinetic_split_frames": int(final_kinetic_split.size),
        "n_pair_ambiguous_review_frames": int(review_pair_ambiguous.size),
        "n_transition_like_candidate_frames": int(transition_like_candidates.size),
        "n_missing_metastate_h_tau_frames": int(missing_metastate_h_tau.size),
        "n_unresolved_lagged_candidate_frames": int(unresolved_lagged_candidates.size),
        "n_missing_metastate_raw_candidate_frames": int(missing_metastate_raw_candidates.size),
        "n_missing_metastate_seed_candidate_frames": int(missing_metastate_seed_candidates.size),
        "n_missing_metastate_candidate_frames": int(missing_metastate_candidates.size),
        "n_existing_core_overlap_frames": int(existing_core_overlap.size),
        "n_persistent_candidate_review_frames": int(persistent_candidate_review.size),
        "candidate_frames": proposal["candidate_frames"],
        "lagged_entropy_classification": proposal["lagged_entropy_classification"],
        "existing_core_overlap_filter": proposal["existing_core_overlap_filter"],
        "existing_core_overlap_frames": proposal["existing_core_overlap_frames"],
        "current_entropy_candidate_frames": proposal["current_entropy_candidate_frames"],
        "missing_metastate_h_tau_frames": proposal["missing_metastate_h_tau_frames"],
        "unresolved_lagged_candidate_frames": proposal["unresolved_lagged_candidate_frames"],
        "missing_metastate_raw_candidate_frames": proposal["missing_metastate_raw_candidate_frames"],
        "missing_metastate_seed_candidate_frames": proposal["missing_metastate_seed_candidate_frames"],
        "missing_metastate_candidate_frames": proposal["missing_metastate_candidate_frames"],
        "two_state_ambiguous_review_frames": proposal["two_state_ambiguous_review_frames"],
        "candidate_persistence": proposal["candidate_persistence"],
        "graph_frames": proposal["graph_frames"],
        "pair_ambiguous_graph_frames": proposal["pair_ambiguous_graph_frames"],
        "missing_metastate_graph_frames": proposal["missing_metastate_graph_frames"],
        "removed_state_ids": proposal["removed_state_ids"],
        "removed_states": proposal["removed_states"],
        "knn_backend_config": proposal["knn_backend_config"],
        "new_core_components": proposal["new_core_components"],
        "existing_core_overlap": proposal["existing_core_overlap_rows"],
        "merged_new_states": proposal["merged_new_states"],
        "basin_kinetic_state_stats": proposal["basin_kinetic_state_stats"],
        "reshaped_basin_groups": proposal["reshaped_basin_groups"],
        "final_kinetic_merges": proposal["final_kinetic_merges"],
        "final_kinetic_splits": proposal["final_kinetic_splits"],
        "plots": [os.path.abspath(path) for path in saved_plots],
        "notes": [
            "Relabel mode removes whole states when most frames are problematic.",
            "High current entropy H(x) marks candidate frames unlabeled to reduce contamination during retraining.",
            "Lagged entropy H_tau classifies those frames: high H_tau means missing metastate, low H_tau means transition/review.",
            "Optional existing-core spatial filtering can keep robust-core overlaps for review; by default stable-label pruning and kinetic merging handle duplicates.",
            "kNN distance calculations use relabel.knn_backend='auto' by default: Torch/CUDA under size limits, sklearn fallback for very large exact searches.",
            "New labels are proposed only from high-H(x), high-H_tau missing-metastate candidates.",
            "Two-state ambiguous frames remain review signals, not automatic new labels.",
            "New kNN components with high label-indicator time-correlation exchange are iteratively merged before density trimming.",
            "New cores are assigned only to the highest weighted-density part of each final merged basin.",
            "Existing labels are reshaped to high-confidence q cores, and large disconnected kinetic groups can be split into new labels.",
            "A final lagged kNN consistency check merges labels that behave as the same metastate and splits labels containing weakly exchanging internal components.",
            "Unassigned missing or pair-ambiguous frames are review signals, not automatic labels.",
        ],
    }
    summary_path = os.path.join(output_dir, "relabel_summary.yaml")
    write_yaml(summary, summary_path)
    print(f"[RELABEL] Removed states: {len(proposal['removed_states'])}")
    print(f"[RELABEL] New core components: {len(proposal['new_core_components'])}")
    print(f"[RELABEL] Merged new states: {len(proposal['merged_new_states'])}")
    print(f"[RELABEL] Reshaped basin groups: {len(proposal['reshaped_basin_groups'])}")
    print(f"[RELABEL] Final kinetic merges: {len(proposal['final_kinetic_merges'])}")
    print(f"[RELABEL] Final kinetic splits: {len(proposal['final_kinetic_splits'])}")
    print(f"[RELABEL] Changed frames: {changed.size}")
    if saved_dataset is not None:
        print(f"[RELABEL] Saved dataset: {saved_dataset}")
    if saved_plots:
        print("[RELABEL] Saved plots:")
        for path in saved_plots:
            print(f"  {path}")
    print(f"[RELABEL] Summary: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Apply confidence/kNN-graph relabeling.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = select_section(raw, "RELABEL", "TENSORQ_RELABEL")
    dataset_path = cfg.get("dataset", cfg.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("Relabel config needs 'dataset' or 'dataset_path'.")
    model_path = cfg.get("model")
    if model_path is None:
        raise KeyError("Relabel config needs 'model' (path to trained checkpoint).")

    cfg["output_dir"] = ensure_dir(cfg.get("output_dir", cfg.get("out_dir", "relabel")))
    run_relabel(
        dataset_path=dataset_path,
        model_path=model_path,
        config=cfg,
        device=str(cfg.get("device", "cuda:0")),
        batch_size=int(cfg.get("batch_size", 65536)),
        dataset_stride=int(cfg.get("dataset_stride", 1)),
    )


if __name__ == "__main__":
    main()

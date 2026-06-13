"""Clean relabeling: high-entropy region → k-NN density clustering → time-lag persistence.

Pipeline
--------
1. Remove states whose frames mostly deviate from their assigned q.
2. Find high-entropy / low-qmax frames (truly uncertain — missing-metastate candidates).
3. Build symmetric k-NN graph in feature space over those candidates; extract connected
   components.
4. Select the weighted-density core of each component (keep only the densest part).
5. Check time-lag *self-retention* of each density core — a core that stays together
   across lag times is a genuine metastable basin.
6. Assign new labels to cores that pass the persistence check.
7. Iteratively merge new labels that have high kinetic exchange (indicator time-correlation).
8. Reshape *existing* labels — trim to high-confidence q-cores, split disconnected kinetic
   groups into new labels.

No time-lagged entropy checks are used anywhere in this pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, infer_n_states, load_dataset, select_model_inputs
from ..next_hit.predict import infer_probabilities, load_committor_model
from .kinetic_groups import analyze_basin_kinetic_groups
from .lag_pair_utils import build_lag_pairs
from .settings import analysis_settings
from .utils import _entropy_confidence, _save_dataset_like_input, plot_relabel_cv


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _relabel_cfg(config: dict) -> dict:
    if "relabel" in config and isinstance(config["relabel"], dict):
        return config["relabel"]
    return config.get("radical", {})


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def _standardize_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std


def _sample_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    max_count: int,
    seed: int,
) -> np.ndarray:
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


def _select_graph_features(pack, model_features, config: dict):
    relabel_cfg = _relabel_cfg(config)
    space = str(relabel_cfg.get("graph_space", "cv")).lower()
    if space in {"cv", "cvs", "colvars"}:
        if pack.cv is None:
            raise RuntimeError("relabel.graph_space='cv' requires saved CV data.")
        return pack.cv.detach().cpu().numpy(), "cv"
    if space in {"features", "feature", "model_features", "model"}:
        return model_features.detach().cpu().numpy(), "model_features"
    raise ValueError("relabel.graph_space must be 'cv' or 'model_features'.")


# ---------------------------------------------------------------------------
# k-NN graph → connected components
# ---------------------------------------------------------------------------

def _build_knn_components(
    z: np.ndarray,
    candidate_idx: np.ndarray,
    weights: Optional[np.ndarray],
    cfg: dict,
) -> tuple[list[dict], list[np.ndarray]]:
    """Build a symmetric k-NN graph over *candidate_idx* in feature space *z*
    and return each connected component that meets the minimum size/weight
    thresholds."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import kneighbors_graph

    rows: list[dict] = []
    assignments: list[np.ndarray] = []
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if candidate_idx.size == 0:
        return rows, assignments

    k = int(cfg.get("k_neighbors", 20))
    min_size = int(cfg.get("min_new_core_size", 100))
    min_weight = float(cfg.get("min_new_core_weight", 0.0))

    if candidate_idx.size < min_size:
        return rows, assignments

    # Single-frame corner case
    if candidate_idx.size == 1:
        w = float(weights[candidate_idx[0]]) if weights is not None else 1.0
        if w >= min_weight:
            rows.append({
                "component": 0,
                "n_frames": 1,
                "weighted_population": w,
            })
            assignments.append(candidate_idx)
        return rows, assignments

    graph_neighbors = min(max(1, k), candidate_idx.size - 1)
    graph = kneighbors_graph(
        z[candidate_idx],
        n_neighbors=graph_neighbors,
        mode="connectivity",
        include_self=False,
    )
    graph = graph.maximum(graph.T)  # make symmetric
    n_components, labels = connected_components(graph, directed=False)

    for comp in range(n_components):
        local = np.flatnonzero(labels == comp)
        frame_idx = candidate_idx[local]
        w = float(np.sum(weights[frame_idx])) if weights is not None else float(frame_idx.size)
        if frame_idx.size < min_size or w < min_weight:
            continue
        rows.append({
            "component": int(comp),
            "n_frames": int(frame_idx.size),
            "weighted_population": w,
        })
        assignments.append(frame_idx)

    return rows, assignments


# ---------------------------------------------------------------------------
# Weighted density core selection
# ---------------------------------------------------------------------------

def _component_weights(
    frame_idx: np.ndarray,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    if weights is None:
        return np.ones(frame_idx.size, dtype=np.float64)
    return np.clip(np.asarray(weights, dtype=np.float64)[frame_idx], 0.0, np.inf)


def _weighted_local_density(
    z: np.ndarray,
    frame_idx: np.ndarray,
    weights: Optional[np.ndarray],
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    from sklearn.neighbors import NearestNeighbors

    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    sample_weights = _component_weights(frame_idx, weights)
    if frame_idx.size == 0:
        return np.zeros(0, dtype=np.float64), {"density_k_neighbors": 0, "density_radius_power": np.nan}
    if frame_idx.size == 1:
        return sample_weights.copy(), {"density_k_neighbors": 0, "density_radius_power": 0.0}

    k_default = int(cfg.get("k_neighbors", 20))
    k = int(cfg.get("density_k_neighbors", k_default))
    k = min(max(1, k), frame_idx.size - 1)
    radius_power = float(cfg.get("density_radius_power", min(z.shape[1], 6)))
    radius_power = max(0.0, radius_power)

    n_neighbors = min(frame_idx.size, k + 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(z[frame_idx])
    distances, neighbor_local = nn.kneighbors(z[frame_idx])
    neighbor_weights = np.sum(sample_weights[neighbor_local], axis=1)
    radius = np.maximum(distances[:, -1], 1e-12)
    density = neighbor_weights / np.power(radius, radius_power)
    return density, {"density_k_neighbors": int(k), "density_radius_power": radius_power}


def _select_density_core(
    z: np.ndarray,
    frame_idx: np.ndarray,
    weights: Optional[np.ndarray],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(core_idx, shell_idx, meta)`` where *core_idx* contains the
    highest weighted-density frames (top *density_core_fraction* by cumulative
    weight)."""
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


# ---------------------------------------------------------------------------
# Time-lag persistence (self-retention) check
# ---------------------------------------------------------------------------

def _positive_lag_list(value, fallback) -> list[int]:
    if value is None:
        value = fallback
    if isinstance(value, (int, float)):
        value = [value]
    if value is None:
        return []
    return [int(lag) for lag in value if int(lag) > 0]


def _check_lag_persistence(
    component_rows: list[dict],
    component_frames: list[np.ndarray],
    trajectory_index: Optional[np.ndarray],
    frame_index: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    config: dict,
    n_frames: int,
) -> tuple[list[dict], list[np.ndarray], list[dict], np.ndarray]:
    """Check time-lag self-retention for each candidate component.

    A component is *metastable* if, across the configured lag times, a
    sufficient fraction of its frames stay inside the component.  No
    entropy / q_max checks — purely geometric self-retention.
    """
    settings = analysis_settings(config)
    relabel_cfg = _relabel_cfg(config)

    review_mask = np.zeros(n_frames, dtype=bool)
    accepted_rows: list[dict] = []
    accepted_frames: list[np.ndarray] = []
    rejected_rows: list[dict] = []

    if not component_rows or not component_frames:
        return accepted_rows, accepted_frames, rejected_rows, review_mask

    if trajectory_index is None or frame_index is None:
        for row, frames in zip(component_rows, component_frames):
            row = dict(row)
            row.update(metastable=False,
                       metastability_reason="trajectory_index/frame_index unavailable",
                       best_retention=np.nan, best_retention_lag=-1)
            rejected_rows.append(row)
            review_mask[np.asarray(frames, dtype=np.int64)] = True
        return accepted_rows, accepted_frames, rejected_rows, review_mask

    lag_list = _positive_lag_list(
        relabel_cfg.get("candidate_metastability_lag_list", None),
        settings["lag_list"],
    )
    if not lag_list:
        for row, frames in zip(component_rows, component_frames):
            row = dict(row)
            row.update(metastable=False,
                       metastability_reason="no candidate metastability lags configured",
                       best_retention=np.nan, best_retention_lag=-1)
            rejected_rows.append(row)
            review_mask[np.asarray(frames, dtype=np.int64)] = True
        return accepted_rows, accepted_frames, rejected_rows, review_mask

    min_pairs = int(relabel_cfg.get("candidate_metastability_min_pairs", settings["min_count"]))
    retention_cutoff = float(
        relabel_cfg.get("candidate_metastability_retention_cutoff", settings["persistent_fraction"])
    )
    lag_pairs = build_lag_pairs(
        np.asarray(trajectory_index, dtype=np.int64),
        np.asarray(frame_index, dtype=np.int64),
        lag_list,
    )

    for row, frames in zip(component_rows, component_frames):
        row = dict(row)
        frames = np.asarray(frames, dtype=np.int64)
        in_component = np.zeros(n_frames, dtype=bool)
        in_component[frames] = True
        best_retention = -np.inf
        best_lag = -1
        best_pairs = 0
        valid_lags = 0

        for lag in lag_list:
            idx_t, idx_tau = lag_pairs[int(lag)]
            start = in_component[idx_t]
            n_pairs = int(np.sum(start))
            row[f"n_pairs_lag_{lag}"] = n_pairs
            if n_pairs < min_pairs:
                row[f"retention_lag_{lag}"] = np.nan
                continue

            if weights is None:
                retention = float(np.mean(in_component[idx_tau[start]]))
            else:
                pair_weights = np.clip(np.asarray(weights, dtype=np.float64)[idx_t[start]], 0.0, np.inf)
                denom = float(np.sum(pair_weights))
                retention = (
                    float(np.sum(pair_weights * in_component[idx_tau[start]].astype(np.float64)) / denom)
                    if denom > 0.0
                    else np.nan
                )
            row[f"retention_lag_{lag}"] = retention
            if np.isfinite(retention):
                valid_lags += 1
                if retention > best_retention:
                    best_retention = retention
                    best_lag = int(lag)
                    best_pairs = n_pairs

        metastable = bool(valid_lags > 0 and best_retention >= retention_cutoff)
        row.update(
            metastable=metastable,
            metastability_retention_cutoff=retention_cutoff,
            metastability_min_pairs=min_pairs,
            metastability_valid_lags=int(valid_lags),
            best_retention=float(best_retention) if np.isfinite(best_retention) else np.nan,
            best_retention_lag=int(best_lag),
            best_retention_pairs=int(best_pairs),
            metastability_reason=(
                "accepted: self-retention above cutoff"
                if metastable
                else "review: insufficient self-retention"
            ),
        )
        if metastable:
            accepted_rows.append(row)
            accepted_frames.append(frames)
        else:
            rejected_rows.append(row)
            review_mask[frames] = True

    return accepted_rows, accepted_frames, rejected_rows, review_mask


# ---------------------------------------------------------------------------
# Merge kinetically mixed new labels
# ---------------------------------------------------------------------------

def _new_label_transition_matrices(
    labels: np.ndarray,
    new_labels: list[int],
    lag_pairs: dict,
    lag_list: list[int],
    weights: Optional[np.ndarray],
) -> tuple[dict, dict]:
    labels = np.asarray(labels, dtype=np.int64)
    new_labels = np.sort(np.asarray(new_labels, dtype=np.int64))
    n_labels = int(new_labels.size)
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    matrices: dict[int, np.ndarray] = {}
    count_matrices: dict[int, np.ndarray] = {}

    for lag in lag_list:
        idx_t, idx_tau = lag_pairs[int(lag)]
        start_labels = labels[idx_t]
        end_labels = labels[idx_tau]
        valid = np.isin(start_labels, new_labels) & np.isin(end_labels, new_labels)

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


def _pair_mixing_decision_from_matrices(
    label_i: int,
    label_j: int,
    active_labels: list[int],
    matrices: dict,
    count_matrices: dict,
    lag_list: list[int],
    cfg: dict,
) -> Optional[dict]:
    min_valid = int(cfg.get("merge_min_valid_pairs", cfg.get("min_valid_pairs", 50)))
    transition_cutoff = float(cfg.get("merge_transition_cutoff", cfg.get("mixing_transition_cutoff", 0.05)))
    require_bidirectional = bool(cfg.get("merge_require_bidirectional", True))
    label_to_local = {int(lb): idx for idx, lb in enumerate(active_labels)}
    local_i = label_to_local[int(label_i)]
    local_j = label_to_local[int(label_j)]

    best: Optional[dict] = None
    best_score = -np.inf
    for lag in lag_list:
        lag = int(lag)
        probs = matrices[lag]
        counts = count_matrices[lag]
        n_i = int(np.sum(counts[local_i]))
        n_j = int(np.sum(counts[local_j]))
        p_ij = float(probs[local_i, local_j]) if n_i >= min_valid and np.isfinite(probs[local_i, local_j]) else np.nan
        p_ji = float(probs[local_j, local_i]) if n_j >= min_valid and np.isfinite(probs[local_j, local_i]) else np.nan
        finite = [v for v in (p_ij, p_ji) if np.isfinite(v)]
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
    best.update(
        label_i=int(label_i),
        label_j=int(label_j),
        mixing_score=float(best_score),
        transition_cutoff=transition_cutoff,
        require_bidirectional=require_bidirectional,
    )
    return best


def _merge_label_into(labels: np.ndarray, source: int, target: int) -> None:
    labels[labels == source] = target


def _iteratively_merge_mixed_new_labels(
    labels: np.ndarray,
    new_labels: list[int],
    trajectory_index: Optional[np.ndarray],
    frame_index: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    config: dict,
) -> tuple[np.ndarray, list[dict]]:
    relabel_cfg = _relabel_cfg(config)
    if not bool(relabel_cfg.get("merge_mixed_new_states", True)):
        return labels, []

    new_labels = sorted(int(lb) for lb in new_labels)
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
    merge_rows: list[dict] = []
    labels = labels.copy()

    for iteration in range(max_iterations):
        active = sorted(lb for lb in new_labels if np.any(labels == lb))
        if len(active) < 2:
            break
        matrices, count_matrices = _new_label_transition_matrices(
            labels, active, lag_pairs, lag_list, weights,
        )
        best: Optional[dict] = None
        for idx, label_i in enumerate(active):
            for label_j in active[idx + 1:]:
                decision = _pair_mixing_decision_from_matrices(
                    label_i, label_j, active, matrices, count_matrices, lag_list, merge_cfg,
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


# ---------------------------------------------------------------------------
# Reshape existing basins (kinetic-group splitting)
# ---------------------------------------------------------------------------

def _reshape_existing_basins(
    new_state: np.ndarray,
    state_labels: np.ndarray,
    q_values: np.ndarray,
    graph_features: np.ndarray,
    trajectory_index: Optional[np.ndarray],
    frame_index: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    config: dict,
    next_label: int,
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray, np.ndarray, np.ndarray, int]:
    relabel_cfg = _relabel_cfg(config)
    if not bool(relabel_cfg.get("reshape_existing_basins", True)):
        return (new_state, [], [],
                np.full_like(new_state, -1, dtype=np.int64),
                np.zeros_like(new_state, dtype=bool),
                np.zeros_like(new_state, dtype=bool),
                next_label)
    if trajectory_index is None or frame_index is None:
        return (new_state, [], [],
                np.full_like(new_state, -1, dtype=np.int64),
                np.zeros_like(new_state, dtype=bool),
                np.zeros_like(new_state, dtype=bool),
                next_label)

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
        return (new_state, [], [],
                np.full_like(new_state, -1, dtype=np.int64),
                np.zeros_like(new_state, dtype=bool),
                np.zeros_like(new_state, dtype=bool),
                next_label)

    lag_pairs = build_lag_pairs(trajectory_index, frame_index, lag_list)
    state_rows, group_rows, group_labels = analyze_basin_kinetic_groups(
        state_labels, q_values, lag_pairs, config,
        weights=weights, n_states=int(q_values.shape[1]), features=graph_features,
    )

    trim_shell = bool(relabel_cfg.get("trim_existing_to_high_confidence_core", True))
    split_groups = bool(relabel_cfg.get("split_existing_kinetic_groups", True))
    min_groups_to_split = int(relabel_cfg.get("min_kinetic_groups_to_split", 2))
    reshape_core_mask = np.zeros_like(new_state, dtype=bool)
    reshape_shell_mask = np.zeros_like(new_state, dtype=bool)
    assignment_rows: list[dict] = []

    groups_by_state: dict[int, list[dict]] = {}
    for row in group_rows:
        groups_by_state.setdefault(int(row["state"]), []).append(row)

    for state, rows in groups_by_state.items():
        rows = sorted(rows, key=lambda r: (int(r["rank_within_state"]), int(r["kinetic_group"])))
        if not rows:
            continue
        state_mask = state_labels == state
        state_current_mask = new_state == state
        group_core_mask = np.isin(group_labels, [int(r["kinetic_group"]) for r in rows])

        if trim_shell:
            shell = state_current_mask & ~group_core_mask
            if np.any(shell):
                new_state[shell] = -1
                reshape_shell_mask[shell] = True

        split_recommended = any(bool(r.get("split_recommended", False)) for r in rows)
        do_split = split_groups and split_recommended and len(rows) >= min_groups_to_split
        for row_idx, row in enumerate(rows):
            group_id = int(row["kinetic_group"])
            group_mask = (group_labels == group_id) & state_mask
            if not np.any(group_mask):
                continue
            target_label = state if row_idx == 0 or not do_split else int(next_label)
            if do_split and row_idx > 0:
                next_label += 1
            new_state[group_mask] = target_label
            reshape_core_mask[group_mask] = True
            assignment_rows.append({
                "old_state": int(state),
                "kinetic_group": group_id,
                "rank_within_state": int(row["rank_within_state"]),
                "assigned_state": int(target_label),
                "n_frames": int(row["n_frames"]),
                "weighted_population": float(row["weighted_population"]),
                "fraction_of_state": float(row["fraction_of_state"]),
                "mean_q_own": float(row["mean_q_own"]),
                "mean_q_max": float(row["mean_q_max"]),
                "mean_entropy_norm": float(row["mean_entropy_norm"]),
                "action": "split_to_new_label" if do_split and row_idx > 0 else "keep_as_existing_label",
            })

    return new_state, state_rows, assignment_rows, group_labels, reshape_core_mask, reshape_shell_mask, next_label


# ---------------------------------------------------------------------------
# Main proposal function
# ---------------------------------------------------------------------------

def propose_relabeling(
    q_values: np.ndarray,
    state_labels: np.ndarray,
    graph_features: np.ndarray,
    weights: Optional[np.ndarray],
    config: dict,
    trajectory_index: Optional[np.ndarray] = None,
    frame_index: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Propose new state labels.

    1. Remove states whose frames mostly disagree with their assigned q.
    2. Identify high-entropy / low-qmax frames as missing-metastate candidates.
    3. k-NN graph → connected components in feature space.
    4. Density-core selection (keep only the densest fraction).
    5. Time-lag self-retention check → only persistent cores become new labels.
    6. Merge kinetically mixed new labels.
    7. Reshape existing labels via basin kinetic-group splitting.
    """
    settings = analysis_settings(config)
    relabel_cfg = _relabel_cfg(config)

    q_label_cutoff = float(settings["q_cutoff"])
    entropy_cutoff = float(settings["entropy_cutoff"])
    remove_cutoff = float(relabel_cfg.get("remove_problem_fraction_cutoff", 0.9))
    candidate_qmax_cutoff = float(relabel_cfg.get("candidate_qmax_cutoff", q_label_cutoff))
    candidate_logic = str(relabel_cfg.get("candidate_logic", "and")).lower()

    state_labels = np.asarray(state_labels, dtype=np.int64)
    new_state = state_labels.copy()
    n_states = int(q_values.shape[1])
    q_max, q_argmax, _, entropy_norm = _entropy_confidence(q_values)
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)

    # Per-frame q-consistency with assigned label
    label_consistency = np.full(state_labels.shape[0], np.nan, dtype=np.float64)
    valid = (state_labels >= 0) & (state_labels < n_states)
    valid_idx = np.flatnonzero(valid)
    label_consistency[valid] = q_values[valid_idx, state_labels[valid]]

    # ---- Step 1: remove bad states ----
    removed_states: list[dict] = []
    removed_state_ids: list[int] = []
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
        if problem_fraction >= remove_cutoff:
            new_state[mask] = -1
            removed_mask |= mask
            removed_state_ids.append(state)
            removed_states.append({
                "removed_state": int(state),
                "n_frames": n_state,
                "n_problem_frames": int(np.sum(problem)),
                "problem_fraction": problem_fraction,
                "mean_q_own": float(np.mean(q_values[mask, state])),
                "mean_entropy_norm": float(np.mean(entropy_norm[mask])),
                "reason": "state removed: most frames are low-consistency or high-entropy",
            })

    # ---- Step 2: unlabel uncertain frames → new-state candidate pool ----
    uncertain = entropy_norm >= entropy_cutoff
    weak_destination = q_max <= candidate_qmax_cutoff
    uncertain_candidate = (uncertain | weak_destination) if candidate_logic == "or" else (uncertain & weak_destination)
    ambiguous_to_unlabeled_mask = uncertain_candidate & (new_state >= 0)
    new_state[ambiguous_to_unlabeled_mask] = -1

    # Missing-metastate candidates: unlabeled, high entropy, low qmax
    # (no two_state_mixture exclusion — persistence check is the gatekeeper)
    candidate_mask = (
        (new_state == -1)
        & (entropy_norm >= entropy_cutoff)
        & (q_max < q_label_cutoff)
    )
    candidate_idx = np.flatnonzero(candidate_mask)

    # ---- Step 3: k-NN graph → connected components ----
    z = _standardize_features(graph_features)
    seed = int(relabel_cfg.get("random_seed", 0))
    max_graph = int(relabel_cfg.get("max_graph_frames", 200000))
    candidate_scores = (
        entropy_norm[candidate_idx]
        * (1.0 - np.clip(q_max[candidate_idx], 0.0, 1.0))
        * (weights_arr[candidate_idx] if weights_arr is not None else 1.0)
    )
    graph_idx = _sample_indices(candidate_idx, candidate_scores, max_graph, seed)

    raw_component_rows: list[dict] = []
    raw_component_frames: list[np.ndarray] = []
    if graph_idx.size:
        raw_component_rows, raw_component_frames = _build_knn_components(
            z, graph_idx, weights_arr, relabel_cfg,
        )

    # ---- Step 4: density-core selection + Step 5: persistence check ----
    persistent_rows: list[dict] = []
    persistent_frames: list[np.ndarray] = []
    rejected_component_rows: list[dict] = []
    nonmetastable_mask = np.zeros_like(candidate_mask, dtype=bool)

    if raw_component_rows:
        # First select density core for each raw component
        core_rows: list[dict] = []
        core_frames: list[np.ndarray] = []
        for row, frames in zip(raw_component_rows, raw_component_frames):
            core_idx, shell_idx, density_meta = _select_density_core(
                z, frames, weights_arr, relabel_cfg,
            )
            if core_idx.size == 0:
                continue
            row = dict(row)
            row.update(density_meta)
            row["component_n_frames_raw"] = int(frames.size)
            row["n_frames"] = int(core_idx.size)
            core_rows.append(row)
            core_frames.append(core_idx)

        # Then check lag persistence on the density cores
        if core_rows:
            persistent_rows, persistent_frames, rejected_component_rows, nonmetastable_mask = \
                _check_lag_persistence(
                    core_rows, core_frames,
                    trajectory_index, frame_index,
                    weights_arr, config, state_labels.shape[0],
                )

    # ---- Step 6: assign new labels to persistent cores ----
    next_label = int(np.max(state_labels[state_labels >= 0]) + 1) if np.any(state_labels >= 0) else 0
    new_core_mask = np.zeros(state_labels.shape[0], dtype=bool)
    provisional_new_labels: list[int] = []
    for row, frames in zip(persistent_rows, persistent_frames):
        row["new_state"] = int(next_label)
        row["provisional_new_state"] = int(next_label)
        row["component_candidate_frames"] = int(frames.size)
        row["component_candidate_weight"] = (
            float(np.sum(weights_arr[frames])) if weights_arr is not None else float(frames.size)
        )
        new_state[frames] = next_label
        new_core_mask[frames] = True
        provisional_new_labels.append(next_label)
        next_label += 1

    # ---- Step 7: merge kinetically mixed new labels ----
    new_state, merged_new_state_rows = _iteratively_merge_mixed_new_labels(
        new_state, provisional_new_labels, trajectory_index, frame_index, weights_arr, config,
    )

    # Update component rows with final labels after merging
    component_final_labels: dict[int, int] = {}
    for idx, row in enumerate(persistent_rows):
        if len(persistent_frames[idx]) > 0:
            provisional = int(row["provisional_new_state"])
            final_label = int(new_state[persistent_frames[idx][0]])
            component_final_labels[provisional] = final_label

    for idx, row in enumerate(persistent_rows):
        provisional = int(row["provisional_new_state"])
        final_label = component_final_labels.get(provisional, provisional)
        row["new_state"] = int(final_label)
        comp_frames = persistent_frames[idx]
        row["component_dense_core_frames"] = int(np.sum(new_state[comp_frames] == final_label))

    # ---- Step 8: reshape existing basins via kinetic-group splitting ----
    next_label = int(np.max(new_state[new_state >= 0]) + 1) if np.any(new_state >= 0) else int(next_label)
    (
        new_state,
        basin_kinetic_state_stats,
        reshaped_basin_groups,
        basin_kinetic_group_labels,
        reshaped_basin_core_mask,
        reshaped_basin_shell_mask,
        next_label,
    ) = _reshape_existing_basins(
        new_state, state_labels, q_values, graph_features,
        trajectory_index, frame_index, weights_arr, config, next_label,
    )

    changed_mask = new_state != state_labels
    # review signals: frames that were considered but not assigned to any new label
    review_missing_mask = (
        (ambiguous_to_unlabeled_mask & ~new_core_mask)
        | nonmetastable_mask
        | (candidate_mask & ~new_core_mask)
    )

    return {
        "proposed_labels": new_state,
        "changed_mask": changed_mask,
        "removed_mask": removed_mask,
        "ambiguous_to_unlabeled_mask": ambiguous_to_unlabeled_mask,
        "new_core_mask": new_core_mask,
        "density_shell_mask": np.zeros(state_labels.shape[0], dtype=bool),
        "reshaped_basin_core_mask": reshaped_basin_core_mask,
        "reshaped_basin_shell_mask": reshaped_basin_shell_mask,
        "review_mixed_mask": np.zeros(state_labels.shape[0], dtype=bool),
        "review_missing_mask": review_missing_mask,
        "nonmetastable_missing_component_mask": nonmetastable_mask,
        "removed_states": removed_states,
        "removed_state_ids": removed_state_ids,
        "new_core_components": persistent_rows,
        "rejected_missing_components": rejected_component_rows,
        "merged_new_states": merged_new_state_rows,
        "basin_kinetic_state_stats": basin_kinetic_state_stats,
        "reshaped_basin_groups": reshaped_basin_groups,
        "basin_kinetic_group_labels": basin_kinetic_group_labels,
        "candidate_frames": int(candidate_idx.size),
        "nonmetastable_component_frames": int(np.sum(nonmetastable_mask)),
        "rejected_components_count": len(rejected_component_rows),
        "graph_frames": int(graph_idx.size),
        "q_max": q_max,
        "q_argmax": q_argmax,
        "entropy_norm": entropy_norm,
        "label_consistency": label_consistency,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_relabel(
    dataset_path: str,
    model_path: str,
    config: dict,
    device: str = "cuda:0",
    batch_size: int = 65536,
    dataset_stride: int = 1,
) -> dict:
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
        q_values, state, graph_features, weights, config,
        trajectory_index=traj_id, frame_index=frame_index,
    )
    new_state = proposal["proposed_labels"]
    changed = np.flatnonzero(proposal["changed_mask"])
    removed = np.flatnonzero(proposal["removed_mask"])
    ambiguous_unlabeled = np.flatnonzero(proposal["ambiguous_to_unlabeled_mask"])
    new_core = np.flatnonzero(proposal["new_core_mask"])
    reshaped_basin_core = np.flatnonzero(proposal["reshaped_basin_core_mask"])
    reshaped_basin_shell = np.flatnonzero(proposal["reshaped_basin_shell_mask"])
    nonmetastable = np.flatnonzero(proposal["nonmetastable_missing_component_mask"])

    output_dir = ensure_dir(config.get("output_dir", "relabel_out"))
    relabel_cfg = _relabel_cfg(config)
    default_output = os.path.join(output_dir, f"relabeled_dataset{Path(str(dataset_path)).suffix or '.pt'}")
    output_dataset = relabel_cfg.get("output_dataset", default_output)
    saved_dataset = None
    if bool(relabel_cfg.get("write_relabel_dataset", True)):
        saved_dataset = _save_dataset_like_input(dataset_path, output_dataset, pack, new_state, config, stride)

    saved_plots: list[str] = []
    if bool(config.get("make_plots", True)) and bool(relabel_cfg.get("make_relabel_plots", True)):
        saved_plots = plot_relabel_cv(pack, state, new_state, proposal, config, output_dir)
        try:
            from .plot import plot_basin_kinetic_eigenvalues, plot_basin_kinetic_groups_cv
            group_results = {
                "basin_kinetic_state_stats": proposal["basin_kinetic_state_stats"],
                "basin_kinetic_groups": proposal["reshaped_basin_groups"],
                "_per_frame": {"basin_kinetic_group": proposal["basin_kinetic_group_labels"]},
            }
            saved_plots.extend(plot_basin_kinetic_groups_cv(pack, q_values, group_results, config, output_dir))
            saved_plots.extend(plot_basin_kinetic_eigenvalues(group_results, config, output_dir))
        except Exception as exc:
            print(f"[RELABEL] Basin kinetic group plots skipped: {exc}")

    summary: dict[str, Any] = {
        "dataset": os.path.abspath(str(dataset_path)),
        "model": os.path.abspath(str(model_path)),
        "output_dataset": None if saved_dataset is None else os.path.abspath(saved_dataset),
        "dataset_stride": stride,
        "graph_space": graph_space,
        "n_frames": int(state.size),
        "n_changed_frames": int(changed.size),
        "n_removed_frames": int(removed.size),
        "n_ambiguous_unlabeled_frames": int(ambiguous_unlabeled.size),
        "n_new_core_frames": int(new_core.size),
        "n_reshaped_basin_core_frames": int(reshaped_basin_core.size),
        "n_reshaped_basin_shell_frames": int(reshaped_basin_shell.size),
        "n_nonmetastable_component_frames": int(nonmetastable.size),
        "candidate_frames": proposal["candidate_frames"],
        "graph_frames": proposal["graph_frames"],
        "rejected_components_count": proposal["rejected_components_count"],
        "removed_state_ids": proposal["removed_state_ids"],
        "removed_states": proposal["removed_states"],
        "new_core_components": proposal["new_core_components"],
        "rejected_missing_components": proposal["rejected_missing_components"],
        "merged_new_states": proposal["merged_new_states"],
        "basin_kinetic_state_stats": proposal["basin_kinetic_state_stats"],
        "reshaped_basin_groups": proposal["reshaped_basin_groups"],
        "plots": [os.path.abspath(p) for p in saved_plots],
        "notes": [
            "Pipeline: remove bad states → high-entropy candidates → k-NN components",
            "→ density-core selection → time-lag self-retention check → new labels",
            "→ merge kinetically mixed → reshape existing basins via kinetic groups.",
            "No time-lagged entropy checks are used.",
            "Density cores are checked for self-retention across lag times (purely geometric).",
            "Only components whose density core is self-retaining become new labels.",
            "Existing labels are reshaped to high-confidence q cores; disconnected kinetic groups can be split.",
        ],
    }
    summary_path = os.path.join(output_dir, "relabel_summary.yaml")
    write_yaml(summary, summary_path)
    print(f"[RELABEL] Removed states: {len(proposal['removed_states'])}")
    print(f"[RELABEL] New core components: {len(proposal['new_core_components'])}")
    print(f"[RELABEL] Merged new states: {len(proposal['merged_new_states'])}")
    print(f"[RELABEL] Reshaped basin groups: {len(proposal['reshaped_basin_groups'])}")
    print(f"[RELABEL] Changed frames: {changed.size}")
    if saved_dataset is not None:
        print(f"[RELABEL] Saved dataset: {saved_dataset}")
    if saved_plots:
        print("[RELABEL] Saved plots:")
        for p in saved_plots:
            print(f"  {p}")
    print(f"[RELABEL] Summary: {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean relabeling (no time-lagged entropy).")
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

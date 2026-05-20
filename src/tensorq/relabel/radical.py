from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, infer_n_states, load_dataset, select_model_inputs
from ..next_hit.predict import infer_probabilities, load_committor_model
from .apply import (
    _entropy_confidence,
    _rows_for_frames,
    _save_dataset_like_input,
    _write_csv,
    plot_relabel_cv,
)
from .lag_pair_utils import build_lag_pairs


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


def _select_graph_features(pack, model_features, config):
    radical_cfg = config.get("radical", {})
    space = str(radical_cfg.get("graph_space", "cv")).lower()
    if space in {"cv", "cvs", "colvars"}:
        if pack.cv is None:
            raise RuntimeError("radical.graph_space='cv' requires saved CV data.")
        return pack.cv.detach().cpu().numpy(), "cv"
    if space in {"features", "feature", "model_features", "model"}:
        return model_features.detach().cpu().numpy(), "model_features"
    raise ValueError("radical.graph_space must be 'cv' or 'model_features'.")


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


def _knn_pair_components(z, pair_candidate_idx, pair_a, pair_b, q_values, weights, cfg):
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import kneighbors_graph

    rows = []
    assignments = []
    pair_candidate_idx = np.asarray(pair_candidate_idx, dtype=np.int64)
    if pair_candidate_idx.size == 0:
        return rows, assignments

    k = int(cfg.get("k_neighbors", 20))
    min_size = int(cfg.get("min_new_core_size", 100))
    min_weight = float(cfg.get("min_new_core_weight", 0.0))

    if pair_candidate_idx.size < min_size:
        return rows, assignments

    if pair_candidate_idx.size == 1:
        weighted_population = float(weights[pair_candidate_idx[0]]) if weights is not None else 1.0
        row = {
            "component": 0,
            "state_a": int(pair_a),
            "state_b": int(pair_b),
            "n_frames": 1,
            "weighted_population": weighted_population,
            "mean_q_state_a": float(q_values[pair_candidate_idx[0], pair_a]),
            "mean_q_state_b": float(q_values[pair_candidate_idx[0], pair_b]),
            "fraction_argmax_state_a": float(q_values[pair_candidate_idx[0], pair_a] >= q_values[pair_candidate_idx[0], pair_b]),
        }
        if weighted_population >= min_weight:
            rows.append(row)
            assignments.append(pair_candidate_idx)
        return rows, assignments

    graph_neighbors = min(max(1, k), pair_candidate_idx.size - 1)
    graph = kneighbors_graph(
        z[pair_candidate_idx],
        n_neighbors=graph_neighbors,
        mode="connectivity",
        include_self=False,
    )
    graph = graph.maximum(graph.T)
    n_components, labels = connected_components(graph, directed=False)

    for component in range(n_components):
        local = np.flatnonzero(labels == component)
        frame_idx = pair_candidate_idx[local]
        weighted_population = float(np.sum(weights[frame_idx])) if weights is not None else float(frame_idx.size)
        if frame_idx.size < min_size or weighted_population < min_weight:
            continue
        row = {
            "component": int(component),
            "state_a": int(pair_a),
            "state_b": int(pair_b),
            "n_frames": int(frame_idx.size),
            "weighted_population": weighted_population,
            "mean_q_state_a": float(np.mean(q_values[frame_idx, pair_a])),
            "mean_q_state_b": float(np.mean(q_values[frame_idx, pair_b])),
            "fraction_argmax_state_a": float(np.mean(np.argmax(q_values[frame_idx][:, [pair_a, pair_b]], axis=1) == 0)),
        }
        rows.append(row)
        assignments.append(frame_idx)

    return rows, assignments


def _component_weights(frame_idx, weights):
    if weights is None:
        return np.ones(frame_idx.size, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)[frame_idx]
    return np.clip(sample_weights, 0.0, np.inf)


def _weighted_local_density(z, frame_idx, weights, cfg):
    from sklearn.neighbors import NearestNeighbors

    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    sample_weights = _component_weights(frame_idx, weights)
    if frame_idx.size == 0:
        return np.zeros(0, dtype=np.float64), {
            "density_k_neighbors": 0,
            "density_radius_power": np.nan,
        }
    if frame_idx.size == 1:
        return sample_weights.copy(), {
            "density_k_neighbors": 0,
            "density_radius_power": 0.0,
        }

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
    return density, {
        "density_k_neighbors": int(k),
        "density_radius_power": radius_power,
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


def _weighted_indicator_mean(mask, weights):
    if mask.size == 0:
        return np.nan
    if weights is None:
        return float(np.mean(mask))
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(weights * mask.astype(np.float64)) / denom)


def _label_pair_indicator_correlations(labels, label_i, label_j, lag_pairs, lag, weights=None, min_valid_pairs=50):
    idx_t, idx_tau = lag_pairs[lag]
    labels_t = labels[idx_t]
    labels_tau = labels[idx_tau]
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)

    start_i = labels_t == label_i
    start_j = labels_t == label_j
    n_i = int(np.sum(start_i))
    n_j = int(np.sum(start_j))

    row = {
        "lag": int(lag),
        "n_pairs_i": n_i,
        "n_pairs_j": n_j,
        "p_i_to_i": np.nan,
        "p_i_to_j": np.nan,
        "p_j_to_i": np.nan,
        "p_j_to_j": np.nan,
    }

    if n_i >= min_valid_pairs:
        w_i = None if weights_arr is None else weights_arr[idx_t[start_i]]
        row["p_i_to_i"] = _weighted_indicator_mean(labels_tau[start_i] == label_i, w_i)
        row["p_i_to_j"] = _weighted_indicator_mean(labels_tau[start_i] == label_j, w_i)

    if n_j >= min_valid_pairs:
        w_j = None if weights_arr is None else weights_arr[idx_t[start_j]]
        row["p_j_to_i"] = _weighted_indicator_mean(labels_tau[start_j] == label_i, w_j)
        row["p_j_to_j"] = _weighted_indicator_mean(labels_tau[start_j] == label_j, w_j)

    return row


def _pair_mixing_decision(labels, label_i, label_j, lag_pairs, lag_list, weights, cfg):
    min_valid = int(cfg.get("merge_min_valid_pairs", cfg.get("min_valid_pairs", 50)))
    transition_cutoff = float(cfg.get("merge_transition_cutoff", cfg.get("mixing_transition_cutoff", 0.05)))
    require_bidirectional = bool(cfg.get("merge_require_bidirectional", True))

    best = None
    best_score = -np.inf
    for lag in lag_list:
        row = _label_pair_indicator_correlations(
            labels,
            label_i,
            label_j,
            lag_pairs,
            int(lag),
            weights=weights,
            min_valid_pairs=min_valid,
        )
        p_ij = row["p_i_to_j"]
        p_ji = row["p_j_to_i"]
        finite = [value for value in (p_ij, p_ji) if np.isfinite(value)]
        if not finite:
            continue

        if require_bidirectional:
            if not (np.isfinite(p_ij) and np.isfinite(p_ji)):
                continue
            score = min(float(p_ij), float(p_ji))
        else:
            score = max(float(value) for value in finite)

        if score > best_score:
            best_score = score
            best = row

    if best is None or best_score < transition_cutoff:
        return None

    best = dict(best)
    best.update({
        "label_i": int(label_i),
        "label_j": int(label_j),
        "mixing_score": float(best_score),
        "transition_cutoff": transition_cutoff,
        "require_bidirectional": require_bidirectional,
    })
    return best


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
    radical_cfg = config.get("radical", {})
    if not bool(radical_cfg.get("merge_mixed_new_states", True)):
        return labels, []

    new_labels = sorted(int(label) for label in new_labels)
    if len(new_labels) < 2 or trajectory_index is None or frame_index is None:
        return labels, []

    kinetics_cfg = config.get("kinetics", {})
    lag_list = radical_cfg.get("merge_lag_list", kinetics_cfg.get("lag_list", [1, 2, 5, 10, 20]))
    if lag_list is None:
        lag_list = kinetics_cfg.get("lag_list", [1, 2, 5, 10, 20])
    if isinstance(lag_list, (int, float)):
        lag_list = [lag_list]
    lag_list = [int(lag) for lag in lag_list if int(lag) > 0]
    if not lag_list:
        return labels, []

    merge_cfg = dict(radical_cfg)
    merge_cfg["min_valid_pairs"] = int(kinetics_cfg.get("min_valid_pairs", 50))
    lag_pairs = build_lag_pairs(trajectory_index, frame_index, lag_list)

    max_iterations = int(radical_cfg.get("merge_max_iterations", max(1, len(new_labels) * len(new_labels))))
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


def propose_radical_relabeling(
    q_values,
    state_labels,
    graph_features,
    weights,
    config,
    trajectory_index=None,
    frame_index=None,
):
    confidence_cfg = config.get("confidence", {})
    radical_cfg = config.get("radical", {})

    q_label_cutoff = float(confidence_cfg.get("q_label_cutoff", 0.7))
    entropy_cutoff = float(confidence_cfg.get("entropy_cutoff_ambiguous", 0.5))
    remove_cutoff = float(radical_cfg.get("remove_problem_fraction_cutoff", 0.9))
    candidate_entropy_cutoff = float(radical_cfg.get("candidate_entropy_cutoff", entropy_cutoff))
    candidate_qmax_cutoff = float(radical_cfg.get("candidate_qmax_cutoff", q_label_cutoff))
    candidate_logic = str(radical_cfg.get("candidate_logic", "and")).lower()
    top2_min_probability = float(radical_cfg.get("top2_min_probability", 0.2))
    top2_margin_cutoff = float(radical_cfg.get("top2_margin_cutoff", 0.2))

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
                "reason": "state removed because most frames are low-consistency or high-entropy",
            })

    uncertain = entropy_norm >= candidate_entropy_cutoff
    weak_destination = q_max <= candidate_qmax_cutoff
    if candidate_logic == "or":
        uncertain_candidate = uncertain | weak_destination
    else:
        uncertain_candidate = uncertain & weak_destination

    ambiguous_to_unlabeled_mask = uncertain_candidate & (new_state >= 0)
    new_state[ambiguous_to_unlabeled_mask] = -1

    surviving_states = np.array(
        [state for state in range(n_states) if state not in set(removed_state_ids)],
        dtype=np.int64,
    )
    top1_state, top2_state, top1_q, top2_q = _surviving_top2(q_values, surviving_states)
    two_state_ambiguous = (
        (new_state == -1)
        & uncertain_candidate
        & (top1_state >= 0)
        & (top2_state >= 0)
        & (top2_q >= top2_min_probability)
        & ((top1_q - top2_q) <= top2_margin_cutoff)
    )

    candidate_mask = two_state_ambiguous
    candidate_idx = np.flatnonzero(candidate_mask)

    z = _standardize_features(graph_features)
    seed = int(radical_cfg.get("random_seed", 0))
    max_graph = int(radical_cfg.get("max_graph_frames", 200000))
    candidate_scores = (
        entropy_norm[candidate_idx]
        * (1.0 - np.clip(q_max[candidate_idx], 0.0, 1.0))
        * (weights[candidate_idx] if weights is not None else 1.0)
    )
    graph_idx = _sample_indices(candidate_idx, candidate_scores, max_graph, seed)

    component_rows = []
    component_frames = []
    if graph_idx.size:
        pairs = np.sort(np.column_stack([top1_state[graph_idx], top2_state[graph_idx]]), axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        for pair_a, pair_b in unique_pairs:
            pair_mask = (pairs[:, 0] == pair_a) & (pairs[:, 1] == pair_b)
            pair_idx = graph_idx[pair_mask]
            rows, assignments = _knn_pair_components(
                z, pair_idx, int(pair_a), int(pair_b), q_values, weights, radical_cfg
            )
            component_rows.extend(rows)
            component_frames.extend(assignments)

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
    final_density = {}
    active_final_labels = sorted({label for label in component_final_labels.values() if label >= 0})
    for final_label in active_final_labels:
        final_idx = np.flatnonzero(new_state == final_label)
        core_idx, shell_idx, density_meta = _select_weighted_density_core(
            z, final_idx, weights, radical_cfg
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
            "dense_core_enabled": bool(radical_cfg.get("density_core_enabled", True)),
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

    changed_mask = new_state != state_labels
    unassigned_pair_ambiguous = two_state_ambiguous & ~new_core_mask

    return {
        "proposed_labels": new_state,
        "changed_mask": changed_mask,
        "removed_mask": removed_mask,
        "ambiguous_to_unlabeled_mask": ambiguous_to_unlabeled_mask,
        "new_core_mask": new_core_mask,
        "density_shell_mask": density_shell_mask,
        "review_mixed_mask": np.zeros(state_labels.shape[0], dtype=bool),
        "review_missing_mask": unassigned_pair_ambiguous,
        "removed_states": removed_states,
        "removed_state_ids": removed_state_ids,
        "new_core_components": component_rows,
        "merged_new_states": merged_new_state_rows,
        "candidate_frames": int(candidate_idx.size),
        "graph_frames": int(graph_idx.size),
        "pair_ambiguous_graph_frames": int(graph_idx.size),
        "q_max": q_max,
        "q_argmax": q_argmax,
        "surviving_top1_state": top1_state,
        "surviving_top2_state": top2_state,
        "surviving_top1_q": top1_q,
        "surviving_top2_q": top2_q,
        "entropy_norm": entropy_norm,
        "label_consistency": label_consistency,
    }


def run_radical_relabel(dataset_path, model_path, config, device="cuda:0", batch_size=65536, dataset_stride=1):
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

    proposal = propose_radical_relabeling(
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
    ambiguous_unlabeled = np.flatnonzero(proposal["ambiguous_to_unlabeled_mask"])
    new_core = np.flatnonzero(proposal["new_core_mask"])
    density_shell = np.flatnonzero(proposal["density_shell_mask"])
    review_pair_ambiguous = np.flatnonzero(proposal["review_missing_mask"] & ~proposal["changed_mask"])

    output_dir = ensure_dir(config.get("output_dir", "relabel_out"))
    radical_cfg = config.get("radical", {})
    default_output = os.path.join(output_dir, f"radical_relabeled_dataset{Path(str(dataset_path)).suffix or '.pt'}")
    output_dataset = radical_cfg.get("output_dataset", default_output)
    saved_dataset = None
    if bool(radical_cfg.get("write_relabel_dataset", True)):
        saved_dataset = _save_dataset_like_input(dataset_path, output_dataset, pack, new_state, config, stride)

    max_review = radical_cfg.get("max_review_frames", 20000)
    _write_csv(os.path.join(output_dir, "radical_removed_states.csv"), proposal["removed_states"])
    _write_csv(os.path.join(output_dir, "radical_new_core_components.csv"), proposal["new_core_components"])
    _write_csv(os.path.join(output_dir, "radical_merged_new_states.csv"), proposal["merged_new_states"])
    _write_csv(
        os.path.join(output_dir, "radical_changed_frames.csv"),
        _rows_for_frames(changed, state, new_state, proposal, traj_id, frame_index),
    )
    _write_csv(
        os.path.join(output_dir, "radical_removed_frames.csv"),
        _rows_for_frames(removed, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "radical_ambiguous_unlabeled_frames.csv"),
        _rows_for_frames(ambiguous_unlabeled, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "radical_new_core_frames.csv"),
        _rows_for_frames(new_core, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "radical_density_shell_frames.csv"),
        _rows_for_frames(density_shell, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "radical_pair_ambiguous_review_frames.csv"),
        _rows_for_frames(review_pair_ambiguous, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    stale_far_review = os.path.join(output_dir, "radical_far_uncertain_review_frames.csv")
    if os.path.exists(stale_far_review):
        os.remove(stale_far_review)

    saved_plots = []
    if bool(config.get("make_plots", True)) and bool(radical_cfg.get("make_relabel_plots", True)):
        saved_plots = plot_relabel_cv(pack, state, new_state, proposal, config, output_dir)

    summary = {
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
        "n_density_shell_frames": int(density_shell.size),
        "n_pair_ambiguous_review_frames": int(review_pair_ambiguous.size),
        "candidate_frames": proposal["candidate_frames"],
        "graph_frames": proposal["graph_frames"],
        "pair_ambiguous_graph_frames": proposal["pair_ambiguous_graph_frames"],
        "removed_state_ids": proposal["removed_state_ids"],
        "removed_states": proposal["removed_states"],
        "new_core_components": proposal["new_core_components"],
        "merged_new_states": proposal["merged_new_states"],
        "plots": [os.path.abspath(path) for path in saved_plots],
        "notes": [
            "Radical mode removes whole states when most frames are problematic.",
            "High-uncertainty candidate frames are marked unlabeled before graph components are assigned new labels.",
            "Removed-state committor dimensions are ignored when selecting the two-state ambiguous graph pool.",
            "New kNN components with high label-indicator time-correlation exchange are iteratively merged before density trimming.",
            "New cores are assigned only to the highest weighted-density part of each final merged basin.",
            "Unassigned pair-ambiguous graph frames are review signals, not automatic labels.",
        ],
    }
    summary_path = os.path.join(output_dir, "radical_relabel_summary.yaml")
    write_yaml(summary, summary_path)
    print(f"[RADICAL RELABEL] Removed states: {len(proposal['removed_states'])}")
    print(f"[RADICAL RELABEL] New core components: {len(proposal['new_core_components'])}")
    print(f"[RADICAL RELABEL] Merged new states: {len(proposal['merged_new_states'])}")
    print(f"[RADICAL RELABEL] Changed frames: {changed.size}")
    if saved_dataset is not None:
        print(f"[RADICAL RELABEL] Saved dataset: {saved_dataset}")
    if saved_plots:
        print("[RADICAL RELABEL] Saved plots:")
        for path in saved_plots:
            print(f"  {path}")
    print(f"[RADICAL RELABEL] Summary: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Apply radical confidence/kNN-graph relabeling.")
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
    run_radical_relabel(
        dataset_path=dataset_path,
        model_path=model_path,
        config=cfg,
        device=str(cfg.get("device", "cuda:0")),
        batch_size=int(cfg.get("batch_size", 65536)),
        dataset_stride=int(cfg.get("dataset_stride", 1)),
    )


if __name__ == "__main__":
    main()

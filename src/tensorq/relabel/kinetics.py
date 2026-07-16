from __future__ import annotations

import numpy as np

from .config_utils import _relabel_cfg
from .density import _select_weighted_density_core
from .entropy import _positive_lag_list
from .kinetic_groups import analyze_basin_kinetic_groups
from .knn import _knn_component_labels
from .lag_pair_utils import build_lag_pairs
from .settings import analysis_settings

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

def _local_label_lookup(labels, active_labels):
    labels = np.asarray(labels, dtype=np.int64)
    active_labels = np.sort(np.asarray(active_labels, dtype=np.int64))
    if active_labels.size == 0:
        return active_labels, np.zeros(0, dtype=np.int64)
    max_label = int(max(np.max(labels[labels >= 0]) if np.any(labels >= 0) else 0, active_labels[-1]))
    lookup = np.full(max_label + 1, -1, dtype=np.int64)
    lookup[active_labels] = np.arange(active_labels.size, dtype=np.int64)
    return active_labels, lookup

def _new_label_transition_matrices(labels, new_labels, lag_pairs, lag_list, weights):
    labels = np.asarray(labels, dtype=np.int64)
    new_labels, label_lookup = _local_label_lookup(labels, new_labels)
    n_labels = int(new_labels.size)
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    matrices = {}
    count_matrices = {}
    if n_labels == 0:
        for lag in lag_list:
            matrices[int(lag)] = np.zeros((0, 0), dtype=np.float64)
            count_matrices[int(lag)] = np.zeros((0, 0), dtype=np.int64)
        return matrices, count_matrices

    for lag in lag_list:
        idx_t, idx_tau = lag_pairs[int(lag)]
        start_labels = labels[idx_t]
        end_labels = labels[idx_tau]
        valid = (
            (start_labels >= 0)
            & (end_labels >= 0)
            & (start_labels < label_lookup.size)
            & (end_labels < label_lookup.size)
        )

        matrix = np.zeros((n_labels, n_labels), dtype=np.float64)
        counts = np.zeros((n_labels, n_labels), dtype=np.int64)
        if np.any(valid):
            starts = label_lookup[start_labels[valid]]
            ends = label_lookup[end_labels[valid]]
            valid_local = (starts >= 0) & (ends >= 0)
            starts = starts[valid_local]
            ends = ends[valid_local]
            pair_weights = (
                np.ones(starts.size, dtype=np.float64)
                if weights_arr is None
                else weights_arr[idx_t[valid][valid_local]]
            )
            if starts.size:
                flat = starts * n_labels + ends
                matrix = np.bincount(
                    flat,
                    weights=pair_weights,
                    minlength=n_labels * n_labels,
                ).reshape(n_labels, n_labels)
                counts = np.bincount(
                    flat,
                    minlength=n_labels * n_labels,
                ).reshape(n_labels, n_labels).astype(np.int64, copy=False)

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
        valid_starts = idx_t[valid]
        starts = component_by_frame[valid_starts]
        ends = component_by_frame[idx_tau[valid]]
        valid_components = (starts >= 0) & (ends >= 0)
        if not np.any(valid_components):
            continue
        starts = starts[valid_components]
        ends = ends[valid_components]
        start_frames = valid_starts[valid_components]
        pair_weights = (
            np.ones(starts.size, dtype=np.float64)
            if weights_arr is None
            else np.clip(weights_arr[start_frames], 0.0, np.inf)
        )

        flat = starts * n_components + ends
        weighted_counts = np.bincount(
            flat,
            weights=pair_weights,
            minlength=n_components * n_components,
        ).reshape(n_components, n_components)
        raw_counts = np.bincount(
            flat,
            minlength=n_components * n_components,
        ).reshape(n_components, n_components).astype(np.int64, copy=False)

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

def _iteratively_merge_kinetically_duplicate_labels(
    labels,
    original_labels,
    trajectory_index,
    frame_index,
    weights,
    config,
    protected_new_labels=None,
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
    protected_new_labels = set(
        int(label)
        for label in (protected_new_labels or [])
        if int(label) >= 0
    )
    protect_split_existing = bool(
        relabel_cfg.get("protect_split_existing_labels_from_merge", True)
    )
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
                if protect_split_existing and (i_original != j_original):
                    new_label = int(label_j) if i_original else int(label_i)
                    if new_label in protected_new_labels:
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
    split_knn_cfg = dict(relabel_cfg)
    for source_key, target_key in [
        ("final_check_split_knn_backend", "knn_backend"),
        ("final_check_split_knn_device", "knn_device"),
        ("final_check_split_torch_knn_auto_max_pairs", "torch_knn_auto_max_pairs"),
        ("final_check_split_torch_knn_query_batch", "torch_knn_query_batch"),
        ("final_check_split_torch_knn_reference_batch", "torch_knn_reference_batch"),
        ("final_check_split_torch_knn_dtype", "torch_knn_dtype"),
    ]:
        if source_key in relabel_cfg:
            split_knn_cfg[target_key] = relabel_cfg[source_key]
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

        n_components, component_labels, knn_meta = _knn_component_labels(z, frame_idx, k, split_knn_cfg)
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
                    np.isfinite(weighted_fraction_of_state)
                    and weighted_fraction_of_state < min_split_weight_fraction
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
                "min_split_core_weight_fraction_basis": "state",
                "action": action,
            })

    return new_state, state_rows, assignment_rows, group_labels, reshape_core_mask, reshape_shell_mask, next_label

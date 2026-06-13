from __future__ import annotations

import numpy as np

from .settings import analysis_settings


def _cfg(config):
    return config.get("basin_kinetic_groups", {})


def _as_lag_list(lag_pairs, config):
    kg_cfg = _cfg(config)
    settings = analysis_settings(config)
    lag_list = kg_cfg.get("lag_list", None)
    if lag_list is None:
        lag_list = settings["lag_list"]
    if isinstance(lag_list, (int, float)):
        lag_list = [lag_list]
    out = []
    for lag in lag_list:
        lag = int(lag)
        if lag > 0 and lag in lag_pairs:
            out.append(lag)
    return out


def _entropy_norm(q_values):
    eps = 1e-12
    q = np.clip(np.asarray(q_values, dtype=np.float64), eps, 1.0)
    entropy = -np.sum(q * np.log(q), axis=1)
    denom = np.log(q.shape[1]) if q.shape[1] > 1 else 1.0
    return entropy / denom


def _standardize_features(features):
    x = np.asarray(features, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std


def high_confidence_basin_core(q_values, state_labels, state, config, q_argmax=None):
    kg_cfg = _cfg(config)
    settings = analysis_settings(config)
    cutoff = float(
        kg_cfg.get(
            "q_core_cutoff",
            settings["core_cutoff"],
        )
    )
    mode = str(kg_cfg.get("q_core_mode", "own_high")).lower()
    use_argmax = bool(kg_cfg.get("require_q_argmax", mode not in {"own_low", "low"}))

    state_labels = np.asarray(state_labels, dtype=np.int64)
    q_values = np.asarray(q_values, dtype=np.float64)
    mask = state_labels == int(state)
    if int(state) < 0 or int(state) >= q_values.shape[1]:
        return np.zeros_like(mask, dtype=bool)

    q_own = q_values[:, int(state)]
    if mode in {"own_low", "low", "qi_low", "q_i_low"}:
        mask &= q_own <= cutoff
    else:
        mask &= q_own >= cutoff
        if use_argmax:
            if q_argmax is None:
                q_argmax = np.argmax(q_values, axis=1)
            mask &= np.asarray(q_argmax, dtype=np.int64) == int(state)
    return mask


def _choose_microstate_count(n_core, kg_cfg):
    if n_core <= 1:
        return 0
    min_micro = int(kg_cfg.get("min_microstates", 8))
    max_micro = int(kg_cfg.get("max_microstates", 100))
    target_frames = int(kg_cfg.get("target_frames_per_microstate", 500))
    min_frames = int(kg_cfg.get("min_frames_per_microstate", 20))
    if n_core < max(2 * min_frames, 2):
        return 0
    n_micro = int(np.ceil(n_core / max(1, target_frames)))
    n_micro = min(max(n_micro, min_micro), max_micro, n_core // max(1, min_frames))
    return int(max(2, n_micro))


def _fit_microstates(features, core_idx, n_micro, kg_cfg, random_seed):
    from sklearn.cluster import MiniBatchKMeans

    max_fit = int(kg_cfg.get("max_core_frames", kg_cfg.get("max_fit_frames", 200000)))
    fit_idx = core_idx
    sampled = False
    if max_fit > 0 and core_idx.size > max_fit:
        rng = np.random.default_rng(int(random_seed))
        fit_idx = np.sort(rng.choice(core_idx, size=max_fit, replace=False))
        sampled = True

    z = _standardize_features(features)
    model = MiniBatchKMeans(
        n_clusters=int(n_micro),
        random_state=int(random_seed),
        batch_size=int(kg_cfg.get("microstate_batch_size", 8192)),
        n_init=int(kg_cfg.get("microstate_n_init", 5)),
        reassignment_ratio=0.0,
    )
    model.fit(z[fit_idx])
    micro_labels = model.predict(z[core_idx]).astype(np.int64, copy=False)
    return micro_labels, sampled


def _transition_matrix(micro_by_frame, lag_pairs, lag, n_micro, weights=None):
    idx_t, idx_tau = lag_pairs[int(lag)]
    start_micro = micro_by_frame[idx_t]
    end_micro = micro_by_frame[idx_tau]
    valid = (start_micro >= 0) & (end_micro >= 0)
    counts = np.zeros((n_micro, n_micro), dtype=np.float64)
    raw_counts = np.zeros((n_micro, n_micro), dtype=np.int64)
    if np.any(valid):
        starts = start_micro[valid]
        ends = end_micro[valid]
        pair_weights = np.ones(starts.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)[idx_t[valid]]
        np.add.at(counts, (starts, ends), pair_weights)
        np.add.at(raw_counts, (starts, ends), 1)
    row_sums = np.sum(counts, axis=1)
    p = np.zeros_like(counts)
    nonzero = row_sums > 0.0
    p[nonzero] = counts[nonzero] / row_sums[nonzero, None]
    p[~nonzero, ~nonzero] = 1.0
    return p, counts, raw_counts, int(np.sum(raw_counts))


def _sorted_eigensystem(matrix):
    values, vectors = np.linalg.eig(matrix)
    order = np.argsort(-np.abs(values))
    values = np.real(values[order])
    vectors = np.real(vectors[:, order])
    return values, vectors


def _suggest_group_count(eigenvalues, n_micro, kg_cfg):
    max_groups = min(int(kg_cfg.get("max_macro_groups", 6)), n_micro)
    min_groups = int(kg_cfg.get("min_macro_groups", 1))
    slow_min = float(kg_cfg.get("min_slow_eigenvalue", 0.80))
    eigengap_min = float(kg_cfg.get("min_eigengap", 0.05))
    if max_groups < 2 or eigenvalues.size < 2:
        return 1, 0.0, "no_split"

    vals = np.clip(np.abs(np.real(eigenvalues)), 0.0, 1.0)
    best_k = 1
    best_gap = 0.0
    for k in range(2, max_groups + 1):
        if k >= vals.size:
            break
        gap = float(vals[k - 1] - vals[k])
        if vals[k - 1] >= slow_min and gap > best_gap:
            best_k = k
            best_gap = gap

    if best_k < max(2, min_groups) or best_gap < eigengap_min:
        if bool(kg_cfg.get("split_on_slow_mode_without_eigengap", True)) and vals.size > 1 and vals[1] >= slow_min:
            return 2, best_gap, "weak_slow_mode"
        return 1, best_gap, "no_split"
    confidence = "stable" if best_gap >= float(kg_cfg.get("strong_eigengap", 0.12)) else "weak"
    return int(best_k), float(best_gap), confidence


def _macro_assignments(eigenvectors, n_groups, random_seed):
    if n_groups <= 1:
        return np.zeros(eigenvectors.shape[0], dtype=np.int64)

    from sklearn.cluster import KMeans

    emb = np.asarray(eigenvectors[:, 1:n_groups], dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1)
    valid = norms > 1e-12
    emb[valid] = emb[valid] / norms[valid, None]
    km = KMeans(n_clusters=int(n_groups), random_state=int(random_seed), n_init=20)
    return km.fit_predict(emb).astype(np.int64, copy=False)


def _feature_macro_assignments(features, core_idx, n_groups, random_seed):
    if n_groups <= 1:
        return np.zeros(core_idx.size, dtype=np.int64)

    from sklearn.cluster import KMeans

    z = _standardize_features(features)
    if np.unique(z[core_idx], axis=0).shape[0] < n_groups:
        return np.zeros(core_idx.size, dtype=np.int64)
    km = KMeans(n_clusters=int(n_groups), random_state=int(random_seed), n_init=20)
    return km.fit_predict(z[core_idx]).astype(np.int64, copy=False)


def _renumber_by_population(labels, micro_counts):
    labels = np.asarray(labels, dtype=np.int64)
    unique = sorted(int(x) for x in np.unique(labels[labels >= 0]))
    populations = []
    for label in unique:
        populations.append((label, float(np.sum(micro_counts[labels == label]))))
    ordered = [label for label, _ in sorted(populations, key=lambda item: (-item[1], item[0]))]
    mapping = {old: new for new, old in enumerate(ordered)}
    out = labels.copy()
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def _renumber_frame_labels_by_population(labels):
    labels = np.asarray(labels, dtype=np.int64)
    unique = sorted(int(x) for x in np.unique(labels[labels >= 0]))
    populations = []
    for label in unique:
        populations.append((label, int(np.sum(labels == label))))
    ordered = [label for label, _ in sorted(populations, key=lambda item: (-item[1], item[0]))]
    mapping = {old: new for new, old in enumerate(ordered)}
    out = labels.copy()
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def _macro_transition_rows(macro_by_frame, lag_pairs, lag_list, n_groups, weights=None):
    matrices = {}
    raw_counts = {}
    for lag in lag_list:
        idx_t, idx_tau = lag_pairs[int(lag)]
        start = macro_by_frame[idx_t]
        end = macro_by_frame[idx_tau]
        valid = (start >= 0) & (end >= 0)
        counts = np.zeros((n_groups, n_groups), dtype=np.float64)
        counts_int = np.zeros((n_groups, n_groups), dtype=np.int64)
        if np.any(valid):
            starts = start[valid]
            ends = end[valid]
            pair_weights = np.ones(starts.size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)[idx_t[valid]]
            np.add.at(counts, (starts, ends), pair_weights)
            np.add.at(counts_int, (starts, ends), 1)
        row_sums = np.sum(counts, axis=1)
        probs = np.full_like(counts, np.nan)
        nonzero = row_sums > 0.0
        probs[nonzero] = counts[nonzero] / row_sums[nonzero, None]
        matrices[int(lag)] = probs
        raw_counts[int(lag)] = counts_int
    return matrices, raw_counts


def _format_eigenvalues(values, max_count=8):
    vals = np.clip(np.abs(np.real(values[:max_count])), 0.0, 1.0)
    return ";".join(f"{value:.4f}" for value in vals)


def analyze_basin_kinetic_groups(state_labels, q_values, lag_pairs, config, weights=None, n_states=None, features=None):
    """Run local spectral metastability analysis inside each current label.

    The workflow is:
    high-confidence q-core -> feature-space microstates -> trajectory-safe
    transition matrix -> eigenvalue/eigengap group count -> slow-eigenvector
    macrostate assignment.
    """

    settings = analysis_settings(config)
    kg_cfg = dict(_cfg(config))
    kg_cfg.setdefault("min_eigengap", settings["eigengap"])
    kg_cfg.setdefault("max_macro_groups", settings["max_groups"])
    kg_cfg.setdefault("min_slow_eigenvalue", settings["min_slow_eigenvalue"])
    if not bool(kg_cfg.get("enabled", True)):
        n_frames = int(np.asarray(state_labels).shape[0])
        return [], [], np.full(n_frames, -1, dtype=np.int64)

    state_labels = np.asarray(state_labels, dtype=np.int64)
    q_values = np.asarray(q_values, dtype=np.float64)
    features = q_values if features is None else np.asarray(features, dtype=np.float64)
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    lag_list = _as_lag_list(lag_pairs, config)
    n_frames = int(state_labels.shape[0])
    n_states = int(q_values.shape[1] if n_states is None else n_states)
    min_size = int(kg_cfg.get("min_group_size", settings["min_group_size"]))
    min_weight = float(kg_cfg.get("min_group_weight", 0.0))
    min_pairs = int(kg_cfg.get("min_valid_pairs", settings["min_count"]))
    random_seed = int(kg_cfg.get("random_seed", settings["random_seed"]))
    analysis_lag_raw = kg_cfg.get("analysis_lag", None)
    analysis_lag = int(lag_list[-1] if analysis_lag_raw is None and lag_list else (analysis_lag_raw or 1))
    if analysis_lag not in lag_pairs and lag_list:
        analysis_lag = int(lag_list[-1])

    q_argmax = np.argmax(q_values, axis=1).astype(np.int64)
    q_max = np.max(q_values, axis=1)
    entropy_norm = _entropy_norm(q_values)
    group_labels = np.full(n_frames, -1, dtype=np.int64)
    group_rows = []
    state_rows = []
    next_group = 0

    for state in range(n_states):
        state_mask = state_labels == state
        n_state = int(np.sum(state_mask))
        if n_state == 0:
            continue

        core_mask = high_confidence_basin_core(q_values, state_labels, state, config, q_argmax=q_argmax)
        core_idx = np.flatnonzero(core_mask)
        n_core = int(core_idx.size)
        state_weight = float(np.sum(weights_arr[state_mask])) if weights_arr is not None else float(n_state)
        core_weight = float(np.sum(weights_arr[core_idx])) if weights_arr is not None else float(n_core)
        n_micro = _choose_microstate_count(n_core, kg_cfg)

        base_row = {
            "state": int(state),
            "method": "spectral_msm",
            "n_state_frames": n_state,
            "n_core_frames": n_core,
            "weighted_state_population": state_weight,
            "weighted_core_population": core_weight,
            "core_weight_fraction_of_state": float(core_weight / state_weight) if state_weight > 0.0 else np.nan,
            "core_fraction_of_state": float(n_core / max(1, n_state)),
            "analysis_lag": int(analysis_lag),
            "n_microstates": int(n_micro),
            "suggested_groups": 1,
            "n_kinetic_groups": 1,
            "split_confidence": "no_split",
            "eigengap_score": 0.0,
            "eigenvalues": "",
            "n_transition_pairs": 0,
            "microstate_fit_sampled": False,
            "mean_q_own_core": float(np.mean(q_values[core_idx, state])) if n_core else np.nan,
            "mean_q_max_core": float(np.mean(q_max[core_idx])) if n_core else np.nan,
            "mean_entropy_norm_core": float(np.mean(entropy_norm[core_idx])) if n_core else np.nan,
        }
        if n_micro < 2 or not lag_list:
            state_rows.append(base_row)
            continue

        micro_labels_core, sampled = _fit_microstates(
            features,
            core_idx,
            n_micro,
            kg_cfg,
            random_seed=random_seed + int(state),
        )
        micro_by_frame = np.full(n_frames, -1, dtype=np.int64)
        micro_by_frame[core_idx] = micro_labels_core
        micro_counts = np.bincount(micro_labels_core, minlength=n_micro).astype(np.int64, copy=False)

        transition, _, _, n_transition_pairs = _transition_matrix(
            micro_by_frame,
            lag_pairs,
            analysis_lag,
            n_micro,
            weights=weights_arr,
        )
        if n_transition_pairs < min_pairs:
            base_row["n_transition_pairs"] = int(n_transition_pairs)
            base_row["microstate_fit_sampled"] = bool(sampled)
            state_rows.append(base_row)
            continue

        eigenvalues, eigenvectors = _sorted_eigensystem(transition)
        n_groups, eigengap_score, split_confidence = _suggest_group_count(eigenvalues, n_micro, kg_cfg)
        macro_by_micro = _macro_assignments(eigenvectors, n_groups, random_seed + int(state))
        macro_by_micro = _renumber_by_population(macro_by_micro, micro_counts)
        macro_labels_core = macro_by_micro[micro_labels_core]
        if n_groups > 1 and np.unique(macro_labels_core).size < n_groups:
            macro_labels_core = _feature_macro_assignments(
                features,
                core_idx,
                n_groups,
                random_seed=random_seed + int(state),
            )
            macro_labels_core = _renumber_frame_labels_by_population(macro_labels_core)

        macro_by_frame = np.full(n_frames, -1, dtype=np.int64)
        macro_by_frame[core_idx] = macro_labels_core
        macro_matrices, macro_counts = _macro_transition_rows(
            macro_by_frame,
            lag_pairs,
            lag_list,
            n_groups,
            weights=weights_arr,
        )

        base_row.update({
            "suggested_groups": int(n_groups),
            "n_kinetic_groups": int(n_groups),
            "split_confidence": split_confidence,
            "eigengap_score": float(eigengap_score),
            "eigenvalues": _format_eigenvalues(eigenvalues),
            "n_transition_pairs": int(n_transition_pairs),
            "microstate_fit_sampled": bool(sampled),
        })
        state_rows.append(base_row)

        for local_group in range(n_groups):
            frame_idx = core_idx[macro_labels_core == local_group]
            if frame_idx.size < min_size:
                continue
            group_weight = float(np.sum(weights_arr[frame_idx])) if weights_arr is not None else float(frame_idx.size)
            if group_weight < min_weight:
                continue
            weighted_fraction_of_state = float(group_weight / state_weight) if state_weight > 0.0 else np.nan
            group_id = next_group
            next_group += 1
            group_labels[frame_idx] = group_id
            row = {
                "state": int(state),
                "kinetic_group": int(group_id),
                "local_group": int(local_group),
                "rank_within_state": int(local_group),
                "split_recommended": bool(n_groups >= 2 and split_confidence != "no_split"),
                "split_confidence": split_confidence,
                "eigengap_score": float(eigengap_score),
                "analysis_lag": int(analysis_lag),
                "n_frames": int(frame_idx.size),
                "weighted_population": group_weight,
                "weighted_fraction_of_state": weighted_fraction_of_state,
                "fraction_of_state": float(frame_idx.size / max(1, n_state)),
                "fraction_of_core": float(frame_idx.size / max(1, n_core)),
                "mean_q_own": float(np.mean(q_values[frame_idx, state])),
                "mean_q_max": float(np.mean(q_max[frame_idx])),
                "mean_entropy_norm": float(np.mean(entropy_norm[frame_idx])),
                "fraction_q_argmax_own": float(np.mean(q_argmax[frame_idx] == state)),
            }
            for lag in lag_list:
                probs = macro_matrices[int(lag)]
                counts = macro_counts[int(lag)]
                row[f"n_pairs_lag_{lag}"] = int(np.sum(counts[local_group]))
                row[f"retention_lag_{lag}"] = (
                    float(probs[local_group, local_group])
                    if np.isfinite(probs[local_group, local_group])
                    else np.nan
                )
                for other in range(n_groups):
                    if other != local_group:
                        row[f"p_to_group_{other}_lag_{lag}"] = (
                            float(probs[local_group, other])
                            if np.isfinite(probs[local_group, other])
                            else np.nan
                        )
            group_rows.append(row)

    return state_rows, group_rows, group_labels

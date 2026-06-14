from __future__ import annotations

import numpy as np

from .lag_pair_utils import build_lag_pairs
from .settings import analysis_settings
from .config_utils import _relabel_cfg


def _remove_inconsistent_states(
    new_state,
    state_labels,
    q_values,
    label_consistency,
    entropy_norm,
    config,
):
    settings = analysis_settings(config)
    relabel_cfg = _relabel_cfg(config)
    removed_mask = np.zeros(state_labels.shape[0], dtype=bool)
    removed_states = []
    removed_state_ids = []
    if not bool(relabel_cfg.get("remove_inconsistent_states", True)):
        return new_state, removed_mask, removed_states, removed_state_ids

    q_label_cutoff = float(settings["q_cutoff"])
    entropy_cutoff = float(settings["entropy_cutoff"])
    remove_cutoff = float(relabel_cfg.get("remove_problem_fraction_cutoff", 0.9))
    min_stable_fraction = float(
        relabel_cfg.get("remove_min_stable_fraction", settings["persistent_fraction"])
    )
    n_states = int(q_values.shape[1])

    for state in range(n_states):
        mask = state_labels == state
        n_state = int(np.sum(mask))
        if n_state == 0:
            continue

        problem = mask & (
            (label_consistency < q_label_cutoff)
            | (entropy_norm >= entropy_cutoff)
        )
        stable = mask & (
            (label_consistency >= q_label_cutoff)
            & (entropy_norm < entropy_cutoff)
        )
        problem_fraction = float(np.sum(problem) / max(1, n_state))
        stable_fraction = float(np.sum(stable) / max(1, n_state))
        if problem_fraction < remove_cutoff and stable_fraction >= min_stable_fraction:
            continue

        new_state[mask] = -1
        removed_mask |= mask
        removed_state_ids.append(int(state))
        reason = (
            "problem_fraction_above_cutoff"
            if problem_fraction >= remove_cutoff
            else "stable_fraction_below_cutoff"
        )
        removed_states.append({
            "removed_state": int(state),
            "n_frames": n_state,
            "n_problem_frames": int(np.sum(problem)),
            "n_stable_frames": int(np.sum(stable)),
            "problem_fraction": problem_fraction,
            "stable_fraction": stable_fraction,
            "problem_fraction_cutoff": remove_cutoff,
            "min_stable_fraction": min_stable_fraction,
            "mean_q_own": float(np.mean(q_values[mask, state])),
            "mean_entropy_norm": float(np.mean(entropy_norm[mask])),
            "reason": reason,
        })

    return new_state, removed_mask, removed_states, removed_state_ids


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

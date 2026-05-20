from __future__ import annotations

import numpy as np


def build_lag_pairs(trajectory_index, frame_index, lag_list):
    trajectory_index = np.asarray(trajectory_index, dtype=np.int64)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    n_frames = int(trajectory_index.shape[0])

    key_dtype = np.dtype([("traj", np.int64), ("frame", np.int64)])
    keys = np.empty(n_frames, dtype=key_dtype)
    keys["traj"] = trajectory_index
    keys["frame"] = frame_index

    order = np.argsort(keys, order=("traj", "frame"))
    sorted_keys = keys[order]

    result = {}
    for lag in lag_list:
        lag = int(lag)
        targets = np.empty(n_frames, dtype=key_dtype)
        targets["traj"] = trajectory_index
        targets["frame"] = frame_index + lag

        pos = np.searchsorted(sorted_keys, targets)
        in_bounds = pos < n_frames
        matched = np.zeros(n_frames, dtype=bool)
        matched[in_bounds] = sorted_keys[pos[in_bounds]] == targets[in_bounds]

        idx_t = np.flatnonzero(matched).astype(np.int64, copy=False)
        idx_tau = order[pos[matched]].astype(np.int64, copy=False)
        result[lag] = (idx_t, idx_tau)
    return result


def _infer_n_labels(labels, n_labels=None):
    if n_labels is not None:
        return int(n_labels)
    valid = labels >= 0
    if not np.any(valid):
        return 0
    return int(np.max(labels[valid])) + 1


def label_q_means(labels, q_values, weights=None, n_labels=None):
    labels = np.asarray(labels, dtype=np.int64)
    q_values = np.asarray(q_values, dtype=np.float64)
    n_labels = _infer_n_labels(labels, n_labels)
    n_states = int(q_values.shape[1])

    means = np.full((n_labels, n_states), np.nan, dtype=np.float64)
    counts = np.zeros(n_labels, dtype=np.int64)
    weight_sums = np.zeros(n_labels, dtype=np.float64)
    if n_labels == 0:
        return means, counts, weight_sums

    valid = (labels >= 0) & (labels < n_labels)
    if not np.any(valid):
        return means, counts, weight_sums

    valid_labels = labels[valid]
    counts = np.bincount(valid_labels, minlength=n_labels).astype(np.int64, copy=False)

    if weights is None:
        sample_weights = np.ones(valid_labels.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)[valid]

    weight_sums = np.bincount(valid_labels, weights=sample_weights, minlength=n_labels)
    q_sums = np.zeros((n_labels, n_states), dtype=np.float64)
    for state in range(n_states):
        q_sums[:, state] = np.bincount(
            valid_labels,
            weights=sample_weights * q_values[valid, state],
            minlength=n_labels,
        )

    nonzero = weight_sums > 0.0
    means[nonzero] = q_sums[nonzero] / weight_sums[nonzero, None]
    return means, counts, weight_sums


def summarize_label_kinetics(
    labels,
    q_values,
    lag_pairs,
    lag_list=None,
    weights=None,
    min_valid_pairs=50,
    n_labels=None,
):
    labels = np.asarray(labels, dtype=np.int64)
    q_values = np.asarray(q_values, dtype=np.float64)
    n_labels = _infer_n_labels(labels, n_labels)
    lag_list = sorted(lag_pairs) if lag_list is None else [int(lag) for lag in lag_list]
    min_valid_pairs = int(min_valid_pairs)
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    q_means, _, _ = label_q_means(labels, q_values, weights=weights_arr, n_labels=n_labels)

    retention = {}
    q_autocorr = {}
    valid_pair_counts = {}

    for lag in lag_list:
        idx_t, idx_tau = lag_pairs[lag]
        ret = np.full(n_labels, np.nan, dtype=np.float64)
        ac = np.full(n_labels, np.nan, dtype=np.float64)
        counts = np.zeros(n_labels, dtype=np.int64)

        if n_labels == 0 or len(idx_t) == 0:
            retention[lag] = ret
            q_autocorr[lag] = ac
            valid_pair_counts[lag] = counts
            continue

        start_labels_all = labels[idx_t]
        valid = (start_labels_all >= 0) & (start_labels_all < n_labels)
        if not np.any(valid):
            retention[lag] = ret
            q_autocorr[lag] = ac
            valid_pair_counts[lag] = counts
            continue

        starts = idx_t[valid]
        ends = idx_tau[valid]
        start_labels = start_labels_all[valid]
        end_labels = labels[ends]
        if weights_arr is None:
            pair_weights = np.ones(start_labels.shape[0], dtype=np.float64)
        else:
            pair_weights = weights_arr[starts]

        counts = np.bincount(start_labels, minlength=n_labels).astype(np.int64, copy=False)
        weight_sums = np.bincount(start_labels, weights=pair_weights, minlength=n_labels)

        stayed = end_labels == start_labels
        stayed_sums = np.bincount(
            start_labels,
            weights=pair_weights * stayed.astype(np.float64),
            minlength=n_labels,
        )

        enough = (counts >= min_valid_pairs) & (weight_sums > 0.0)
        ret[enough] = stayed_sums[enough] / weight_sums[enough]

        means = q_means[start_labels]
        dq0 = q_values[starts] - means
        dq1 = q_values[ends] - means
        num_vals = np.sum(dq0 * dq1, axis=1)
        den_vals = np.sum(dq0 * dq0, axis=1)

        num_sums = np.bincount(start_labels, weights=pair_weights * num_vals, minlength=n_labels)
        den_sums = np.bincount(start_labels, weights=pair_weights * den_vals, minlength=n_labels)
        avg_num = np.zeros(n_labels, dtype=np.float64)
        avg_den = np.zeros(n_labels, dtype=np.float64)
        nonzero = weight_sums > 0.0
        avg_num[nonzero] = num_sums[nonzero] / weight_sums[nonzero]
        avg_den[nonzero] = den_sums[nonzero] / weight_sums[nonzero]
        ac[enough] = avg_num[enough] / (avg_den[enough] + 1e-12)

        retention[lag] = ret
        q_autocorr[lag] = ac
        valid_pair_counts[lag] = counts

    return {
        "retention": retention,
        "q_autocorr": q_autocorr,
        "valid_pair_counts": valid_pair_counts,
    }


def estimate_retention(region_mask, lag_pairs, lag, weights=None, min_valid_pairs=50):
    idx_t, idx_tau = lag_pairs[lag]
    valid = region_mask[idx_t]
    n_valid = np.sum(valid)
    if n_valid < min_valid_pairs:
        return np.nan
    values = region_mask[idx_tau[valid]]
    if weights is not None:
        return float(np.average(values, weights=weights[idx_t[valid]]))
    return float(np.mean(values))

def estimate_q_autocorrelation(region_mask, q_values, lag_pairs, lag, weights=None, min_valid_pairs=50):
    eps = 1e-12
    idx_t, idx_tau = lag_pairs[lag]
    valid = region_mask[idx_t]
    n_valid = np.sum(valid)
    if n_valid < min_valid_pairs:
        return np.nan
    region_frames = np.where(region_mask)[0]
    if weights is not None:
        q_mean = np.average(q_values[region_frames], weights=weights[region_frames], axis=0)
    else:
        q_mean = np.mean(q_values[region_frames], axis=0)
    idx_t_valid = idx_t[valid]
    idx_tau_valid = idx_tau[valid]
    dq0 = q_values[idx_t_valid] - q_mean
    dq1 = q_values[idx_tau_valid] - q_mean
    num_vals = np.sum(dq0 * dq1, axis=1)
    den_vals = np.sum(dq0 * dq0, axis=1)
    if weights is not None:
        w = weights[idx_t_valid]
        numerator = np.average(num_vals, weights=w)
        denominator = np.average(den_vals, weights=w)
    else:
        numerator = np.mean(num_vals)
        denominator = np.mean(den_vals)
    return float(numerator / (denominator + eps))

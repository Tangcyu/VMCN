from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .selection import normalize_weights
from .shooting import GradientPath, reparameterize_path


@dataclass(frozen=True)
class PathCluster:
    label: int
    member_indices: np.ndarray
    weights: np.ndarray
    center_path: np.ndarray
    medoid_index: int
    medoid_path: np.ndarray
    total_weight: float


def path_array(paths: list[GradientPath] | np.ndarray, *, num_images: int | None = None) -> np.ndarray:
    if isinstance(paths, np.ndarray):
        array = np.asarray(paths, dtype=np.float64)
    else:
        array = np.asarray([p.path for p in paths], dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("paths must have shape (n_paths, n_images, n_dim).")
    if num_images is not None and array.shape[1] != int(num_images):
        array = np.asarray([reparameterize_path(path, int(num_images)) for path in array], dtype=np.float64)
    return array


def path_weights(paths: list[GradientPath] | np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is not None:
        out = np.asarray(weights, dtype=np.float64)
    elif isinstance(paths, np.ndarray):
        out = np.ones(paths.shape[0], dtype=np.float64)
    else:
        out = np.asarray([p.weight for p in paths], dtype=np.float64)
    if out.ndim != 1:
        raise ValueError("path weights must be one-dimensional.")
    if np.any(out < 0.0):
        raise ValueError("path weights must be nonnegative.")
    if np.sum(out) <= 0.0:
        raise ValueError("path weights must have positive total mass.")
    return out


def _normalize_periods(periods: Sequence[float | None] | None, ndim: int) -> list[float | None]:
    if periods is None:
        return [None] * int(ndim)
    out = [None if item is None else float(item) for item in periods]
    if len(out) != int(ndim):
        raise ValueError(f"periods must have length {int(ndim)}.")
    return out


def _minimum_image_delta(diff: np.ndarray, periods: Sequence[float | None]) -> np.ndarray:
    out = np.asarray(diff, dtype=np.float64).copy()
    for dim, period in enumerate(periods):
        if period is not None and float(period) > 0.0:
            p = float(period)
            out[..., dim] = ((out[..., dim] + 0.5 * p) % p) - 0.5 * p
    return out


def pairwise_rmsd_matrix(
    paths: list[GradientPath] | np.ndarray,
    *,
    num_images: int | None = None,
    periods: Sequence[float | None] | None = None,
) -> np.ndarray:
    array = path_array(paths, num_images=num_images)
    period_list = _normalize_periods(periods, array.shape[2])
    n_paths = array.shape[0]
    dist = np.zeros((n_paths, n_paths), dtype=np.float64)
    for i in range(n_paths):
        diff = array[i + 1 :] - array[i]
        if diff.size == 0:
            continue
        diff = _minimum_image_delta(diff, period_list)
        values = np.sqrt(np.mean(diff * diff, axis=(1, 2)))
        dist[i, i + 1 :] = values
        dist[i + 1 :, i] = values
    return dist


def _cluster_distance(
    distance_matrix: np.ndarray,
    weights: np.ndarray,
    members_a: list[int],
    members_b: list[int],
) -> float:
    ia = np.asarray(members_a, dtype=np.int64)
    ib = np.asarray(members_b, dtype=np.int64)
    wa = weights[ia]
    wb = weights[ib]
    pair_weights = wa[:, None] * wb[None, :]
    total = float(np.sum(pair_weights))
    if total <= 0.0:
        return float("inf")
    return float(np.sum(distance_matrix[np.ix_(ia, ib)] * pair_weights) / total)


def weighted_center_path(
    paths: np.ndarray,
    weights: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted average pathway, with weights normalized inside the cluster."""

    paths = path_array(paths)
    weights = normalize_weights(np.asarray(weights, dtype=np.float64), allow_zero=True)
    period_list = _normalize_periods(periods, paths.shape[2])
    if not any(period is not None and period > 0.0 for period in period_list):
        return np.average(paths, axis=0, weights=weights)
    ref = paths[0] if reference is None else np.asarray(reference, dtype=np.float64)
    if ref.shape != paths.shape[1:]:
        raise ValueError(f"reference must have shape {paths.shape[1:]}.")
    aligned = ref[None, :, :] + _minimum_image_delta(paths - ref[None, :, :], period_list)
    return np.average(aligned, axis=0, weights=weights)


def _weighted_medoid(
    paths: np.ndarray,
    weights: np.ndarray,
    global_indices: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
) -> int:
    weights = normalize_weights(np.asarray(weights, dtype=np.float64), allow_zero=True)
    local_distance = pairwise_rmsd_matrix(paths, periods=periods)
    score = local_distance @ weights
    return int(global_indices[int(np.argmin(score))])


def cluster_paths(
    paths: list[GradientPath] | np.ndarray,
    *,
    weights: np.ndarray | None = None,
    distance_threshold: float = 0.25,
    num_images: int | None = None,
    periods: Sequence[float | None] | None = None,
) -> tuple[np.ndarray, list[PathCluster], np.ndarray]:
    """
    Cluster pathways using weighted average-link agglomeration.

    Cluster-to-cluster distances are weighted by the Boltzmann path weights.
    The center_path of each output cluster is a weighted mean pathway.
    """

    array = path_array(paths, num_images=num_images)
    period_list = _normalize_periods(periods, array.shape[2])
    if array.shape[0] == 0:
        raise ValueError("Need at least one path to cluster.")
    weights_arr = path_weights(paths, weights)
    if weights_arr.shape[0] != array.shape[0]:
        raise ValueError("weights length must match number of paths.")
    threshold = float(distance_threshold)
    if threshold < 0.0:
        raise ValueError("distance_threshold must be nonnegative.")

    distance_matrix = pairwise_rmsd_matrix(array, periods=period_list)
    clusters: list[list[int]] = [[idx] for idx in range(array.shape[0])]

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                dist = _cluster_distance(distance_matrix, weights_arr, clusters[a], clusters[b])
                if dist < best_distance:
                    best_distance = dist
                    best_pair = (a, b)
        if best_pair is None or best_distance > threshold:
            break
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]

    clusters = sorted(clusters, key=lambda members: (-float(np.sum(weights_arr[members])), min(members)))
    labels = np.zeros(array.shape[0], dtype=np.int64)
    out: list[PathCluster] = []
    for label, members in enumerate(clusters, start=1):
        ids = np.asarray(sorted(members), dtype=np.int64)
        labels[ids] = label
        member_paths = array[ids]
        member_weights = weights_arr[ids]
        medoid = _weighted_medoid(member_paths, member_weights, ids, periods=period_list)
        center = weighted_center_path(member_paths, member_weights, periods=period_list, reference=array[medoid])
        out.append(
            PathCluster(
                label=label,
                member_indices=ids,
                weights=member_weights.copy(),
                center_path=center,
                medoid_index=medoid,
                medoid_path=array[medoid].copy(),
                total_weight=float(np.sum(member_weights)),
            )
        )
    return labels, out, distance_matrix


def cluster_paths_with_linkage(
    paths: list[GradientPath] | np.ndarray,
    *,
    weights: np.ndarray | None = None,
    distance_threshold: float = 0.25,
    num_images: int | None = None,
    periods: Sequence[float | None] | None = None,
) -> tuple[np.ndarray, list[PathCluster], np.ndarray, np.ndarray]:
    """
    Cluster pathways and return a linkage matrix suitable for dendrogram plots.

    The linkage matrix uses the SciPy convention: each row contains
    cluster_a, cluster_b, merge_distance, and merged_member_count. The
    agglomeration itself uses weighted average-link distances.
    """

    array = path_array(paths, num_images=num_images)
    period_list = _normalize_periods(periods, array.shape[2])
    if array.shape[0] == 0:
        raise ValueError("Need at least one path to cluster.")
    weights_arr = path_weights(paths, weights)
    if weights_arr.shape[0] != array.shape[0]:
        raise ValueError("weights length must match number of paths.")
    threshold = float(distance_threshold)
    if threshold < 0.0:
        raise ValueError("distance_threshold must be nonnegative.")

    distance_matrix = pairwise_rmsd_matrix(array, periods=period_list)
    clusters: list[list[int]] = [[idx] for idx in range(array.shape[0])]
    cluster_ids: list[int] = list(range(array.shape[0]))
    next_cluster_id = array.shape[0]
    linkage_rows: list[list[float]] = []

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                dist = _cluster_distance(distance_matrix, weights_arr, clusters[a], clusters[b])
                if dist < best_distance:
                    best_distance = dist
                    best_pair = (a, b)
        if best_pair is None:
            break
        a, b = best_pair
        merged = clusters[a] + clusters[b]
        linkage_rows.append(
            [
                float(cluster_ids[a]),
                float(cluster_ids[b]),
                float(best_distance),
                float(len(merged)),
            ]
        )
        clusters[a] = merged
        cluster_ids[a] = next_cluster_id
        next_cluster_id += 1
        del clusters[b]
        del cluster_ids[b]

    id_to_members: dict[int, list[int]] = {idx: [idx] for idx in range(array.shape[0])}
    threshold_clusters: list[list[int]] = [[idx] for idx in range(array.shape[0])]
    for row_idx, row in enumerate(linkage_rows):
        left = int(row[0])
        right = int(row[1])
        merged = id_to_members[left] + id_to_members[right]
        new_id = array.shape[0] + row_idx
        id_to_members[new_id] = merged
        if float(row[2]) <= threshold:
            threshold_clusters = [members for members in threshold_clusters if members is not id_to_members[left]]
            threshold_clusters = [members for members in threshold_clusters if members is not id_to_members[right]]
            threshold_clusters = [
                members
                for members in threshold_clusters
                if set(members).isdisjoint(id_to_members[left]) and set(members).isdisjoint(id_to_members[right])
            ]
            threshold_clusters.append(merged)

    threshold_clusters = sorted(
        threshold_clusters,
        key=lambda members: (-float(np.sum(weights_arr[members])), min(members)),
    )
    labels = np.ones(array.shape[0], dtype=np.int64)
    out: list[PathCluster] = []
    for label, members in enumerate(threshold_clusters, start=1):
        ids = np.asarray(sorted(members), dtype=np.int64)
        labels[ids] = label
        member_paths = array[ids]
        member_weights = weights_arr[ids]
        medoid = _weighted_medoid(member_paths, member_weights, ids, periods=period_list)
        center = weighted_center_path(member_paths, member_weights, periods=period_list, reference=array[medoid])
        out.append(
            PathCluster(
                label=label,
                member_indices=ids,
                weights=member_weights.copy(),
                center_path=center,
                medoid_index=medoid,
                medoid_path=array[medoid].copy(),
                total_weight=float(np.sum(member_weights)),
            )
        )
    return labels, out, distance_matrix, np.asarray(linkage_rows, dtype=np.float64)

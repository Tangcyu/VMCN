from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import (
    assign_voronoi_cells,
    minimum_image_delta,
    normalize_periods,
    normalize_weights,
    periodic_sincos_embed,
    periodic_sincos_project,
    periodic_sincos_weighted_mean,
    periodic_weighted_mean,
    _use_sincos_geometry,
)


@dataclass(frozen=True)
class PathwayIteration:
    iteration: int
    paths: list[np.ndarray]
    relaxed_paths: list[np.ndarray]
    local_labels: np.ndarray
    local_distances: np.ndarray
    node_labels: np.ndarray
    shared_segments: list[list[tuple[int, int]]]
    exchange_counts: np.ndarray
    n_cells: int
    max_shift: float
    converged: bool


@dataclass(frozen=True)
class IterativePathwayResult:
    paths: list[np.ndarray]
    history: list[PathwayIteration]
    converged: bool


@dataclass(frozen=True)
class PathwayNetwork:
    """Image-level reactive pathway graph.

    adjacency maps (path_idx, image_idx) -> [(path_idx, image_idx, weight), ...].
    Edges include both intra-path (consecutive images) and inter-path (exchange)
    connections.
    """
    adjacency: dict[tuple[int, int], list[tuple[int, int, float]]]
    start_nodes: list[tuple[int, int]]
    end_nodes: list[tuple[int, int]]
    branch_points: list[tuple[int, int]]


class _UnionFind:
    def __init__(self, n_items: int):
        self.parent = list(range(int(n_items)))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def labels(self) -> np.ndarray:
        roots: dict[int, int] = {}
        out = np.empty(len(self.parent), dtype=np.int64)
        for idx in range(len(self.parent)):
            root = self.find(idx)
            if root not in roots:
                roots[root] = len(roots)
            out[idx] = roots[root]
        return out


def _build_offsets(n_images_per_path: Sequence[int]) -> np.ndarray:
    n_per = np.asarray(n_images_per_path, dtype=np.int64)
    offsets = np.empty(len(n_per) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(n_per, out=offsets[1:])
    return offsets


def _path_to_node(path_idx: int, image_idx: int, offsets: np.ndarray) -> int:
    return int(offsets[path_idx]) + int(image_idx)


def _node_to_path(node_idx: int, offsets: np.ndarray) -> tuple[int, int]:
    path_idx = int(np.searchsorted(offsets, node_idx, side="right") - 1)
    image_idx = int(node_idx) - int(offsets[path_idx])
    return path_idx, image_idx


def _path_arc(path: np.ndarray, periods: Sequence[float | None] | None = None) -> np.ndarray:
    if path.shape[0] == 1:
        return np.asarray([0.0], dtype=np.float64)
    delta = minimum_image_delta(np.diff(path, axis=0), periods)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(delta, axis=1))])


def _reparameterize_values(values: np.ndarray, arc: np.ndarray, num_images: int) -> np.ndarray:
    if values.shape[0] == 1 or float(arc[-1]) <= 0.0:
        return np.repeat(values[:1], int(num_images), axis=0)
    target = np.linspace(0.0, float(arc[-1]), int(num_images))
    columns = [np.interp(target, arc, values[:, dim]) for dim in range(values.shape[1])]
    return np.stack(columns, axis=1)


def _normalize_wrap_bounds(
    wrap_bounds: Sequence[Sequence[float] | None] | None,
    periods: Sequence[float | None] | None,
    ndim: int,
) -> list[tuple[float, float] | None]:
    if wrap_bounds is not None:
        out = []
        if len(wrap_bounds) != int(ndim):
            raise ValueError(f"wrap_bounds must have length {int(ndim)}.")
        for item in wrap_bounds:
            if item is None:
                out.append(None)
            else:
                if len(item) != 2:
                    raise ValueError("Each wrap_bounds entry must be null or [lower, upper].")
                out.append((float(item[0]), float(item[1])))
        return out
    period_list = normalize_periods(periods, ndim)
    return [None if period is None else (-0.5 * float(period), 0.5 * float(period)) for period in period_list]


def wrap_periodic_points(
    values: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    wrap_bounds: Sequence[Sequence[float] | None] | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    bounds = _normalize_wrap_bounds(wrap_bounds, periods, arr.shape[-1])
    for dim, bound in enumerate(bounds):
        if bound is None:
            continue
        lower, upper = bound
        width = upper - lower
        if width <= 0.0:
            raise ValueError("wrap bound upper must be greater than lower.")
        arr[..., dim] = lower + np.mod(arr[..., dim] - lower, width)
    return arr


def reparameterize_path_periodic(
    path: np.ndarray,
    num_images: int,
    *,
    periods: Sequence[float | None] | None = None,
) -> np.ndarray:
    path_arr = np.asarray(path, dtype=np.float64)
    if path_arr.ndim != 2:
        raise ValueError("path must have shape (n_images, n_dim).")
    n = int(num_images)
    if n < 2:
        raise ValueError("num_images must be at least 2.")
    if path_arr.shape[0] == 1:
        return np.repeat(path_arr, n, axis=0)
    unwrapped = unwrap_path(path_arr, periods=periods)
    segment = np.linalg.norm(np.diff(unwrapped, axis=0), axis=1)
    keep = np.concatenate([[True], segment > 1.0e-14])
    unwrapped = unwrapped[keep]
    if unwrapped.shape[0] == 1:
        return np.repeat(unwrapped, n, axis=0)
    return _reparameterize_values(unwrapped, _path_arc(unwrapped, None), n)


def reparameterize_path_with_geometry(
    path: np.ndarray,
    num_images: int,
    *,
    periods: Sequence[float | None] | None = None,
    periodic_geometry: str | None = "minimum_image",
) -> np.ndarray:
    path_arr = np.asarray(path, dtype=np.float64)
    if path_arr.ndim != 2:
        raise ValueError("path must have shape (n_images, n_dim).")
    if _use_sincos_geometry(periodic_geometry, periods, path_arr.shape[1]):
        embedded = periodic_sincos_embed(path_arr, periods=periods)
        reparam = reparameterize_path_periodic(embedded, num_images, periods=None)
        return periodic_sincos_project(reparam, periods=periods)
    return reparameterize_path_periodic(path_arr, num_images, periods=periods)


def smooth_path_periodic(
    path: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    iterations: int = 1,
    window: int = 3,
    preserve_endpoints: bool = True,
) -> np.ndarray:
    out = unwrap_path(np.asarray(path, dtype=np.float64), periods=periods)
    iterations_i = int(iterations)
    window_i = int(window)
    if iterations_i <= 0 or window_i <= 1 or out.shape[0] <= 2:
        return out.copy()
    if window_i % 2 == 0:
        window_i += 1
    half = window_i // 2
    for _ in range(iterations_i):
        padded = np.pad(out, ((half, half), (0, 0)), mode="edge")
        smoothed = np.empty_like(out)
        for idx in range(out.shape[0]):
            smoothed[idx] = np.mean(padded[idx : idx + window_i], axis=0)
        if preserve_endpoints:
            smoothed[0] = out[0]
            smoothed[-1] = out[-1]
        out = smoothed
    return out


def unwrap_path(path: np.ndarray, *, periods: Sequence[float | None] | None = None) -> np.ndarray:
    path_arr = np.asarray(path, dtype=np.float64)
    if path_arr.ndim != 2:
        raise ValueError("path must have shape (n_images, n_dim).")
    period_list = normalize_periods(periods, path_arr.shape[1])
    if not any(period is not None for period in period_list):
        return path_arr.copy()
    out = np.empty_like(path_arr)
    out[0] = path_arr[0]
    for idx in range(1, path_arr.shape[0]):
        out[idx] = out[idx - 1] + minimum_image_delta(path_arr[idx] - path_arr[idx - 1], period_list)
    return out



def assign_pathway_expansions(
    points: np.ndarray,
    paths: Sequence[np.ndarray],
    *,
    periods: Sequence[float | None] | None = None,
    periodic_geometry: str | None = "minimum_image",
    chunk_size: int = 65536,
    device: str | None = None,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    path_list = [np.asarray(p, dtype=np.float64) for p in paths]
    if not path_list:
        raise ValueError("paths must not be empty.")
    ndim = path_list[0].shape[-1]
    point_arr = np.asarray(points, dtype=np.float64)
    use_sincos = _use_sincos_geometry(periodic_geometry, periods, ndim)
    if use_sincos:
        point_arr = periodic_sincos_embed(point_arr, periods=periods)
    labels = []
    distances = []
    for path in path_list:
        query_path = periodic_sincos_embed(path, periods=periods) if use_sincos else path
        lab, dist = assign_voronoi_cells(
            point_arr,
            query_path,
            periods=None if use_sincos else periods,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
        )
        labels.append(lab)
        distances.append(dist)
    return np.asarray(labels, dtype=np.int64), np.asarray(distances, dtype=np.float64)


def cross_path_exchange_counts(
    local_labels: np.ndarray,
    n_images_per_path: Sequence[int],
    *,
    weights: np.ndarray | None = None,
    weight_mode: str = "count",
    traj_id: np.ndarray | None = None,
    lag: int = 1,
    terminal_image_margin: int = 1,
) -> np.ndarray:
    labels = np.asarray(local_labels, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("local_labels must have shape (n_paths, n_samples).")
    n_paths, n_samples = labels.shape
    lag_i = int(lag)
    if lag_i < 1:
        raise ValueError("lag must be >= 1.")
    n_imgs = [int(n) for n in n_images_per_path]
    if len(n_imgs) != n_paths:
        raise ValueError("n_images_per_path length must match n_paths.")
    offsets = _build_offsets(n_imgs)
    total_nodes = int(offsets[-1])
    counts = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    if n_samples <= lag_i:
        return counts
    mode = str(weight_mode).lower()
    if mode in {"count", "counts", "raw", "raw_count", "raw_counts"}:
        weights_arr = np.ones(n_samples, dtype=np.float64)
    elif mode in {"weight", "weights", "sample_weight", "sample_weights"}:
        weights_arr = normalize_weights(weights, n_samples)
    else:
        raise ValueError("weight_mode must be 'count' or 'sample_weight'.")
    if traj_id is None:
        keep = np.ones(n_samples - lag_i, dtype=bool)
    else:
        traj = np.asarray(traj_id).reshape(-1)
        if traj.shape[0] != n_samples:
            raise ValueError("traj_id length must match local_labels samples.")
        keep = traj[:-lag_i] == traj[lag_i:]
    margin = max(0, int(terminal_image_margin))

    step_weights = weights_arr[:-lag_i][keep]
    for src_path in range(n_paths):
        src_n = n_imgs[src_path]
        src_lab = labels[src_path, :-lag_i][keep]
        src_valid = np.ones(src_n, dtype=bool)
        if margin > 0:
            if src_n > 2 * margin:
                src_valid[:margin] = False
                src_valid[-margin:] = False
            elif src_n > 0:
                src_valid[:] = False
        src_v = src_valid[src_lab]
        for dst_path in range(src_path + 1, n_paths):
            dst_n = n_imgs[dst_path]
            dst_lab = labels[dst_path, lag_i:][keep]
            dst_valid = np.ones(dst_n, dtype=bool)
            if margin > 0:
                if dst_n > 2 * margin:
                    dst_valid[:margin] = False
                    dst_valid[-margin:] = False
                elif dst_n > 0:
                    dst_valid[:] = False
            dst_v = dst_valid[dst_lab]
            valid = src_v & dst_v
            if not np.any(valid):
                continue
            src_nodes = int(offsets[src_path]) + src_lab[valid]
            dst_nodes = int(offsets[dst_path]) + dst_lab[valid]
            np.add.at(counts, (src_nodes, dst_nodes), step_weights[valid])

            rev_src_lab = labels[dst_path, :-lag_i][keep]
            rev_src_v = dst_valid[rev_src_lab]
            rev_dst_lab = labels[src_path, lag_i:][keep]
            rev_dst_v = src_valid[rev_dst_lab]
            rev_valid = rev_src_v & rev_dst_v
            if np.any(rev_valid):
                rev_src = int(offsets[dst_path]) + rev_src_lab[rev_valid]
                rev_dst = int(offsets[src_path]) + rev_dst_lab[rev_valid]
                np.add.at(counts, (rev_src, rev_dst), step_weights[rev_valid])
    counts = counts + counts.T
    np.fill_diagonal(counts, 0.0)
    return counts


def shared_node_labels_from_counts(
    counts: np.ndarray,
    *,
    node_points: np.ndarray | None = None,
    periods: Sequence[float | None] | None = None,
    max_cell_distance: float | None = None,
    min_exchange_count: float = 1.0,
    min_exchange_probability: float = 0.0,
) -> np.ndarray:
    counts_arr = np.asarray(counts, dtype=np.float64)
    if counts_arr.ndim != 2 or counts_arr.shape[0] != counts_arr.shape[1]:
        raise ValueError("counts must be a square matrix.")
    total = float(np.sum(counts_arr))
    probs = counts_arr / total if total > 0.0 else np.zeros_like(counts_arr)
    node_points_arr = None if node_points is None else np.asarray(node_points, dtype=np.float64)
    if node_points_arr is not None and node_points_arr.shape[0] != counts_arr.shape[0]:
        raise ValueError("node_points length must match counts.")
    uf = _UnionFind(counts_arr.shape[0])
    for i in range(counts_arr.shape[0]):
        for j in range(i + 1, counts_arr.shape[1]):
            if counts_arr[i, j] < float(min_exchange_count) or probs[i, j] < float(min_exchange_probability):
                continue
            if node_points_arr is not None and max_cell_distance is not None:
                delta = minimum_image_delta(node_points_arr[j] - node_points_arr[i], periods)
                if float(np.linalg.norm(delta)) > float(max_cell_distance):
                    continue
            uf.union(i, j)
    return uf.labels()


def shared_segments_from_node_labels(
    node_labels: np.ndarray,
    n_paths: int,
    n_images_per_path: Sequence[int],
    *,
    terminal_image_margin: int = 1,
) -> list[list[tuple[int, int]]]:
    labels = np.asarray(node_labels, dtype=np.int64)
    n_imgs = [int(n) for n in n_images_per_path]
    if len(n_imgs) != int(n_paths):
        raise ValueError("n_images_per_path length must match n_paths.")
    offsets = _build_offsets(n_imgs)
    margin = max(0, int(terminal_image_margin))
    segments: list[list[tuple[int, int]]] = []
    for label in np.unique(labels):
        members_flat = np.flatnonzero(labels == int(label))
        members = []
        for node_idx in members_flat:
            path_idx, image_idx = _node_to_path(int(node_idx), offsets)
            n_i = n_imgs[path_idx]
            if margin > 0 and (image_idx < margin or image_idx >= n_i - margin):
                continue
            members.append((path_idx, image_idx))
        if len(members) == 0:
            continue
        paths_in_segment = {p for p, _ in members}
        if len(paths_in_segment) < 2:
            continue
        segments.append(members)
    return segments


def decompose_pathway_segments(
    n_images_per_path: Sequence[int],
    shared_segments: list[list[tuple[int, int]]],
) -> list[list[dict[str, int | str | None]]]:
    """Decompose each pathway into contiguous shared and unique segments.

    Returns one list per pathway. Each entry is a dict with keys:
        start, end (int): inclusive image index range on this pathway
        type: "shared" or "unique"
        global_segment_id: int for shared segments, None for unique
    """
    n_imgs = [int(n) for n in n_images_per_path]
    n_paths = len(n_imgs)

    image_to_global: dict[tuple[int, int], int] = {}
    for gid, members in enumerate(shared_segments):
        for path_idx, image_idx in members:
            image_to_global[(int(path_idx), int(image_idx))] = gid

    all_segments: list[list[dict[str, int | str | None]]] = []
    for path_idx in range(n_paths):
        path_segments: list[dict[str, int | str | None]] = []
        if n_imgs[path_idx] == 0:
            all_segments.append(path_segments)
            continue
        current_start = 0
        current_gid = image_to_global.get((path_idx, 0))
        for img in range(1, n_imgs[path_idx]):
            gid = image_to_global.get((path_idx, img))
            if gid != current_gid:
                path_segments.append({
                    "start": current_start,
                    "end": img - 1,
                    "type": "shared" if current_gid is not None else "unique",
                    "global_segment_id": current_gid,
                })
                current_start = img
                current_gid = gid
        path_segments.append({
            "start": current_start,
            "end": n_imgs[path_idx] - 1,
            "type": "shared" if current_gid is not None else "unique",
            "global_segment_id": current_gid,
        })
        all_segments.append(path_segments)
    return all_segments


def _valid_exchange_edges(
    counts: np.ndarray,
    node_points: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    max_cell_distance: float | None = None,
    min_exchange_count: float = 1.0,
    min_exchange_probability: float = 0.0,
) -> np.ndarray:
    counts_arr = np.asarray(counts, dtype=np.float64)
    node_points_arr = np.asarray(node_points, dtype=np.float64)
    if counts_arr.ndim != 2 or counts_arr.shape[0] != counts_arr.shape[1]:
        raise ValueError("counts must be a square matrix.")
    if node_points_arr.shape[0] != counts_arr.shape[0]:
        raise ValueError("node_points length must match counts.")
    total = float(np.sum(counts_arr))
    probs = counts_arr / total if total > 0.0 else np.zeros_like(counts_arr)
    edges = (counts_arr >= float(min_exchange_count)) & (probs >= float(min_exchange_probability))
    if max_cell_distance is not None and np.any(edges):
        ids_i, ids_j = np.nonzero(np.triu(edges, k=1))
        for i, j in zip(ids_i, ids_j):
            delta = minimum_image_delta(node_points_arr[j] - node_points_arr[i], periods)
            if float(np.linalg.norm(delta)) > float(max_cell_distance):
                edges[i, j] = False
                edges[j, i] = False
    np.fill_diagonal(edges, False)
    return edges


def exchange_edge_table(
    counts: np.ndarray,
    node_points: np.ndarray,
    *,
    n_images_per_path: Sequence[int],
    periods: Sequence[float | None] | None = None,
    max_cell_distance: float | None = None,
    min_exchange_count: float = 1.0,
    min_exchange_probability: float = 0.0,
) -> list[dict[str, float | int | bool]]:
    counts_arr = np.asarray(counts, dtype=np.float64)
    node_points_arr = np.asarray(node_points, dtype=np.float64)
    offsets = _build_offsets(n_images_per_path)
    total = float(np.sum(counts_arr))
    rows: list[dict[str, float | int | bool]] = []
    for i, j in np.argwhere(np.triu(counts_arr, k=1) > 0.0):
        p_i, im_i = _node_to_path(int(i), offsets)
        p_j, im_j = _node_to_path(int(j), offsets)
        delta = minimum_image_delta(node_points_arr[j] - node_points_arr[i], periods)
        distance = float(np.linalg.norm(delta))
        probability = float(counts_arr[i, j] / total) if total > 0.0 else 0.0
        accepted = counts_arr[i, j] >= float(min_exchange_count) and probability >= float(min_exchange_probability)
        if max_cell_distance is not None and distance > float(max_cell_distance):
            accepted = False
        rows.append(
            {
                "path_i": p_i,
                "image_i": im_i,
                "path_j": p_j,
                "image_j": im_j,
                "count": float(counts_arr[i, j]),
                "probability": probability,
                "distance": distance,
                "accepted": bool(accepted),
            }
        )
    rows.sort(key=lambda row: (not bool(row["accepted"]), -float(row["count"]), float(row["distance"])))
    return rows


def build_pathway_network(
    paths: Sequence[np.ndarray],
    exchange_counts: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    min_exchange_count: float = 1.0,
    min_exchange_probability: float = 0.0,
    max_cell_distance: float | None = None,
    terminal_image_margin: int = 1,
) -> PathwayNetwork:
    """Build an image-level graph from intra-path and exchange edges."""
    path_list = [np.asarray(p, dtype=np.float64) for p in paths]
    if not path_list:
        raise ValueError("paths must not be empty.")
    n_imgs = [p.shape[0] for p in path_list]
    offsets = _build_offsets(n_imgs)
    node_points = np.vstack(path_list)
    counts = np.asarray(exchange_counts, dtype=np.float64)
    margins = max(0, int(terminal_image_margin))

    edges = _valid_exchange_edges(
        counts,
        node_points,
        periods=periods,
        max_cell_distance=max_cell_distance,
        min_exchange_count=min_exchange_count,
        min_exchange_probability=min_exchange_probability,
    )

    adj: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for path_idx in range(len(path_list)):
        for img in range(n_imgs[path_idx]):
            adj.setdefault((path_idx, img), [])

    for path_idx in range(len(path_list)):
        for img in range(n_imgs[path_idx] - 1):
            adj[(path_idx, img)].append((path_idx, img + 1, 1.0))
            adj[(path_idx, img + 1)].append((path_idx, img, 1.0))

    for i in range(edges.shape[0]):
        for j in range(i + 1, edges.shape[1]):
            if not edges[i, j]:
                continue
            p_i, im_i = _node_to_path(int(i), offsets)
            p_j, im_j = _node_to_path(int(j), offsets)
            if margins > 0:
                if im_i < margins or im_i >= n_imgs[p_i] - margins:
                    continue
                if im_j < margins or im_j >= n_imgs[p_j] - margins:
                    continue
            weight = float(counts[i, j])
            adj[(p_i, im_i)].append((p_j, im_j, weight))
            adj[(p_j, im_j)].append((p_i, im_i, weight))

    start_nodes: list[tuple[int, int]] = []
    end_nodes: list[tuple[int, int]] = []
    branch_points: list[tuple[int, int]] = []
    for (p, _), neighbors in adj.items():
        degree = len(neighbors)
        if degree > 2:
            branch_points.append((p, _))
    for path_idx in range(len(path_list)):
        if n_imgs[path_idx] > 0:
            start_nodes.append((path_idx, 0))
            end_nodes.append((path_idx, n_imgs[path_idx] - 1))

    return PathwayNetwork(
        adjacency=adj,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
        branch_points=branch_points,
    )


def find_all_reactive_pathways(
    network: PathwayNetwork,
    *,
    max_routes: int = 10000,
) -> list[list[tuple[int, int]]]:
    """DFS enumeration of all state-0 to state-1 routes through the network.

    Each route is a sequence of (path_idx, image_idx) nodes. Cycles are
    prevented by tracking visited nodes per branch.
    """
    end_set = set(network.end_nodes)
    routes: list[list[tuple[int, int]]] = []

    def _dfs(node: tuple[int, int], path: list[tuple[int, int]], visited: set[tuple[int, int]]):
        if len(routes) >= int(max_routes):
            return
        if node in end_set and len(path) > 1:
            routes.append(list(path))
            return
        for n_p, n_i, _w in network.adjacency.get(node, []):
            neighbor = (int(n_p), int(n_i))
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            _dfs(neighbor, path, visited)
            path.pop()
            visited.discard(neighbor)
            if len(routes) >= int(max_routes):
                return

    for start in network.start_nodes:
        if len(routes) >= int(max_routes):
            break
        _dfs(start, [start], {start})

    return routes


def relax_images_by_dynamic_edges(
    paths: Sequence[np.ndarray],
    exchange_counts: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    terminal_image_margin: int = 1,
    min_exchange_count: float = 1.0,
    min_exchange_probability: float = 0.0,
    exchange_weight_mode: str = "count",
    max_cell_distance: float | None = None,
    relaxation: float = 1.0,
    endpoints: np.ndarray | None = None,
    wrap_bounds: Sequence[Sequence[float] | None] | None = None,
    periodic_geometry: str | None = "minimum_image",
) -> list[np.ndarray]:
    out = [np.asarray(p, dtype=np.float64).copy() for p in paths]
    if not out:
        return out
    ndim = out[0].shape[-1]
    n_imgs = [p.shape[0] for p in out]
    offsets = _build_offsets(n_imgs)
    node_points = np.vstack(out)
    counts = np.asarray(exchange_counts, dtype=np.float64)
    edges = _valid_exchange_edges(
        counts,
        node_points,
        periods=periods,
        max_cell_distance=max_cell_distance,
        min_exchange_count=min_exchange_count,
        min_exchange_probability=min_exchange_probability,
    )
    margin = max(0, int(terminal_image_margin))
    relax = float(relaxation)
    use_sincos = _use_sincos_geometry(periodic_geometry, periods, ndim)
    for node_idx in range(edges.shape[0]):
        path_idx, image_idx = _node_to_path(node_idx, offsets)
        n_i = n_imgs[path_idx]
        if margin > 0 and (image_idx < margin or image_idx >= n_i - margin):
            continue
        neighbors = np.flatnonzero(edges[node_idx])
        if neighbors.size == 0:
            continue
        coords = np.vstack([node_points[node_idx][None, :], node_points[neighbors]])
        edge_weights = counts[node_idx, neighbors]
        center_weights = np.concatenate([[max(float(np.sum(edge_weights)), 1.0e-300)], edge_weights])
        if use_sincos:
            center = periodic_sincos_weighted_mean(coords, center_weights, periods=periods)
        else:
            center = periodic_weighted_mean(coords, center_weights, periods=periods)
        delta = minimum_image_delta(center - out[path_idx][image_idx], periods)
        out[path_idx][image_idx] = out[path_idx][image_idx] + relax * delta
    if endpoints is not None:
        endpoint_arr = np.asarray(endpoints, dtype=np.float64)
        for path_idx in range(len(out)):
            out[path_idx][0] = endpoint_arr[path_idx, 0]
            out[path_idx][-1] = endpoint_arr[path_idx, 1]
    return [wrap_periodic_points(p, periods=periods, wrap_bounds=wrap_bounds) for p in out]


def smooth_reparameterize_paths_independently(
    paths: Sequence[np.ndarray],
    *,
    periods: Sequence[float | None] | None = None,
    num_images: int | None = None,
    image_spacing: float | None = None,
    smooth_iterations: int = 2,
    smooth_window: int = 5,
    endpoints: np.ndarray | None = None,
    wrap_bounds: Sequence[Sequence[float] | None] | None = None,
    periodic_geometry: str | None = "minimum_image",
) -> list[np.ndarray]:
    path_list = [np.asarray(p, dtype=np.float64) for p in paths]
    if not path_list:
        return []
    ndim = path_list[0].shape[-1]
    use_sincos = _use_sincos_geometry(periodic_geometry, periods, ndim)
    out = []
    for path_idx, path in enumerate(path_list):
        arc = _path_arc(path, periods)
        arc_len = float(arc[-1])
        if image_spacing is not None and float(image_spacing) > 0.0:
            n_images_i = max(2, int(round(arc_len / float(image_spacing))))
        elif num_images is not None:
            n_images_i = int(num_images)
        else:
            n_images_i = int(path.shape[0])
        if use_sincos:
            embedded = periodic_sincos_embed(path, periods=periods)
            current_embedded = smooth_path_periodic(
                embedded,
                periods=None,
                iterations=int(smooth_iterations),
                window=int(smooth_window),
                preserve_endpoints=True,
            )
            current_embedded = reparameterize_path_periodic(current_embedded, n_images_i, periods=None)
            current = periodic_sincos_project(current_embedded, periods=periods)
        else:
            current = smooth_path_periodic(
                path,
                periods=periods,
                iterations=int(smooth_iterations),
                window=int(smooth_window),
                preserve_endpoints=True,
            )
            current = reparameterize_path_periodic(current, n_images_i, periods=periods)
        if endpoints is not None:
            endpoint_arr = np.asarray(endpoints, dtype=np.float64)
            if endpoint_arr.ndim == 3:
                current[0] = endpoint_arr[path_idx, 0]
                current[-1] = endpoint_arr[path_idx, 1]
            else:
                current[0] = endpoint_arr[path_idx, 0]
                current[-1] = endpoint_arr[path_idx, 1]
        current = wrap_periodic_points(current, periods=periods, wrap_bounds=wrap_bounds)
        out.append(current)
    return out


def _path_shift(old: Sequence[np.ndarray], new: Sequence[np.ndarray], periods: Sequence[float | None] | None) -> float:
    if len(old) != len(new):
        return float("inf")
    max_shift = 0.0
    for p_old, p_new in zip(old, new):
        if p_old.shape != p_new.shape:
            return float("inf")
        delta = minimum_image_delta(p_new - p_old, periods)
        if delta.size:
            max_shift = max(max_shift, float(np.max(np.linalg.norm(delta, axis=1))))
    return max_shift


def run_iterative_pathway_expansion(
    paths: np.ndarray | Sequence[np.ndarray],
    points: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    traj_id: np.ndarray | None = None,
    periods: Sequence[float | None] | None = None,
    lag: int = 1,
    terminal_image_margin: int = 1,
    min_exchange_count: float = 1.0,
    min_exchange_probability: float = 0.0,
    exchange_weight_mode: str = "count",
    max_cell_distance: float | None = None,
    max_iterations: int = 10,
    convergence_tol: float = 1.0e-6,
    num_images: int | None = None,
    image_spacing: float | None = None,
    smooth_iterations: int = 2,
    smooth_window: int = 5,
    fixed_endpoints: bool = True,
    cell_relaxation: float = 1.0,
    wrap_bounds: Sequence[Sequence[float] | None] | None = None,
    chunk_size: int = 65536,
    device: str | None = None,
    dtype: str = "float32",
    periodic_geometry: str | None = "minimum_image",
    out_dir: str | None = None,
    plot_config: dict | None = None,
    final_image_spacing: float | None = None,
) -> IterativePathwayResult:
    if isinstance(paths, np.ndarray):
        if paths.ndim == 3:
            current = [np.asarray(p, dtype=np.float64) for p in paths]
        elif paths.ndim == 2:
            current = [np.asarray(paths, dtype=np.float64)]
        else:
            raise ValueError("paths must have shape (n_paths, n_images, n_dim) or (n_images, n_dim).")
    else:
        current = [np.asarray(p, dtype=np.float64) for p in paths]
    if not current:
        raise ValueError("paths must not be empty.")

    spacing = float(image_spacing) if image_spacing is not None and float(image_spacing) > 0.0 else None
    n_paths = len(current)
    if spacing is not None:
        n_imgs = []
        for p in current:
            arc = _path_arc(p, periods)
            n_imgs.append(max(2, int(round(float(arc[-1]) / spacing))))
    elif num_images is not None:
        n_imgs = [int(num_images)] * n_paths
    else:
        n_imgs = [int(p.shape[0]) for p in current]

    current = [
        reparameterize_path_with_geometry(p, n_i, periods=periods, periodic_geometry=periodic_geometry)
        for p, n_i in zip(current, n_imgs)
    ]

    if bool(fixed_endpoints):
        endpoint_values = np.stack([[p[0], p[-1]] for p in current], axis=0)
    else:
        endpoint_values = None
    current = [
        wrap_periodic_points(p, periods=periods, wrap_bounds=wrap_bounds) for p in current
    ]

    history: list[PathwayIteration] = []
    converged = False
    previous_n_paths: int | None = None
    for iteration in range(int(max_iterations)):
        local_labels, local_distances = assign_pathway_expansions(
            points,
            current,
            periods=periods,
            periodic_geometry=periodic_geometry,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
        )
        n_imgs = [p.shape[0] for p in current]
        counts = cross_path_exchange_counts(
            local_labels,
            n_imgs,
            weights=weights,
            weight_mode=exchange_weight_mode,
            traj_id=traj_id,
            lag=lag,
            terminal_image_margin=terminal_image_margin,
        )
        node_points = np.vstack(current)
        node_labels = shared_node_labels_from_counts(
            counts,
            node_points=node_points,
            periods=periods,
            max_cell_distance=max_cell_distance,
            min_exchange_count=min_exchange_count,
            min_exchange_probability=min_exchange_probability,
        )
        shared_segments = shared_segments_from_node_labels(
            node_labels,
            n_paths,
            n_imgs,
            terminal_image_margin=terminal_image_margin,
        )
        relaxed = relax_images_by_dynamic_edges(
            current,
            counts,
            periods=periods,
            terminal_image_margin=terminal_image_margin,
            min_exchange_count=min_exchange_count,
            min_exchange_probability=min_exchange_probability,
            max_cell_distance=max_cell_distance,
            relaxation=cell_relaxation,
            endpoints=endpoint_values,
            wrap_bounds=wrap_bounds,
            periodic_geometry=periodic_geometry,
        )
        if spacing is not None:
            updated_n_imgs = []
            for p in relaxed:
                arc = _path_arc(p, periods)
                updated_n_imgs.append(max(2, int(round(float(arc[-1]) / spacing))))
        elif num_images is not None:
            updated_n_imgs = [int(num_images)] * n_paths
        else:
            updated_n_imgs = [int(p.shape[0]) for p in relaxed]

        updated = smooth_reparameterize_paths_independently(
            relaxed,
            periods=periods,
            num_images=num_images if spacing is None else None,
            image_spacing=spacing,
            smooth_iterations=smooth_iterations,
            smooth_window=smooth_window,
            endpoints=endpoint_values,
            wrap_bounds=wrap_bounds,
            periodic_geometry=periodic_geometry,
        )
        n_imgs = updated_n_imgs
        max_shift = _path_shift(current, updated, periods)
        converged = (
            previous_n_paths == len(updated) == len(current)
            and max_shift <= float(convergence_tol)
        )
        item = PathwayIteration(
            iteration=iteration,
            paths=[p.copy() for p in current],
            relaxed_paths=[p.copy() for p in updated],
            local_labels=local_labels.copy(),
            local_distances=local_distances.copy(),
            node_labels=node_labels.copy(),
            shared_segments=[list(segment) for segment in shared_segments],
            exchange_counts=counts.copy(),
            n_cells=int(np.unique(node_labels).size),
            max_shift=max_shift,
            converged=converged,
        )
        history.append(item)
        if out_dir is not None:
            iter_dir = os.path.join(out_dir, f"iteration_{iteration:03d}")
            os.makedirs(iter_dir, exist_ok=True)
            np.save(os.path.join(iter_dir, "paths.npy"), np.asarray(item.paths, dtype=object), allow_pickle=True)
            np.save(os.path.join(iter_dir, "relaxed_paths.npy"), np.asarray(item.relaxed_paths, dtype=object), allow_pickle=True)
            np.save(os.path.join(iter_dir, "local_labels.npy"), item.local_labels)
            np.save(os.path.join(iter_dir, "local_distances.npy"), item.local_distances)
            np.save(os.path.join(iter_dir, "node_labels.npy"), item.node_labels)
            np.save(os.path.join(iter_dir, "exchange_counts.npy"), item.exchange_counts)
            with open(os.path.join(iter_dir, "pathway_lengths.txt"), "w") as f:
                f.write(" ".join(str(p.shape[0]) for p in item.paths) + "\n")
            with open(os.path.join(iter_dir, "shared_segments.txt"), "w", encoding="utf-8") as f:
                for segment in shared_segments:
                    f.write(" ".join(f"({path_idx},{image_idx})" for path_idx, image_idx in segment) + "\n")
            edge_rows = exchange_edge_table(
                counts,
                node_points,
                n_images_per_path=[p.shape[0] for p in current],
                periods=periods,
                max_cell_distance=max_cell_distance,
                min_exchange_count=min_exchange_count,
                min_exchange_probability=min_exchange_probability,
            )
            with open(os.path.join(iter_dir, "exchange_edges.tsv"), "w", encoding="utf-8") as f:
                f.write("path_i\timage_i\tpath_j\timage_j\tcount\tprobability\tdistance\taccepted\n")
                for row in edge_rows:
                    f.write(
                        f"{row['path_i']}\t{row['image_i']}\t{row['path_j']}\t{row['image_j']}\t"
                        f"{row['count']:.12g}\t{row['probability']:.12g}\t{row['distance']:.12g}\t"
                        f"{int(bool(row['accepted']))}\n"
                    )
            plot_cfg = {} if plot_config is None else dict(plot_config)
            if bool(plot_cfg.get("enabled", True)):
                from .plot import plot_pathway_iteration_2d
                import matplotlib.pyplot as plt

                plot_cfg.setdefault("title", f"Pathway expansion iteration {iteration}")
                fig = plot_pathway_iteration_2d(
                    points,
                    item.paths,
                    item.local_distances,
                    periods=periods,
                    save_path=os.path.join(iter_dir, "pathway_expansion.png"),
                    config=plot_cfg,
                )
                plt.close(fig)
                relaxed_cfg = dict(plot_cfg)
                relaxed_cfg["title"] = f"Shared-segment aligned pathways iteration {iteration}"
                fig = plot_pathway_iteration_2d(
                    points,
                    item.relaxed_paths,
                    item.local_distances,
                    periods=periods,
                    save_path=os.path.join(iter_dir, "pathway_relaxed.png"),
                    config=relaxed_cfg,
                )
                plt.close(fig)
                if shared_segments:
                    from .plot import plot_shared_segments

                    decomposed = decompose_pathway_segments(
                        [p.shape[0] for p in item.paths], shared_segments
                    )
                    seg_cfg = dict(plot_cfg)
                    seg_cfg["title"] = f"Shared segments iteration {iteration}"
                    fig = plot_shared_segments(
                        points,
                        item.paths,
                        decomposed,
                        item.local_distances,
                        periods=periods,
                        save_path=os.path.join(iter_dir, "shared_segments.png"),
                        config=seg_cfg,
                    )
                    plt.close(fig)
                network = build_pathway_network(
                    current,
                    counts,
                    periods=periods,
                    min_exchange_count=min_exchange_count,
                    min_exchange_probability=min_exchange_probability,
                    max_cell_distance=max_cell_distance,
                    terminal_image_margin=terminal_image_margin,
                )
                adj_path = os.path.join(iter_dir, "pathway_network_adjacency.tsv")
                with open(adj_path, "w", encoding="utf-8") as f:
                    f.write("source_path\tsource_image\ttarget_path\ttarget_image\tweight\n")
                    for (p_i, im_i), neighbors in network.adjacency.items():
                        for p_j, im_j, w in neighbors:
                            f.write(f"{p_i}\t{im_i}\t{p_j}\t{im_j}\t{w:.12g}\n")
                routes = find_all_reactive_pathways(network, max_routes=10000)
                if routes:
                    import json as _json
                    route_data = {
                        "n_routes": len(routes),
                        "n_branch_points": len(network.branch_points),
                        "branch_points": [list(bp) for bp in network.branch_points],
                        "routes": [
                            {
                                "route_id": ridx,
                                "n_nodes": len(route),
                                "node_sequence": [list(n) for n in route],
                            }
                            for ridx, route in enumerate(routes)
                        ],
                    }
                    with open(os.path.join(iter_dir, "pathway_routes.json"), "w", encoding="utf-8") as f:
                        _json.dump(route_data, f, indent=2)
                if bool(plot_cfg.get("enabled", True)):
                    from .plot import plot_pathway_network
                    import matplotlib.pyplot as plt

                    net_cfg = dict(plot_cfg)
                    net_cfg["title"] = f"Reactive pathway network iteration {iteration}"
                    fig = plot_pathway_network(
                        points,
                        item.paths,
                        network,
                        periods=periods,
                        save_path=os.path.join(iter_dir, "pathway_network.png"),
                        config=net_cfg,
                    )
                    plt.close(fig)
        if converged:
            current = updated
            break
        if np.isfinite(max_shift) and max_shift <= float(convergence_tol):
            current = updated
            converged = True
            break
        previous_n_paths = len(current)
        current = updated
    final_eps = (
        float(final_image_spacing)
        if final_image_spacing is not None and float(final_image_spacing) > 0.0
        else (spacing / 2.0 if spacing is not None else None)
    )
    if final_eps is not None:
        coarse_eps = final_eps * 2.0
        current = smooth_reparameterize_paths_independently(
            current,
            periods=periods,
            image_spacing=coarse_eps,
            smooth_iterations=smooth_iterations,
            smooth_window=smooth_window,
            endpoints=endpoint_values,
            wrap_bounds=wrap_bounds,
            periodic_geometry=periodic_geometry,
        )
        current = smooth_reparameterize_paths_independently(
            current,
            periods=periods,
            image_spacing=final_eps,
            smooth_iterations=smooth_iterations,
            smooth_window=smooth_window,
            endpoints=endpoint_values,
            wrap_bounds=wrap_bounds,
            periodic_geometry=periodic_geometry,
        )
    if out_dir is not None and bool((plot_config or {}).get("enabled", True)):
        from .plot import plot_pathway_iteration_2d
        import matplotlib.pyplot as plt

        final_cfg = {} if plot_config is None else dict(plot_config)
        final_cfg["title"] = "Final converged pathways"
        fig = plot_pathway_iteration_2d(
            points,
            current,
            np.zeros((len(current), points.shape[0]), dtype=np.float64),
            periods=periods,
            save_path=os.path.join(out_dir, "final_pathways.png"),
            config=final_cfg,
        )
        plt.close(fig)
    return IterativePathwayResult(paths=current, history=history, converged=converged)

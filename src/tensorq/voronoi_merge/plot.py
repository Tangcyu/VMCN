from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .core import normalize_periods


def _thin_indices(n_points: int, max_points: int) -> np.ndarray:
    max_points = int(max_points)
    if max_points <= 0 or int(n_points) <= max_points:
        return np.arange(int(n_points), dtype=np.int64)
    return np.linspace(0, int(n_points) - 1, max_points, dtype=np.int64)


def _break_periodic_line(values: np.ndarray, periods: Sequence[float | None] | None) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.shape[0] < 2:
        return values_arr
    period_list = normalize_periods(periods, values_arr.shape[1])
    if not any(period is not None for period in period_list):
        return values_arr
    jumps = np.zeros(values_arr.shape[0] - 1, dtype=bool)
    for dim, period in enumerate(period_list):
        if period is not None:
            jumps |= np.abs(np.diff(values_arr[:, dim])) > 0.5 * float(period)
    if not bool(np.any(jumps)):
        return values_arr
    rows = []
    for idx in range(values_arr.shape[0] - 1):
        rows.append(values_arr[idx])
        if jumps[idx]:
            rows.append(np.full(values_arr.shape[1], np.nan, dtype=np.float64))
    rows.append(values_arr[-1])
    return np.asarray(rows, dtype=np.float64)


def plot_pathway_iteration_2d(
    points: np.ndarray,
    paths: np.ndarray | Sequence[np.ndarray],
    local_distances: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    save_path: str | os.PathLike[str] | None = None,
    config: dict | None = None,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cfg = {} if config is None else dict(config)
    points_arr = np.asarray(points, dtype=np.float64)
    if isinstance(paths, np.ndarray) and paths.dtype != object:
        paths_arr = np.asarray(paths, dtype=np.float64)
    else:
        paths_arr = [np.asarray(p, dtype=np.float64) for p in paths]
    distances = np.asarray(local_distances, dtype=np.float64)
    if points_arr.ndim != 2 or points_arr.shape[1] < 2:
        raise ValueError("points must have shape (n_samples, n_dim>=2).")
    n_paths = paths_arr.shape[0] if isinstance(paths_arr, np.ndarray) else len(paths_arr)
    if isinstance(paths_arr, np.ndarray):
        if paths_arr.ndim != 3 or paths_arr.shape[2] < 2:
            raise ValueError("paths must have shape (n_paths, n_images, n_dim>=2).")
    else:
        if not paths_arr or paths_arr[0].ndim != 2 or paths_arr[0].shape[1] < 2:
            raise ValueError("each path must have shape (n_images, n_dim>=2).")
    if distances.shape != (n_paths, points_arr.shape[0]):
        raise ValueError("local_distances must have shape (n_paths, n_samples).")

    axes = list(cfg.get("plot_axes", cfg.get("axes", [0, 1])))
    if len(axes) != 2:
        raise ValueError("plot_axes must contain exactly two axes.")
    x_idx, y_idx = int(axes[0]), int(axes[1])
    axis_names = cfg.get("axis_names", cfg.get("cvs_to_use", None))
    xlabel = str(axis_names[x_idx]) if axis_names is not None and x_idx < len(axis_names) else f"axis {x_idx}"
    ylabel = str(axis_names[y_idx]) if axis_names is not None and y_idx < len(axis_names) else f"axis {y_idx}"
    plot_periods = None if periods is None else [periods[x_idx], periods[y_idx]]

    nearest_path = np.argmin(distances, axis=0)
    ids = _thin_indices(points_arr.shape[0], int(cfg.get("plot_max_points", cfg.get("max_points", 100000))))
    cmap = plt.get_cmap(str(cfg.get("pathway_cmap", "tab20")))
    colors = [cmap(idx % cmap.N) for idx in range(n_paths)]

    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [6.0, 5.2])))
    point_size = float(cfg.get("point_size", 4.0))
    point_alpha = float(cfg.get("point_alpha", 0.16))
    for path_idx in range(n_paths):
        mask = ids[nearest_path[ids] == path_idx]
        if mask.size == 0:
            continue
        ax.scatter(
            points_arr[mask, x_idx],
            points_arr[mask, y_idx],
            s=point_size,
            color=colors[path_idx],
            alpha=point_alpha,
            linewidths=0,
        )

    for path_idx, path in enumerate(paths_arr):
        xy = _break_periodic_line(path[:, [x_idx, y_idx]], plot_periods)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=colors[path_idx],
            linewidth=float(cfg.get("path_linewidth", 2.2)),
            alpha=float(cfg.get("path_alpha", 0.95)),
            label=f"path {path_idx + 1}",
        )
        ax.scatter(
            path[:, x_idx],
            path[:, y_idx],
            s=float(cfg.get("image_marker_size", 18.0)),
            color=colors[path_idx],
            edgecolors="black",
            linewidths=0.25,
            alpha=0.95,
        )

    for key, setter in (("xlim", ax.set_xlim), ("ylim", ax.set_ylim)):
        value = cfg.get(key, None)
        if value is not None:
            setter(float(value[0]), float(value[1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = cfg.get("title", None)
    if title is not None:
        ax.set_title(str(title))
    if bool(cfg.get("legend", True)):
        ax.legend(loc=str(cfg.get("legend_loc", "upper right")), frameon=False, fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=int(cfg.get("dpi", 220)), bbox_inches="tight")
    return fig


def plot_shared_segments(
    points: np.ndarray,
    paths: np.ndarray | Sequence[np.ndarray],
    decomposed_segments: list[list[dict[str, int | str | None]]],
    local_distances: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
    save_path: str | os.PathLike[str] | None = None,
    config: dict | None = None,
):
    """Plot pathways with shared segments highlighted in consistent colours.

    Shared segments with the same global_segment_id are drawn in the same
    colour across pathways.  Unique (unshared) portions are drawn in grey
    with dashed lines.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cfg = {} if config is None else dict(config)
    points_arr = np.asarray(points, dtype=np.float64)
    if isinstance(paths, np.ndarray) and paths.dtype != object:
        paths_arr = np.asarray(paths, dtype=np.float64)
    else:
        paths_arr = [np.asarray(p, dtype=np.float64) for p in paths]
    distances = np.asarray(local_distances, dtype=np.float64)
    n_paths = paths_arr.shape[0] if isinstance(paths_arr, np.ndarray) else len(paths_arr)

    axes = list(cfg.get("plot_axes", cfg.get("axes", [0, 1])))
    x_idx, y_idx = int(axes[0]), int(axes[1])
    axis_names = cfg.get("axis_names", cfg.get("cvs_to_use", None))
    xlabel = str(axis_names[x_idx]) if axis_names is not None and x_idx < len(axis_names) else f"axis {x_idx}"
    ylabel = str(axis_names[y_idx]) if axis_names is not None and y_idx < len(axis_names) else f"axis {y_idx}"
    plot_periods = None if periods is None else [periods[x_idx], periods[y_idx]]

    nearest_path = np.argmin(distances, axis=0)
    ids = _thin_indices(points_arr.shape[0], int(cfg.get("plot_max_points", cfg.get("max_points", 100000))))
    path_cmap = plt.get_cmap(str(cfg.get("pathway_cmap", "tab20")))
    path_colors = [path_cmap(idx % path_cmap.N) for idx in range(n_paths)]

    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [6.0, 5.2])))
    point_size = float(cfg.get("point_size", 4.0))
    point_alpha = float(cfg.get("point_alpha", 0.16))
    for path_idx in range(n_paths):
        mask = ids[nearest_path[ids] == path_idx]
        if mask.size == 0:
            continue
        ax.scatter(
            points_arr[mask, x_idx],
            points_arr[mask, y_idx],
            s=point_size,
            color=path_colors[path_idx],
            alpha=point_alpha,
            linewidths=0,
        )

    unique_color = str(cfg.get("unique_color", "gray"))
    unique_alpha = float(cfg.get("unique_alpha", 0.4))
    unique_style = str(cfg.get("unique_line_style", "--"))
    shared_alpha = float(cfg.get("shared_alpha", 0.95))
    image_size = float(cfg.get("image_marker_size", 18.0))

    all_gids = set()
    for path_segs in decomposed_segments:
        for seg in path_segs:
            gid = seg["global_segment_id"]
            if gid is not None:
                all_gids.add(int(gid))
    n_shared = max(len(all_gids), 1)
    shared_cmap = plt.get_cmap(str(cfg.get("shared_cmap", "tab10")))
    shared_colors: dict[int, tuple] = {}
    for gid in sorted(all_gids):
        shared_colors[gid] = shared_cmap(gid % shared_cmap.N)

    legend_added: set[str] = set()
    for path_idx in range(n_paths):
        path = paths_arr[path_idx]
        segs = decomposed_segments[path_idx]
        for seg in segs:
            start = int(seg["start"])
            end = int(seg["end"]) + 1
            coords = path[start:end]
            xy = _break_periodic_line(coords[:, [x_idx, y_idx]], plot_periods)
            gid = seg["global_segment_id"]
            if gid is not None:
                color = shared_colors[int(gid)]
                label = f"shared {int(gid)}" if f"shared_{int(gid)}" not in legend_added else None
                if label is not None:
                    legend_added.add(f"shared_{int(gid)}")
                ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=float(cfg.get("path_linewidth", 2.2)),
                        alpha=shared_alpha, label=label)
                ax.scatter(coords[:, x_idx], coords[:, y_idx], s=image_size * 0.6,
                           color=color, edgecolors="black", linewidths=0.25, alpha=0.8)
            else:
                label = "unique" if "unique" not in legend_added else None
                if label is not None:
                    legend_added.add("unique")
                ax.plot(xy[:, 0], xy[:, 1], color=unique_color, linestyle=unique_style,
                        linewidth=float(cfg.get("path_linewidth", 1.2)), alpha=unique_alpha, label=label)

    for key, setter in (("xlim", ax.set_xlim), ("ylim", ax.set_ylim)):
        value = cfg.get(key, None)
        if value is not None:
            setter(float(value[0]), float(value[1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = cfg.get("title", None)
    if title is not None:
        ax.set_title(str(title))
    if bool(cfg.get("legend", True)):
        ax.legend(loc=str(cfg.get("legend_loc", "upper right")), frameon=False, fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=int(cfg.get("dpi", 220)), bbox_inches="tight")
    return fig


def plot_pathway_network(
    points: np.ndarray,
    paths: np.ndarray | Sequence[np.ndarray],
    network,  # PathwayNetwork (duck-typed to avoid circular import)
    *,
    periods: Sequence[float | None] | None = None,
    save_path: str | os.PathLike[str] | None = None,
    config: dict | None = None,
):
    """Plot the full reactive pathway network overlaid on data.

    Pathway lines are drawn as thin grey background.  Exchange edges are
    overlaid coloured by weight.  Branch points are marked with red stars,
    start nodes with green triangles and end nodes with blue squares.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cfg = {} if config is None else dict(config)
    points_arr = np.asarray(points, dtype=np.float64)
    if isinstance(paths, np.ndarray) and paths.dtype != object:
        paths_arr = np.asarray(paths, dtype=np.float64)
    else:
        paths_arr = [np.asarray(p, dtype=np.float64) for p in paths]
    n_paths = paths_arr.shape[0] if isinstance(paths_arr, np.ndarray) else len(paths_arr)

    axes = list(cfg.get("plot_axes", cfg.get("axes", [0, 1])))
    x_idx, y_idx = int(axes[0]), int(axes[1])
    axis_names = cfg.get("axis_names", cfg.get("cvs_to_use", None))
    xlabel = str(axis_names[x_idx]) if axis_names is not None and x_idx < len(axis_names) else f"axis {x_idx}"
    ylabel = str(axis_names[y_idx]) if axis_names is not None and y_idx < len(axis_names) else f"axis {y_idx}"
    plot_periods = None if periods is None else [periods[x_idx], periods[y_idx]]

    fig, ax = plt.subplots(figsize=tuple(cfg.get("figsize", [6.0, 5.2])))

    ids = _thin_indices(points_arr.shape[0], int(cfg.get("plot_max_points", cfg.get("max_points", 100000))))
    ax.scatter(
        points_arr[ids, x_idx], points_arr[ids, y_idx],
        s=float(cfg.get("point_size", 2.0)),
        color="lightgray",
        alpha=float(cfg.get("point_alpha", 0.08)),
        linewidths=0,
    )

    path_cmap = plt.get_cmap(str(cfg.get("pathway_cmap", "tab20")))
    for path_idx in range(n_paths):
        path = paths_arr[path_idx]
        xy = _break_periodic_line(path[:, [x_idx, y_idx]], plot_periods)
        ax.plot(xy[:, 0], xy[:, 1], color="grey", linewidth=0.5, alpha=0.4)

    edge_cmap = plt.get_cmap(str(cfg.get("edge_colormap", "plasma")))
    weights: list[float] = []
    for _, neighbors in network.adjacency.items():
        for _, _, w in neighbors:
            weights.append(w)
    max_weight = max(weights) if weights else 1.0

    edge_max_alpha = float(cfg.get("edge_max_alpha", 0.85))
    edge_max_width = float(cfg.get("edge_max_width", 3.0))
    for (p_i, im_i), neighbors in network.adjacency.items():
        xi = paths_arr[p_i][im_i, x_idx], paths_arr[p_i][im_i, y_idx]
        for p_j, im_j, w in neighbors:
            if p_j < p_i or (p_j == p_i and im_j <= im_i):
                continue
            xj = paths_arr[p_j][im_j, x_idx], paths_arr[p_j][im_j, y_idx]
            norm = w / max_weight if max_weight > 0 else 0.0
            ax.plot(
                [xi[0], xj[0]], [xi[1], xj[1]],
                color=edge_cmap(min(1.0, max(0.0, norm))),
                linewidth=edge_max_width * norm,
                alpha=edge_max_alpha * (0.2 + 0.8 * norm),
            )

    if network.branch_points:
        bp_coords = np.array([[paths_arr[p][im, x_idx], paths_arr[p][im, y_idx]]
                              for p, im in network.branch_points])
        ax.scatter(bp_coords[:, 0], bp_coords[:, 1],
                   marker=str(cfg.get("branch_marker", "*")),
                   s=float(cfg.get("branch_size", 120)),
                   color=str(cfg.get("branch_color", "red")),
                   edgecolors="black", linewidths=0.5,
                   label="branch", zorder=5)

    start_coords = np.array([[paths_arr[p][im, x_idx], paths_arr[p][im, y_idx]]
                             for p, im in network.start_nodes])
    ax.scatter(start_coords[:, 0], start_coords[:, 1],
               marker="^", s=float(cfg.get("terminal_size", 80)),
               color="green", edgecolors="black", linewidths=0.5,
               label="start", zorder=5)

    end_coords = np.array([[paths_arr[p][im, x_idx], paths_arr[p][im, y_idx]]
                           for p, im in network.end_nodes])
    ax.scatter(end_coords[:, 0], end_coords[:, 1],
               marker="s", s=float(cfg.get("terminal_size", 80)),
               color="blue", edgecolors="black", linewidths=0.5,
               label="end", zorder=5)

    for key, setter in (("xlim", ax.set_xlim), ("ylim", ax.set_ylim)):
        value = cfg.get(key, None)
        if value is not None:
            setter(float(value[0]), float(value[1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = cfg.get("title", None)
    if title is not None:
        ax.set_title(str(title))
    if bool(cfg.get("legend", True)):
        ax.legend(loc=str(cfg.get("legend_loc", "upper right")), frameon=False, fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=int(cfg.get("dpi", 220)), bbox_inches="tight")
    return fig

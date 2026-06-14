from __future__ import annotations

import os

import numpy as np

from ..common.config import ensure_dir


def _thin_indices(indices, max_points):
    indices = np.asarray(indices, dtype=np.int64)
    max_points = int(max_points)
    if max_points <= 0 or indices.size <= max_points:
        return indices
    keep = np.linspace(0, indices.size - 1, max_points, dtype=np.int64)
    return indices[keep]


def _plot_cfg(config):
    relabel_cfg = config.get("relabel", {})
    return config.get("plotting", relabel_cfg.get("plotting", {}))


def _plot_enabled(config):
    relabel_cfg = config.get("relabel", {})
    return bool(
        relabel_cfg.get(
            "make_plots",
            relabel_cfg.get("make_relabel_plots", config.get("make_plots", False)),
        )
    )


def _resolve_plot_space(pack, graph_features, config):
    cfg = _plot_cfg(config)
    requested = str(cfg.get("space", "cv")).lower()
    if requested in {"features", "feature", "model_features", "graph"} or pack.cv is None:
        data = np.asarray(graph_features, dtype=np.float64)
        headers = [f"feature_{idx}" for idx in range(data.shape[1])]
        source = "model_features" if requested != "graph" else "graph_features"
    else:
        from ..common.data import cv_headers_for_pack

        data = pack.cv.detach().cpu().numpy().astype(np.float64)
        headers = cv_headers_for_pack(pack)
        source = "cv"

    plane = cfg.get("plane", config.get("planes", config.get("cv_plane", [])))
    if isinstance(plane, (list, tuple)) and plane and isinstance(plane[0], (list, tuple)):
        plane = plane[0]
    if not plane:
        n_dims = min(2, data.shape[1])
        indices = list(range(n_dims))
        names = [headers[idx] for idx in indices]
    else:
        indices = []
        names = []
        for item in plane:
            if isinstance(item, int):
                idx = int(item)
            else:
                name = str(item)
                if name not in headers:
                    raise ValueError(f"Plot plane column {name!r} not found in {source} headers.")
                idx = headers.index(name)
            indices.append(idx)
            names.append(headers[idx])
    if len(indices) not in {2, 3}:
        raise ValueError("Relabel plot plane must contain 2 or 3 dimensions.")
    return data[:, indices], names, source


def _scatter_by_value(ax, xy, ids, values, title, labels, *, is_3d, cmap="viridis"):
    if ids.size == 0:
        return None
    if is_3d:
        sc = ax.scatter(
            xy[ids, 0], xy[ids, 1], xy[ids, 2],
            c=values[ids], s=5, cmap=cmap, alpha=0.8, linewidths=0,
            rasterized=True,
        )
        ax.set_zlabel(labels[2])
    else:
        sc = ax.scatter(
            xy[ids, 0], xy[ids, 1],
            c=values[ids], s=5, cmap=cmap, alpha=0.8, linewidths=0,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return sc


def _scatter_by_label(ax, xy, ids, labels_value, title, labels, *, is_3d):
    if ids.size == 0:
        return None
    values = labels_value[ids]
    if is_3d:
        sc = ax.scatter(
            xy[ids, 0], xy[ids, 1], xy[ids, 2],
            c=values, s=5, cmap="tab20", alpha=0.75, linewidths=0,
            rasterized=True,
        )
        ax.set_zlabel(labels[2])
    else:
        sc = ax.scatter(
            xy[ids, 0], xy[ids, 1],
            c=values, s=5, cmap="tab20", alpha=0.75, linewidths=0,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return sc


def _candidate_category(proposal):
    masks = proposal["masks"]
    category = np.zeros(proposal["proposed_labels"].shape[0], dtype=np.int64)
    category[masks["current_entropy_candidate"]] = 1
    category[masks["transition_like_candidate"]] = 2
    category[masks["missing_metastate_h_tau"]] = 3
    category[masks["unresolved_lagged_candidate"]] = 4
    category[masks["density_shell"]] = 5
    category[masks["new_core"]] = 6
    category[masks["removed"]] = 7
    category[masks["changed"]] = np.maximum(category[masks["changed"]], 8)
    return category


def _candidate_legend(ax):
    import matplotlib.lines as mlines

    labels = [
        ("other", "#d0d0d0"),
        ("current entropy", "#1f77b4"),
        ("transition-like", "#ff7f0e"),
        ("persistent lagged entropy", "#9467bd"),
        ("unresolved lagged", "#8c564b"),
        ("density shell", "#7f7f7f"),
        ("new core", "#2ca02c"),
        ("removed state", "#d62728"),
        ("changed", "#e377c2"),
    ]
    handles = [
        mlines.Line2D([], [], color=color, marker="o", linestyle="None", markersize=5, label=name)
        for name, color in labels
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="best")


def plot_relabel_diagnostics(pack, graph_features, old_state, proposal, config, output_dir):
    if not _plot_enabled(config):
        return []

    cfg = _plot_cfg(config)
    plot_dir = ensure_dir(os.path.join(output_dir, str(cfg.get("dir", "plots"))))
    os.environ.setdefault("MPLCONFIGDIR", ensure_dir(os.path.join(plot_dir, ".mplconfig")))

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fmt = str(cfg.get("format", config.get("format", "png")))
    max_points = int(cfg.get("max_points", config.get("max_scatter_points", 20000)))
    max_label_points = int(cfg.get("max_label_points", max(max_points, 100000)))
    xy, plane, source = _resolve_plot_space(pack, graph_features, config)
    is_3d = xy.shape[1] == 3
    ids = _thin_indices(np.arange(xy.shape[0], dtype=np.int64), max_points)
    scores = proposal["scores"]
    new_state = proposal["proposed_labels"]
    saved = []

    projection = {"projection": "3d"} if is_3d else {}

    lagged = scores.get("mean_lagged_entropy_norm")
    has_lagged = lagged is not None and np.any(np.isfinite(lagged))
    fig = plt.figure(figsize=(11.2 if has_lagged else 6.0, 5.0), dpi=160)
    ax = fig.add_subplot(121 if has_lagged else 111, **projection)
    sc = _scatter_by_value(ax, xy, ids, scores["entropy_norm"], "Current entropy H", plane, is_3d=is_3d)
    if sc is not None:
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="H(q)")
    if has_lagged:
        ax = fig.add_subplot(122, **projection)
        finite_ids = ids[np.isfinite(lagged[ids])]
        sc = _scatter_by_value(ax, xy, finite_ids, lagged, "Time-lagged entropy H_tau", plane, is_3d=is_3d)
        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="mean H(q_tau)")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"relabel_entropy.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    fig = plt.figure(figsize=(6.2, 5.1), dpi=160)
    ax = fig.add_subplot(111, **projection)
    category = _candidate_category(proposal)
    colors = [
        "#d0d0d0", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b",
        "#7f7f7f", "#2ca02c", "#d62728", "#e377c2",
    ]
    if is_3d:
        ax.scatter(
            xy[ids, 0], xy[ids, 1], xy[ids, 2],
            c=category[ids], s=6, cmap=ListedColormap(colors),
            vmin=0, vmax=len(colors) - 1, alpha=0.78, linewidths=0,
            rasterized=True,
        )
        ax.set_zlabel(plane[2])
    else:
        ax.scatter(
            xy[ids, 0], xy[ids, 1],
            c=category[ids], s=6, cmap=ListedColormap(colors),
            vmin=0, vmax=len(colors) - 1, alpha=0.78, linewidths=0,
            rasterized=True,
        )
    ax.set_title("Relabel candidates")
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    _candidate_legend(ax)
    fig.tight_layout()
    path = os.path.join(plot_dir, f"relabel_candidates.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    fig = plt.figure(figsize=(11.0, 5.0), dpi=160)
    ax1 = fig.add_subplot(121, **projection)
    ax2 = fig.add_subplot(122, **projection)
    old_state = np.asarray(old_state)
    new_state = np.asarray(new_state)
    old_label_ids = _thin_indices(np.flatnonzero(old_state >= 0), max_label_points)
    new_label_ids = _thin_indices(np.flatnonzero(new_state >= 0), max_label_points)
    sc1 = _scatter_by_label(ax1, xy, old_label_ids, old_state, "Before relabel", plane, is_3d=is_3d)
    sc2 = _scatter_by_label(ax2, xy, new_label_ids, new_state, "After relabel", plane, is_3d=is_3d)
    if sc1 is not None:
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label="old label")
    if sc2 is not None:
        fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04, label="new label")
    fig.suptitle(f"Relabel labels ({source})")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"relabel_labels_before_after.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)

    return saved

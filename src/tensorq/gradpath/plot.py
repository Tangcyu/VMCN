from __future__ import annotations

import os
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from .cluster import PathCluster
from .selection import ChannelSelection
from .shooting import GradientPath

# ── global dpi for output ─────────────────────────────────────────────
DPI = 200

# ── number of colour-bar levels ───────────────────────────────────────
N_CBAR_LEVELS = 10

# ── max sampled points for background scatter ─────────────────────────
MAX_BG_POINTS = 160000

# ── colour palettes for state pairs / pathways ────────────────────────
PAIR_CMAP = plt.get_cmap("tab10")

PAIR_CMAP_NAMES = [
    "Blues",
    "Oranges",
    "Greens",
    "Purples",
    "Reds",
    "YlGnBu",
    "YlOrRd",
    "PuBuGn",
    "BuPu",
    "GnBu",
]

# ── CV label map for math formatting ──────────────────────────────────
CV_LABEL_MAP = {
    "phi": r"$\phi$",
    "psi": r"$\psi$",
    "phi1": r"$\phi_1$",
    "phi2": r"$\phi_2$",
    "phi3": r"$\phi_3$",
    "X": r"$X$",
    "Y": r"$Y$",
    "sphi": r"$\sin\phi$",
    "cphi": r"$\cos\phi$",
    "spsi": r"$\sin\psi$",
    "cpsi": r"$\cos\psi$",
    "sphi1": r"$\sin\phi_1$",
    "cphi1": r"$\cos\phi_1$",
    "sphi2": r"$\sin\phi_2$",
    "cphi2": r"$\cos\phi_2$",
    "sphi3": r"$\sin\phi_3$",
    "cphi3": r"$\cos\phi_3$",
}


# ── style infrastructure ─────────────────────────────────────────────

def cv_label(raw: str, config: dict | None = None) -> str:
    """Return publication label for a raw CV name."""
    if config is not None:
        overrides = config.get("cv_labels", {})
        if raw in overrides:
            return overrides[raw]
    if raw in CV_LABEL_MAP:
        return CV_LABEL_MAP[raw]
    return raw


def cm2inch(value: float) -> float:
    return float(value) / 2.54


def plot_style(config: dict | None = None) -> dict:
    """Shared sizing/font defaults for publication-quality plots."""
    if config is None:
        config = {}
    return {
        "figsize_cm": float(config.get("figsize_cm", 3.5)),
        "fig_aspect": float(config.get("fig_aspect", 1.2)),
        "font_size": float(config.get("font_size", 7)),
        "label_font_size": float(config.get("label_font_size", 7)),
        "title_font_size": float(config.get("title_font_size", config.get("font_size", 7))),
        "line_width": float(config.get("line_width", 0.4)),
        "colorbar_tick_font_size": float(config.get("colorbar_tick_font_size", config.get("font_size", 7))),
        "legend_font_size": float(config.get("legend_font_size", config.get("font_size", 7))),
    }


def apply_plot_style(config: dict | None = None):
    """Apply matplotlib rcParams for publication-quality output."""
    if config is None:
        config = {}
    style = plot_style(config)
    font_path = config.get("font_path", "")
    if font_path and os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        rcParams.update({"font.sans-serif": prop.get_name(), "font.family": "sans-serif"})

    rcParams.update(
        {
            "text.usetex": False,
            "axes.labelsize": style["label_font_size"],
            "axes.linewidth": style["line_width"],
            "font.size": style["font_size"],
            "xtick.labelsize": style["font_size"],
            "ytick.labelsize": style["font_size"],
            "axes.unicode_minus": False,
        }
    )


def figure_size(config: dict | None = None) -> tuple[float, float]:
    style = plot_style(config)
    side = cm2inch(style["figsize_cm"])
    return side, side / style["fig_aspect"]


def new_2d_figure(config: dict | None = None):
    return plt.subplots(figsize=figure_size(config))


def new_3d_figure(config: dict | None = None):
    return plt.figure(figsize=figure_size(config))


def title_size(config: dict | None = None) -> float:
    return plot_style(config)["title_font_size"]


def discrete_cmap_norm(
    cmap_name: str,
    vmin: float,
    vmax: float,
    levels: int = N_CBAR_LEVELS,
    color_window: tuple[float, float] = (0.0, 1.0),
) -> tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    """Return a colormap/norm where level slices are real color bins."""
    levels = max(2, int(levels))
    boundaries = np.linspace(vmin, vmax, levels + 1)
    base = plt.get_cmap(cmap_name)
    lo, hi = color_window
    samples = np.linspace(lo, hi, levels)
    cmap = ListedColormap(base(samples), name=f"{base.name}_{levels}_levels")
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm, boundaries


def colorbar_ticks(boundaries: np.ndarray, max_ticks: int = 6) -> np.ndarray:
    """Readable ticks for a discrete colorbar without defining its slices."""
    if boundaries.size <= max_ticks:
        return boundaries
    idx = np.linspace(0, boundaries.size - 1, max_ticks).round().astype(int)
    return boundaries[np.unique(idx)]


def add_discrete_colorbar(
    fig, ax, mappable, boundaries: np.ndarray, label: str, shrink: float = 0.82, pad: float = 0.02
):
    cb = fig.colorbar(
        mappable,
        ax=ax,
        shrink=shrink,
        pad=pad,
        boundaries=boundaries,
        ticks=colorbar_ticks(boundaries),
        spacing="proportional",
    )
    cb.set_label(label)
    cb.ax.tick_params(labelsize=rcParams["font.size"], width=rcParams["axes.linewidth"])
    return cb


def pathway_gradient_cmap(pathway_idx: int, config: dict | None = None):
    """Return a distinct gradient colormap for a single pathway.

    Different pathways within the same reaction pair each get their own
    colormap so they can be distinguished in the plot.
    """
    if config is None:
        config = {}
    cmap_spec = config.get("pathway_pair_cmaps", None)
    if isinstance(cmap_spec, list) and cmap_spec:
        cmap_name = cmap_spec[pathway_idx % len(cmap_spec)]
    else:
        cmap_name = PAIR_CMAP_NAMES[pathway_idx % len(PAIR_CMAP_NAMES)]

    vmin = float(config.get("pathway_vmin", 0.0))
    vmax = float(config.get("pathway_vmax", config.get("separatrix_vmax", 0.25)))
    color_min = float(config.get("pathway_cmap_min", 0.25))
    return discrete_cmap_norm(
        cmap_name, vmin, vmax, config.get("pathway_color_levels", N_CBAR_LEVELS), (color_min, 1.0)
    )


# ── axis helpers ─────────────────────────────────────────────────────


def _axis_indices_nd(names: Sequence[str] | None, axes: Sequence[int | str], ndim: int) -> tuple[int, ...]:
    if len(axes) != int(ndim):
        raise ValueError(f"plot axes must contain exactly {int(ndim)} entries.")
    out = []
    for axis in axes:
        if isinstance(axis, str):
            if names is None or axis not in names:
                raise ValueError(f"Unknown CV/feature axis {axis!r}.")
            out.append(list(names).index(axis))
        else:
            out.append(int(axis))
    return tuple(out)


def _axis_indices(names: Sequence[str] | None, axes: Sequence[int | str]) -> tuple[int, int]:
    x_idx, y_idx = _axis_indices_nd(names, axes, 2)
    return x_idx, y_idx


def _thin_indices(n_points: int, max_points: int) -> np.ndarray:
    max_points = int(max_points)
    if max_points <= 0 or int(n_points) <= max_points:
        return np.arange(int(n_points), dtype=np.int64)
    return np.linspace(0, int(n_points) - 1, max_points, dtype=np.int64)


def _add_3d_subplot(fig: plt.Figure):
    try:
        return fig.add_subplot(111, projection="3d", computed_zorder=False)
    except (AttributeError, TypeError):
        return fig.add_subplot(111, projection="3d")


def _apply_3d_axis_limits(ax, config: dict | None) -> None:
    if config is None:
        return
    for setter, key in ((ax.set_xlim, "xlim"), (ax.set_ylim, "ylim"), (ax.set_zlim, "zlim")):
        value = config.get(key, None)
        if value is not None:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{key} must be null or a two-value list.")
            setter(float(value[0]), float(value[1]))


def _periodic_axis_periods(
    axis_names: Sequence[str] | None,
    axes: Sequence[int | str],
    config: dict | None,
) -> list[float | None]:
    if config is None:
        return [None] * len(axes)
    periodic_cfg = config.get("periodic_plot_axes", config.get("periodic_cvs", []))
    if periodic_cfg is True:
        periodic_names = None
    elif periodic_cfg is False or periodic_cfg is None:
        periodic_names = set()
    elif isinstance(periodic_cfg, str):
        periodic_names = {periodic_cfg}
    else:
        periodic_names = {str(item) for item in periodic_cfg}
    units = str(config.get("periodic_cv_units", config.get("model_periodic_cv_units", "degrees"))).lower()
    default_period = 2.0 * np.pi if units in {"radians", "radian", "rad"} else 360.0
    limit_keys = ["xlim", "ylim", "zlim"]
    out: list[float | None] = []
    for pos, axis in enumerate(axes):
        axis_name = axis if isinstance(axis, str) else (axis_names[int(axis)] if axis_names is not None else None)
        is_periodic = periodic_names is None or (axis_name is not None and str(axis_name) in periodic_names)
        limit = config.get(limit_keys[pos], None) if pos < len(limit_keys) else None
        if is_periodic and isinstance(limit, (list, tuple)) and len(limit) == 2:
            out.append(float(limit[1]) - float(limit[0]))
        elif is_periodic:
            out.append(default_period)
        else:
            out.append(None)
    return out


def _break_periodic_line(values: np.ndarray, periods: Sequence[float | None]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2 or not any(period is not None and period > 0.0 for period in periods):
        return values
    jumps = np.zeros(values.shape[0] - 1, dtype=bool)
    for dim, period in enumerate(periods):
        if period is None or float(period) <= 0.0:
            continue
        jumps |= np.abs(np.diff(values[:, dim])) > 0.5 * float(period)
    if not bool(np.any(jumps)):
        return values
    rows = []
    for idx in range(values.shape[0] - 1):
        rows.append(values[idx])
        if jumps[idx]:
            rows.append(np.full(values.shape[1], np.nan, dtype=np.float64))
    rows.append(values[-1])
    return np.asarray(rows, dtype=np.float64)


def _legend_loc(config: dict | None) -> str:
    if config is None:
        return "upper right"
    return str(config.get("path_legend_loc", config.get("legend_loc", "upper right")))


def _format_axes_2d(ax, xlabel: str, ylabel: str, config: dict | None = None):
    """Apply consistent 2D axis formatting."""
    style = plot_style(config)
    ax.set_xlabel(xlabel, fontsize=style["label_font_size"])
    ax.set_ylabel(ylabel, fontsize=style["label_font_size"])
    ax.tick_params(labelsize=style["font_size"], width=style["line_width"])
    if config and config.get("xlim"):
        ax.set_xlim(*config["xlim"])
    if config and config.get("ylim"):
        ax.set_ylim(*config["ylim"])


# ── split periodic segments for coloured paths ───────────────────────


def split_periodic_segments(
    points: np.ndarray, periods: dict[int, float], values: np.ndarray | None = None
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Split a path and keep any per-point values aligned with each segment."""
    if not periods or points.shape[0] < 2:
        return [(points, values)]

    segments = []
    current_pts = [points[0]]
    current_vals = [values[0]] if values is not None and len(values) == points.shape[0] else None

    for i in range(1, points.shape[0]):
        broken = False
        for dim, period in periods.items():
            if dim < points.shape[1] and abs(points[i, dim] - points[i - 1, dim]) > period * 0.5:
                broken = True
                break
        if broken:
            if len(current_pts) >= 2:
                seg_vals = np.array(current_vals) if current_vals is not None else None
                segments.append((np.array(current_pts), seg_vals))
            current_pts = [points[i]]
            current_vals = [values[i]] if values is not None and len(values) == points.shape[0] else None
        else:
            current_pts.append(points[i])
            if current_vals is not None:
                current_vals.append(values[i])

    if len(current_pts) >= 2:
        seg_vals = np.array(current_vals) if current_vals is not None else None
        segments.append((np.array(current_pts), seg_vals))
    return segments if segments else [(points, values)]


def _projection_periods(periods_list: list[float | None], axes: tuple[int, ...]) -> dict[int, float]:
    """Convert a list of per-axis periods to {proj_idx: period} dict.

    periods_list is aligned with the plotting axes (same length, same order).
    Returns {0: period_x, 1: period_y, ...} mapping projected dimension to its period.
    """
    return {proj_idx: periods_list[proj_idx] for proj_idx in range(len(axes))
            if proj_idx < len(periods_list) and periods_list[proj_idx] is not None}


def _add_colored_path_2d(ax, points, values, periods, cmap, norm, linewidth, fallback_color):
    for seg, seg_values in split_periodic_segments(points, periods, values):
        if seg.shape[0] < 2:
            continue
        segments = np.stack([seg[:-1], seg[1:]], axis=1)
        if seg_values is not None and len(seg_values) == seg.shape[0]:
            segment_values = 0.5 * (seg_values[:-1] + seg_values[1:])
            lc = LineCollection(
                segments, cmap=cmap, norm=norm, linewidths=linewidth,
                capstyle="round", joinstyle="round", alpha=0.95,
            )
            lc.set_array(segment_values)
            ax.add_collection(lc)
        else:
            ax.plot(seg[:, 0], seg[:, 1], color=fallback_color, linewidth=linewidth, alpha=0.85)


def _add_colored_path_3d(ax, points, values, periods, cmap, norm, linewidth, fallback_color):
    for seg, seg_values in split_periodic_segments(points, periods, values):
        if seg.shape[0] < 2:
            continue
        segments = np.stack([seg[:-1], seg[1:]], axis=1)
        if seg_values is not None and len(seg_values) == seg.shape[0]:
            segment_values = 0.5 * (seg_values[:-1] + seg_values[1:])
            lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth, alpha=0.95)
            lc.set_array(segment_values)
            ax.add_collection(lc)
        else:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=fallback_color, linewidth=linewidth, alpha=0.85)


# ── plot functions ───────────────────────────────────────────────────


def plot_selected_points(
    coords: np.ndarray,
    selection: ChannelSelection,
    *,
    axes: Sequence[int | str] = (0, 1),
    axis_names: Sequence[str] | None = None,
    save_path: str | os.PathLike[str] | None = None,
    background_weights: np.ndarray | None = None,
    config: dict | None = None,
) -> plt.Figure:
    """Plot selected channel points on a 2D CV/feature plane."""

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("coords must have shape (n_samples, n_dim).")
    x_idx, y_idx = _axis_indices(axis_names, axes)

    cfg = {} if config is None else dict(config)
    apply_plot_style(cfg)
    fig, ax = new_2d_figure(cfg)

    if background_weights is None:
        alpha = 0.12
        sizes = 4.0
    else:
        w = np.asarray(background_weights, dtype=np.float64)
        w = w / max(float(np.nanmax(w)), 1e-300)
        alpha = 0.20
        sizes = 4.0 + 14.0 * w

    ax.scatter(coords[:, x_idx], coords[:, y_idx], s=sizes, c="0.75", alpha=alpha, linewidths=0, rasterized=True)

    selected = coords[selection.indices]
    score = selection.channel_score

    sep_cmap_name = str(cfg.get("separatrix_cmap", "magma"))
    vmax = float(cfg.get("separatrix_vmax", 0.25))
    cmap, norm, boundaries = discrete_cmap_norm(sep_cmap_name, 0.0, vmax)

    points = ax.scatter(
        selected[:, x_idx], selected[:, y_idx],
        c=score, s=22.0, cmap=cmap, norm=norm,
        edgecolors="black", linewidths=0.25,
    )

    add_discrete_colorbar(
        fig, ax, points, boundaries,
        f"$q_{{{selection.state_i}}} q_{{{selection.state_j}}}$",
    )

    xlabel = cv_label(str(axis_names[x_idx]), cfg) if axis_names is not None else f"axis {x_idx}"
    ylabel = cv_label(str(axis_names[y_idx]), cfg) if axis_names is not None else f"axis {y_idx}"
    _format_axes_2d(ax, xlabel, ylabel, cfg)
    ax.set_title(
        f"Selected {selection.state_i}-{selection.state_j} channel points",
        fontsize=title_size(cfg),
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig


def plot_paths_2d(
    paths: list[GradientPath] | np.ndarray,
    *,
    axes: Sequence[int | str] = (0, 1),
    axis_names: Sequence[str] | None = None,
    clusters: list[PathCluster] | None = None,
    background: np.ndarray | None = None,
    selected: ChannelSelection | None = None,
    save_path: str | os.PathLike[str] | None = None,
    config: dict | None = None,
) -> plt.Figure:
    """Plot sampled pathways and, optionally, weighted cluster centers."""

    path_array = np.asarray(paths if isinstance(paths, np.ndarray) else [p.path for p in paths], dtype=np.float64)
    if path_array.ndim != 3:
        raise ValueError("paths must have shape (n_paths, n_images, n_dim).")
    x_idx, y_idx = _axis_indices(axis_names, axes)
    periods = _periodic_axis_periods(axis_names, axes, config)

    cfg = {} if config is None else dict(config)
    apply_plot_style(cfg)
    fig, ax = new_2d_figure(cfg)

    if background is not None:
        bg = np.asarray(background, dtype=np.float64)
        n_bg = min(bg.shape[0], MAX_BG_POINTS)
        if bg.shape[0] > n_bg:
            rng = np.random.default_rng(42)
            bg = bg[rng.choice(bg.shape[0], n_bg, replace=False)]
        ax.scatter(bg[:, x_idx], bg[:, y_idx], s=2.0, c="0.82", alpha=0.10, linewidths=0, rasterized=True)

    if selected is not None and background is not None:
        pts = np.asarray(background, dtype=np.float64)[selected.indices]
        ax.scatter(pts[:, x_idx], pts[:, y_idx], s=12.0, c="black", alpha=0.35, linewidths=0)

    for path in path_array:
        xy = _break_periodic_line(path[:, [x_idx, y_idx]], periods)
        ax.plot(xy[:, 0], xy[:, 1], color="0.55", alpha=0.35, linewidth=0.9)

    if clusters:
        cmap = plt.get_cmap("tab10")
        for cluster in clusters:
            color = cmap((cluster.label - 1) % 10)
            center = cluster.center_path
            xy = _break_periodic_line(center[:, [x_idx, y_idx]], periods)
            ax.plot(
                xy[:, 0], xy[:, 1], color=color, linewidth=2.2,
                label=f"path {cluster.label} (n={cluster.member_indices.size})",
            )

    if clusters:
        ax.legend(loc=_legend_loc(cfg), frameon=False, fontsize=plot_style(cfg)["legend_font_size"])

    xlabel = cv_label(str(axis_names[x_idx]), cfg) if axis_names is not None else f"axis {x_idx}"
    ylabel = cv_label(str(axis_names[y_idx]), cfg) if axis_names is not None else f"axis {y_idx}"
    _format_axes_2d(ax, xlabel, ylabel, cfg)
    ax.set_title("Gradient pathways", fontsize=title_size(cfg))

    if save_path is not None:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig


def plot_selected_points_3d(
    coords: np.ndarray,
    selection: ChannelSelection,
    *,
    axes: Sequence[int | str] = (0, 1, 2),
    axis_names: Sequence[str] | None = None,
    save_path: str | os.PathLike[str] | None = None,
    background_weights: np.ndarray | None = None,
    config: dict | None = None,
) -> plt.Figure:
    """Plot selected channel points on a 3D CV/feature projection."""

    cfg = {} if config is None else dict(config)
    apply_plot_style(cfg)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("coords must have shape (n_samples, n_dim).")
    x_idx, y_idx, z_idx = _axis_indices_nd(axis_names, axes, 3)

    ids = _thin_indices(coords.shape[0], int(cfg.get("max_3d_points", 40000)))
    fig = new_3d_figure(cfg)
    ax = _add_3d_subplot(fig)

    if background_weights is None:
        sizes = float(cfg.get("3d_point_size", 2.0))
        colors = "0.75"
    else:
        w = np.asarray(background_weights, dtype=np.float64)
        w = w / max(float(np.nanmax(w)), 1e-300)
        sizes = float(cfg.get("3d_point_size", 2.0)) * (1.0 + 3.0 * w[ids])
        colors = "0.75"

    ax.scatter(
        coords[ids, x_idx], coords[ids, y_idx], coords[ids, z_idx],
        s=sizes, c=colors, alpha=float(cfg.get("3d_alpha", 0.35)),
        linewidths=0,
        depthshade=bool(cfg.get("3d_depthshade", False)),
        zorder=1,
    )

    selected = coords[selection.indices]
    sep_cmap_name = str(cfg.get("separatrix_cmap", "magma"))
    vmax = float(cfg.get("separatrix_vmax", 0.25))
    cmap, norm, boundaries = discrete_cmap_norm(sep_cmap_name, 0.0, vmax)

    points = ax.scatter(
        selected[:, x_idx], selected[:, y_idx], selected[:, z_idx],
        c=selection.channel_score, s=float(cfg.get("selected_3d_point_size", 24.0)),
        cmap=cmap, norm=norm, edgecolors="black", linewidths=0.25,
        depthshade=False, zorder=30,
    )

    add_discrete_colorbar(
        fig, ax, points, boundaries,
        f"$q_{{{selection.state_i}}} q_{{{selection.state_j}}}$",
        shrink=0.75, pad=0.12,
    )

    labels = [
        cv_label(str(axis_names[idx]), cfg) if axis_names is not None else f"axis {idx}"
        for idx in (x_idx, y_idx, z_idx)
    ]
    ax.set_xlabel(labels[0], fontsize=plot_style(cfg)["label_font_size"])
    ax.set_ylabel(labels[1], fontsize=plot_style(cfg)["label_font_size"])
    ax.set_zlabel(labels[2], fontsize=plot_style(cfg)["label_font_size"])
    ax.tick_params(labelsize=plot_style(cfg)["font_size"], width=plot_style(cfg)["line_width"])
    ax.set_title(
        f"Selected {selection.state_i}-{selection.state_j} channel points",
        fontsize=title_size(cfg),
    )
    _apply_3d_axis_limits(ax, cfg)
    ax.view_init(elev=float(cfg.get("3d_elev", 24.0)), azim=float(cfg.get("3d_azim", -60.0)))

    if save_path is not None:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig


def plot_paths_3d(
    paths: list[GradientPath] | np.ndarray,
    *,
    axes: Sequence[int | str] = (0, 1, 2),
    axis_names: Sequence[str] | None = None,
    clusters: list[PathCluster] | None = None,
    background: np.ndarray | None = None,
    selected: ChannelSelection | None = None,
    save_path: str | os.PathLike[str] | None = None,
    config: dict | None = None,
) -> plt.Figure:
    """Plot sampled pathways and weighted cluster centers in 3D."""

    cfg = {} if config is None else dict(config)
    apply_plot_style(cfg)
    path_array = np.asarray(paths if isinstance(paths, np.ndarray) else [p.path for p in paths], dtype=np.float64)
    if path_array.ndim != 3:
        raise ValueError("paths must have shape (n_paths, n_images, n_dim).")
    x_idx, y_idx, z_idx = _axis_indices_nd(axis_names, axes, 3)
    periods = _periodic_axis_periods(axis_names, axes, cfg)

    fig = new_3d_figure(cfg)
    ax = _add_3d_subplot(fig)

    if background is not None:
        bg = np.asarray(background, dtype=np.float64)
        ids = _thin_indices(bg.shape[0], int(cfg.get("max_3d_points", 40000)))
        ax.scatter(
            bg[ids, x_idx], bg[ids, y_idx], bg[ids, z_idx],
            s=float(cfg.get("3d_point_size", 2.0)),
            c="0.82", alpha=float(cfg.get("3d_alpha", 0.18)),
            linewidths=0,
            depthshade=bool(cfg.get("3d_depthshade", False)),
            zorder=1,
        )

    if selected is not None and background is not None:
        pts = np.asarray(background, dtype=np.float64)[selected.indices]
        ax.scatter(
            pts[:, x_idx], pts[:, y_idx], pts[:, z_idx],
            s=12.0, c="black", alpha=0.35,
            linewidths=0, depthshade=False, zorder=30,
        )

    for path in path_array:
        xyz = _break_periodic_line(path[:, [x_idx, y_idx, z_idx]], periods)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.55", alpha=0.35, linewidth=0.9, zorder=20)

    if clusters:
        cmap = plt.get_cmap("tab10")
        for cluster in clusters:
            color = cmap((cluster.label - 1) % 10)
            center = cluster.center_path
            xyz = _break_periodic_line(center[:, [x_idx, y_idx, z_idx]], periods)
            ax.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=color, linewidth=2.2,
                label=f"path {cluster.label} (n={cluster.member_indices.size})",
                zorder=25,
            )

    if clusters:
        ax.legend(loc=_legend_loc(cfg), frameon=False, fontsize=plot_style(cfg)["legend_font_size"])

    labels = [
        cv_label(str(axis_names[idx]), cfg) if axis_names is not None else f"axis {idx}"
        for idx in (x_idx, y_idx, z_idx)
    ]
    ax.set_xlabel(labels[0], fontsize=plot_style(cfg)["label_font_size"])
    ax.set_ylabel(labels[1], fontsize=plot_style(cfg)["label_font_size"])
    ax.set_zlabel(labels[2], fontsize=plot_style(cfg)["label_font_size"])
    ax.tick_params(labelsize=plot_style(cfg)["font_size"], width=plot_style(cfg)["line_width"])
    ax.set_title("Gradient pathways", fontsize=title_size(cfg))
    _apply_3d_axis_limits(ax, cfg)
    ax.view_init(elev=float(cfg.get("3d_elev", 24.0)), azim=float(cfg.get("3d_azim", -60.0)))

    if save_path is not None:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig


def plot_path_dendrogram(
    linkage: np.ndarray,
    *,
    distance_threshold: float | None = None,
    save_path: str | os.PathLike[str] | None = None,
    config: dict | None = None,
) -> plt.Figure:
    """Plot a path-clustering dendrogram from a SciPy-style linkage matrix."""

    cfg = {} if config is None else dict(config)
    apply_plot_style(cfg)
    linkage = np.asarray(linkage, dtype=np.float64)
    n_paths = linkage.shape[0] + 1 if linkage.size else 1
    style = plot_style(cfg)
    fig_w = cm2inch(style["figsize_cm"]) * max(1.0, 0.22 * n_paths / cm2inch(style["figsize_cm"]))
    fig, ax = plt.subplots(figsize=(fig_w, figure_size(cfg)[1]))
    ax.tick_params(labelsize=style["font_size"], width=style["line_width"])

    if linkage.size == 0:
        ax.text(0.5, 0.5, "one path", ha="center", va="center", transform=ax.transAxes, fontsize=style["font_size"])
        ax.set_xticks([])
        ax.set_ylabel("weighted RMSD", fontsize=style["label_font_size"])
        if save_path is not None:
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        return fig

    x_pos: dict[int, float] = {idx: float(idx) for idx in range(n_paths)}
    y_pos: dict[int, float] = {idx: 0.0 for idx in range(n_paths)}
    counts: dict[int, float] = {idx: 1.0 for idx in range(n_paths)}
    for merge_idx, row in enumerate(linkage):
        left = int(row[0])
        right = int(row[1])
        height = float(row[2])
        new_id = n_paths + merge_idx
        xl, xr = x_pos[left], x_pos[right]
        yl, yr = y_pos[left], y_pos[right]
        ax.plot([xl, xl], [yl, height], color="black", linewidth=style["line_width"])
        ax.plot([xr, xr], [yr, height], color="black", linewidth=style["line_width"])
        ax.plot([xl, xr], [height, height], color="black", linewidth=style["line_width"])
        total = counts[left] + counts[right]
        x_pos[new_id] = (x_pos[left] * counts[left] + x_pos[right] * counts[right]) / total
        y_pos[new_id] = height
        counts[new_id] = total

    if distance_threshold is not None:
        ax.axhline(float(distance_threshold), color="tab:red", linestyle="--", linewidth=style["line_width"])

    ax.set_xlabel("path", fontsize=style["label_font_size"])
    ax.set_ylabel("weighted RMSD", fontsize=style["label_font_size"])
    ax.set_xlim(-0.5, n_paths - 0.5)
    ax.set_xticks(np.arange(n_paths))
    if n_paths > 50:
        ax.set_xticklabels([])
    ax.set_title("Path clustering dendrogram", fontsize=title_size(cfg))

    if save_path is not None:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig


def plot_colored_paths_2d(
    paths: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    *,
    axes: Sequence[int | str] = (0, 1),
    axis_names: Sequence[str] | None = None,
    background_points: np.ndarray | None = None,
    save_path: str | os.PathLike[str] | None = None,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    linewidth: float = 2.6,
    background_point_size: float = 4.0,
    background_point_alpha: float = 0.18,
    label_paths: bool = True,
    title: str = "Gradient pathways",
    config: dict | None = None,
) -> plt.Figure:
    """Plot pathways colored by a scalar value along each path."""

    if len(paths) == 0:
        raise ValueError("Need at least one path to plot.")
    if len(paths) != len(scores):
        raise ValueError("paths and scores must have the same length.")
    path_arrays = [np.asarray(path, dtype=np.float64) for path in paths]
    score_arrays = [np.asarray(score, dtype=np.float64) for score in scores]
    for path, score in zip(path_arrays, score_arrays):
        if path.ndim != 2:
            raise ValueError("Each path must have shape (n_images, n_dim).")
        if score.ndim != 1 or score.shape[0] != path.shape[0]:
            raise ValueError("Each score array must have shape (n_images,).")

    cfg = {} if config is None else dict(config)
    apply_plot_style(cfg)

    x_idx, y_idx = _axis_indices(axis_names, axes)
    periods = _periodic_axis_periods(axis_names, axes, cfg)
    proj_periods = _projection_periods(periods, (x_idx, y_idx))

    finite_scores = np.concatenate([score[np.isfinite(score)] for score in score_arrays])
    if finite_scores.size == 0:
        raise ValueError("Cannot plot colored paths: all scores are non-finite.")
    color_max = float(vmax) if vmax is not None else float(np.nanmax(finite_scores))
    color_min = float(vmin) if vmin is not None else float(np.nanmin(finite_scores))
    if color_max <= color_min:
        color_max = color_min + 1e-12

    cmap_name = str(cfg.get("colored_path_cmap", cmap))
    levels = int(cfg.get("colored_path_color_levels", N_CBAR_LEVELS))
    disc_cmap, disc_norm, boundaries = discrete_cmap_norm(cmap_name, color_min, color_max, levels)

    fig, ax = new_2d_figure(cfg)

    if background_points is not None:
        background_array = np.asarray(background_points, dtype=np.float64)
        if background_array.ndim != 2:
            raise ValueError("background_points must have shape (n_points, n_dim).")
        n_bg = min(background_array.shape[0], MAX_BG_POINTS)
        if background_array.shape[0] > n_bg:
            rng = np.random.default_rng(42)
            bg_ids = rng.choice(background_array.shape[0], n_bg, replace=False)
            bg_plot = background_array[bg_ids]
        else:
            bg_plot = background_array
        ax.scatter(
            bg_plot[:, x_idx], bg_plot[:, y_idx],
            s=float(background_point_size), c="0.55",
            alpha=float(background_point_alpha), linewidths=0,
            zorder=1, rasterized=True,
        )

    last_collection = None
    for idx, (path, score) in enumerate(zip(path_arrays, score_arrays), start=1):
        xy = path[:, [x_idx, y_idx]]
        if xy.shape[0] < 2:
            continue
        tube = np.clip(score, color_min, color_max)
        _add_colored_path_2d(
            ax, xy, tube, proj_periods, disc_cmap, disc_norm,
            linewidth, disc_cmap(disc_norm(0.72 * color_max)),
        )
        # endpoint markers
        ax.scatter(
            [xy[0, 0], xy[-1, 0]], [xy[0, 1], xy[-1, 1]],
            c=[tube[0], tube[-1]], cmap=disc_cmap, norm=disc_norm,
            s=12.0, edgecolors="black", linewidths=0.25, zorder=3,
        )
        if label_paths:
            ax.text(xy[-1, 0], xy[-1, 1], f" {idx}", fontsize=plot_style(cfg)["font_size"], va="center")

        last_collection = plt.cm.ScalarMappable(norm=disc_norm, cmap=disc_cmap)
        last_collection.set_array(finite_scores)

    if last_collection is None:
        raise ValueError("No path had enough points to plot.")

    xy_arrays = [path[:, [x_idx, y_idx]] for path in path_arrays]
    if background_points is not None:
        xy_arrays.append(np.asarray(background_points, dtype=np.float64)[:, [x_idx, y_idx]])
    all_xy = np.vstack(xy_arrays)
    pad_x = max(float(np.nanmax(all_xy[:, 0]) - np.nanmin(all_xy[:, 0])), 1e-12) * 0.05
    pad_y = max(float(np.nanmax(all_xy[:, 1]) - np.nanmin(all_xy[:, 1])), 1e-12) * 0.05
    ax.set_xlim(float(np.nanmin(all_xy[:, 0]) - pad_x), float(np.nanmax(all_xy[:, 0]) + pad_x))
    ax.set_ylim(float(np.nanmin(all_xy[:, 1]) - pad_y), float(np.nanmax(all_xy[:, 1]) + pad_y))

    if cfg.get("xlim"):
        ax.set_xlim(*cfg["xlim"])
    if cfg.get("ylim"):
        ax.set_ylim(*cfg["ylim"])

    add_discrete_colorbar(fig, ax, last_collection, boundaries, "$q_i q_j$")

    xlabel = cv_label(str(axis_names[x_idx]), cfg) if axis_names is not None else f"axis {x_idx}"
    ylabel = cv_label(str(axis_names[y_idx]), cfg) if axis_names is not None else f"axis {y_idx}"
    _format_axes_2d(ax, xlabel, ylabel, cfg)
    ax.set_title(title, fontsize=title_size(cfg))

    if save_path is not None:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig


def plot_colored_paths_3d(
    paths: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    *,
    axes: Sequence[int | str] = (0, 1, 2),
    axis_names: Sequence[str] | None = None,
    background_points: np.ndarray | None = None,
    save_path: str | os.PathLike[str] | None = None,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    linewidth: float = 2.6,
    background_point_size: float = 4.0,
    background_point_alpha: float = 0.18,
    label_paths: bool = True,
    title: str = "Gradient pathways",
    config: dict | None = None,
) -> plt.Figure:
    """Plot 3D pathways colored by a scalar value along each path."""

    if len(paths) == 0:
        raise ValueError("Need at least one path to plot.")
    if len(paths) != len(scores):
        raise ValueError("paths and scores must have the same length.")
    cfg = {} if config is None else dict(config)
    apply_plot_style(cfg)

    path_arrays = [np.asarray(path, dtype=np.float64) for path in paths]
    score_arrays = [np.asarray(score, dtype=np.float64) for score in scores]
    for path, score in zip(path_arrays, score_arrays):
        if path.ndim != 2:
            raise ValueError("Each path must have shape (n_images, n_dim).")
        if score.ndim != 1 or score.shape[0] != path.shape[0]:
            raise ValueError("Each score array must have shape (n_images,).")

    x_idx, y_idx, z_idx = _axis_indices_nd(axis_names, axes, 3)
    periods = _periodic_axis_periods(axis_names, axes, cfg)
    proj_periods = _projection_periods(periods, (x_idx, y_idx, z_idx))

    finite_scores = np.concatenate([score[np.isfinite(score)] for score in score_arrays])
    if finite_scores.size == 0:
        raise ValueError("Cannot plot colored paths: all scores are non-finite.")
    color_max = float(vmax) if vmax is not None else float(np.nanmax(finite_scores))
    color_min = float(vmin) if vmin is not None else float(np.nanmin(finite_scores))
    if color_max <= color_min:
        color_max = color_min + 1e-12

    cmap_name = str(cfg.get("colored_path_cmap", cmap))
    levels = int(cfg.get("colored_path_color_levels", N_CBAR_LEVELS))
    disc_cmap, disc_norm, boundaries = discrete_cmap_norm(cmap_name, color_min, color_max, levels)

    fig = new_3d_figure(cfg)
    ax = _add_3d_subplot(fig)

    if background_points is not None:
        background_array = np.asarray(background_points, dtype=np.float64)
        if background_array.ndim != 2:
            raise ValueError("background_points must have shape (n_points, n_dim).")
        ids = _thin_indices(background_array.shape[0], int(cfg.get("max_3d_points", 40000)))
        ax.scatter(
            background_array[ids, x_idx],
            background_array[ids, y_idx],
            background_array[ids, z_idx],
            s=float(background_point_size), c="0.55",
            alpha=float(background_point_alpha), linewidths=0,
            depthshade=bool(cfg.get("3d_depthshade", False)),
            zorder=1,
        )

    last_collection = None
    for idx, (path, score) in enumerate(zip(path_arrays, score_arrays), start=1):
        xyz = path[:, [x_idx, y_idx, z_idx]]
        if xyz.shape[0] < 2:
            continue
        tube = np.clip(score, color_min, color_max)
        _add_colored_path_3d(
            ax, xyz, tube, proj_periods, disc_cmap, disc_norm,
            linewidth, disc_cmap(disc_norm(0.72 * color_max)),
        )
        ax.scatter(
            [xyz[0, 0], xyz[-1, 0]],
            [xyz[0, 1], xyz[-1, 1]],
            [xyz[0, 2], xyz[-1, 2]],
            c=[tube[0], tube[-1]], cmap=disc_cmap, norm=disc_norm,
            s=12.0, edgecolors="black", linewidths=0.25,
            depthshade=False, zorder=30,
        )
        if label_paths:
            ax.text(
                xyz[-1, 0], xyz[-1, 1], xyz[-1, 2],
                f" {idx}", fontsize=plot_style(cfg)["font_size"], va="center", zorder=31,
            )

        last_collection = plt.cm.ScalarMappable(norm=disc_norm, cmap=disc_cmap)
        last_collection.set_array(finite_scores)

    if last_collection is None:
        raise ValueError("No path had enough points to plot.")

    xyz_arrays = [path[:, [x_idx, y_idx, z_idx]] for path in path_arrays]
    if background_points is not None:
        xyz_arrays.append(np.asarray(background_points, dtype=np.float64)[:, [x_idx, y_idx, z_idx]])
    all_xyz = np.vstack(xyz_arrays)
    spans = np.maximum(np.nanmax(all_xyz, axis=0) - np.nanmin(all_xyz, axis=0), 1e-12)
    mins = np.nanmin(all_xyz, axis=0) - 0.05 * spans
    maxs = np.nanmax(all_xyz, axis=0) + 0.05 * spans
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))

    labels = [
        cv_label(str(axis_names[idx]), cfg) if axis_names is not None else f"axis {idx}"
        for idx in (x_idx, y_idx, z_idx)
    ]
    style = plot_style(cfg)
    ax.set_xlabel(labels[0], fontsize=style["label_font_size"])
    ax.set_ylabel(labels[1], fontsize=style["label_font_size"])
    ax.set_zlabel(labels[2], fontsize=style["label_font_size"])
    ax.tick_params(labelsize=style["font_size"], width=style["line_width"])
    ax.set_title(title, fontsize=title_size(cfg))
    _apply_3d_axis_limits(ax, cfg)
    ax.view_init(elev=float(cfg.get("3d_elev", 24.0)), azim=float(cfg.get("3d_azim", -60.0)))

    add_discrete_colorbar(fig, ax, last_collection, boundaries, "$q_i q_j$", shrink=0.75, pad=0.12)

    if save_path is not None:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return fig

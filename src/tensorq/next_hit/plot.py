from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import (
    apply_stride,
    build_lagged_indices,
    cv_headers_for_pack,
    featurize_cv_inputs,
    infer_n_states,
    load_dataset,
    select_model_inputs,
)
from ..common.flux import make_thresholds, resolve_ordered_pairs, unordered_pairs
from .predict import infer_probabilities, load_committor_model
from .rate_constant import estimate_flux_profiles, resolve_lag_timing


def weighted_mean_2d(x, y, v, w, xedges, yedges):
    denom, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w)
    numer, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w * v)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = numer / denom
    avg[denom <= 0] = np.nan
    return avg


def state_names_for_plot(config: dict[str, Any], n_states: int) -> list[str]:
    names = config.get("state_names", None)
    if names is None:
        return [f"state {idx}" for idx in range(n_states)]
    out = [str(name) for name in names]
    if len(out) < n_states:
        out.extend(f"state {idx}" for idx in range(len(out), n_states))
    return out


def basin_colors(n_states: int, config: dict[str, Any]) -> list:
    cmap = plt.get_cmap(str(config.get("basin_cmap", "tab10")))
    return [cmap(idx % cmap.N) for idx in range(n_states)]


def basin_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    state: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    n_states: int,
) -> np.ndarray:
    counts = []
    for state_idx in range(n_states):
        mask = state == state_idx
        if np.any(mask):
            hist, _, _ = np.histogram2d(x[mask], y[mask], bins=[xedges, yedges])
        else:
            hist = np.zeros((len(xedges) - 1, len(yedges) - 1), dtype=np.float64)
        counts.append(hist)
    stack = np.stack(counts, axis=0)
    best = np.argmax(stack, axis=0).astype(float)
    best[np.max(stack, axis=0) <= 0] = np.nan
    return best


def add_basin_overlay_2d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    state: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    n_states: int,
    config: dict[str, Any],
) -> None:
    if not bool(config.get("plot_basin_overlay", True)):
        return
    field = basin_field_2d(x, y, state, xedges, yedges, n_states)
    if np.all(np.isnan(field)):
        return
    colors = basin_colors(n_states, config)
    cmap = ListedColormap(colors)
    ax.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_invalid(field).T,
        cmap=cmap,
        shading="auto",
        vmin=-0.5,
        vmax=max(0.5, n_states - 0.5),
        alpha=float(config.get("basin_overlay_alpha", 0.22)),
    )


def add_basin_overlay_3d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    state: np.ndarray,
    n_states: int,
    config: dict[str, Any],
) -> None:
    if not bool(config.get("plot_basin_overlay", True)):
        return
    max_points = int(config.get("basin_overlay_max_points", 20000))
    colors = basin_colors(n_states, config)
    names = state_names_for_plot(config, n_states)
    alpha = float(config.get("basin_overlay_alpha", 0.22))
    size = float(config.get("basin_overlay_point_size", 10.0))
    show_labels = bool(config.get("basin_overlay_labels", True))
    label_size = float(config.get("basin_overlay_label_size", 8.0))
    z_span = max(float(np.nanmax(z) - np.nanmin(z)), 1e-12)
    label_offset = float(config.get("basin_overlay_label_z_offset", 0.04)) * z_span
    for state_idx in range(n_states):
        state_ids = np.flatnonzero(state == state_idx)
        if state_ids.size == 0:
            continue
        ids = state_ids[thin_3d_points(state_ids.size, max_points // max(1, n_states))]
        ax.scatter(
            x[ids],
            y[ids],
            z[ids],
            s=size,
            color=colors[state_idx],
            alpha=alpha,
            linewidths=0,
            depthshade=False,
            label=names[state_idx],
        )
        if show_labels:
            ax.text(
                float(np.nanmean(x[state_ids])),
                float(np.nanmean(y[state_ids])),
                float(np.nanmax(z[state_ids]) + label_offset),
                names[state_idx],
                color=colors[state_idx],
                fontsize=label_size,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    if bool(config.get("basin_overlay_legend", True)):
        ax.legend(frameon=False, fontsize=7, loc="upper left")


def reaction_tube_plot_threshold(config: dict[str, Any]) -> float | None:
    value = config.get("reaction_tube_plot_threshold", config.get("reaction_tube_threshold", None))
    return None if value is None else float(value)


def model_input_space(config: dict[str, Any]) -> str:
    return str(config.get("model_input_space", config.get("input_space", config.get("feature_space", "features")))).lower()


def list_or_empty(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def selected_model_cvs(config: dict[str, Any], cv_headers: list[str]) -> list[str]:
    return list_or_empty(config.get("cvs_to_use", config.get("model_cvs", []))) or list(cv_headers)


def should_center_evaluate_reaction_tube(config: dict[str, Any], plane: list[str], cv_headers: list[str]) -> bool:
    mode = str(config.get("reaction_tube_center_evaluate", "auto")).lower()
    if mode in {"false", "no", "off", "0"}:
        return False
    if model_input_space(config) not in {"cv", "cvs", "colvars"}:
        if mode in {"true", "yes", "on", "1"}:
            raise ValueError("reaction_tube_center_evaluate=true requires model_input_space='cv'.")
        return False
    model_cvs = selected_model_cvs(config, cv_headers)
    covered = set(model_cvs).issubset(set(plane))
    if mode in {"true", "yes", "on", "1"} and not covered:
        raise ValueError(
            "reaction_tube_center_evaluate=true requires the plotted plane to contain all model CVs: "
            f"missing {sorted(set(model_cvs) - set(plane))}"
        )
    return covered


def finite_minmax(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot infer axis limits from all-NaN/all-inf values.")
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def resolve_cv_axis_limits(
    arrays: list[np.ndarray],
    config: dict[str, Any],
) -> list[tuple[float, float]]:
    same_limits = bool(config.get("same_axis_limits", config.get("equal_axis_limits", False)))
    if same_limits:
        shared = config.get("axis_limits", config.get("xlim", None))
        if shared is not None:
            lo, hi = map(float, shared)
        else:
            lo = min(finite_minmax(arr)[0] for arr in arrays)
            hi = max(finite_minmax(arr)[1] for arr in arrays)
        return [(lo, hi) for _ in arrays]

    keys = ["xlim", "ylim", "zlim"]
    limits = []
    for arr, key in zip(arrays, keys):
        cfg = config.get(key, None)
        limits.append(tuple(map(float, cfg)) if cfg is not None else finite_minmax(arr))
    return limits


def normalize_planes(planes: Any) -> list[list[str]]:
    if not planes:
        return []
    if isinstance(planes, (list, tuple)) and planes and all(isinstance(item, str) for item in planes):
        planes = [planes]
    out = []
    for plane in planes:
        names = [str(name) for name in plane]
        if len(names) not in {2, 3}:
            raise ValueError("Each CV plane must contain exactly 2 or 3 CV names.")
        out.append(names)
    return out


def thin_3d_points(n_points: int, max_points: int) -> np.ndarray:
    max_points = int(max_points)
    if max_points <= 0 or n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    return np.linspace(0, n_points - 1, max_points, dtype=np.int64)


def select_reaction_tube_points(
    tube: np.ndarray,
    threshold: float | None,
    max_points: int,
    mode: str,
) -> np.ndarray:
    mask = np.ones_like(tube, dtype=bool) if threshold is None else tube > float(threshold)
    ids = np.flatnonzero(mask)
    max_points = int(max_points)
    if max_points <= 0 or ids.size <= max_points:
        return ids

    mode = str(mode).lower()
    if mode == "top":
        keep = np.argpartition(tube[ids], -max_points)[-max_points:]
        return ids[keep[np.argsort(tube[ids][keep])]]
    if mode == "random":
        rng = np.random.default_rng(0)
        return np.sort(rng.choice(ids, size=max_points, replace=False))
    if mode == "uniform":
        return ids[np.linspace(0, ids.size - 1, max_points, dtype=np.int64)]
    raise ValueError("reaction_tube_3d_point_selection must be one of: top, random, uniform.")


def reaction_network_colors(n_pairs: int, config: dict[str, Any]) -> list:
    if n_pairs <= 0:
        return []
    cmap = plt.get_cmap(str(config.get("reaction_tube_network_cmap", "turbo")))
    colors = [to_rgba(config.get("reaction_tube_network_start_color", "white"))]
    if n_pairs > 1:
        stops = np.linspace(0.08, 0.95, n_pairs - 1)
        colors.extend(cmap(float(stop)) for stop in stops)
    return colors


def plot_reaction_tube_network_2d(
    fields: list[np.ndarray],
    pairs: list[tuple[int, int]],
    *,
    x: np.ndarray,
    y: np.ndarray,
    basin_state: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    n_states: int,
    labels: tuple[str, str],
    out_path: str,
    config: dict[str, Any],
) -> bool:
    if not fields:
        return False
    stack = np.stack(fields, axis=0)
    valid = np.isfinite(stack)
    any_valid = np.any(valid, axis=0)
    if not np.any(any_valid):
        return False
    filled = np.where(valid, stack, -np.inf)
    winners = np.argmax(filled, axis=0).astype(float)
    winners[~any_valid] = np.nan

    colors = reaction_network_colors(len(pairs), config)
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
    add_basin_overlay_2d(ax, x, y, basin_state, xedges, yedges, n_states, config)
    ax.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_invalid(winners).T,
        cmap=cmap,
        shading="auto",
        vmin=-0.5,
        vmax=max(0.5, len(pairs) - 0.5),
        alpha=float(config.get("reaction_tube_network_alpha", 0.78)),
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=colors[idx],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=f"q_{i} q_{j}",
            markersize=6,
        )
        for idx, (i, j) in enumerate(pairs)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper right")
    ax.set_xlabel(str(labels[0]))
    ax.set_ylabel(str(labels[1]))
    ax.set_title("reactive network")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_reaction_tube_network_3d(
    tube_points: list[tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray]],
    *,
    labels: tuple[str, str, str],
    out_path: str,
    config: dict[str, Any],
    axis_limits: list[tuple[float, float]],
    basin_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    basin_state: np.ndarray,
    n_states: int,
) -> bool:
    tube_points = [item for item in tube_points if item[1].size > 0]
    if not tube_points:
        return False
    colors = reaction_network_colors(len(tube_points), config)
    fig = plt.figure(figsize=(5.8, 4.9), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    add_basin_overlay_3d(ax, *basin_xyz, basin_state, n_states, config)
    size = float(config.get("reaction_tube_network_3d_point_size", config.get("3d_point_size", 4.0)))
    alpha = float(config.get("reaction_tube_network_3d_alpha", config.get("reaction_tube_3d_alpha", 0.75)))
    handles = []
    for idx, ((i, j), px, py, pz) in enumerate(tube_points):
        color = colors[idx]
        ax.scatter(
            px,
            py,
            pz,
            s=size,
            color=to_rgba(color, alpha=alpha),
            linewidths=0.15 if idx == 0 else 0,
            edgecolors="black" if idx == 0 else "none",
            depthshade=bool(config.get("3d_depthshade", False)),
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                label=f"q_{i} q_{j}",
                markersize=5,
            )
        )
    ax.set_xlabel(str(labels[0]))
    ax.set_ylabel(str(labels[1]))
    ax.set_zlabel(str(labels[2]))
    ax.set_title("reactive network")
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])
    ax.view_init(elev=float(config.get("3d_elev", 24.0)), azim=float(config.get("3d_azim", -60.0)))
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def scatter_3d_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    *,
    weights: np.ndarray,
    labels: tuple[str, str, str],
    title: str,
    color_label: str,
    out_path: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    config: dict[str, Any],
    axis_limits: list[tuple[float, float]] | None = None,
    max_points: int | None = None,
    alpha: float | None = None,
    basin_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    basin_state: np.ndarray | None = None,
    n_states: int | None = None,
) -> None:
    ids = thin_3d_points(len(x), int(config.get("max_3d_points", 50000) if max_points is None else max_points))
    field_alpha = float(config.get("3d_alpha", 0.75) if alpha is None else alpha)
    size_scale = float(config.get("3d_weight_size_scale", 0.0))
    if size_scale > 0.0:
        w = weights[ids].astype(np.float64)
        w = w / (np.nanmax(w) + 1e-300)
        field_sizes = float(config.get("3d_point_size", 4.0)) * (1.0 + size_scale * w)
    else:
        field_sizes = np.full(ids.size, float(config.get("3d_point_size", 4.0)), dtype=np.float64)

    fig = plt.figure(figsize=(5.6, 4.8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    draw_basin = (
        basin_xyz is not None
        and basin_state is not None
        and n_states is not None
        and bool(config.get("plot_basin_overlay", True))
    )
    norm = Normalize(
        vmin=float(np.nanmin(values[ids])) if vmin is None and ids.size else vmin,
        vmax=float(np.nanmax(values[ids])) if vmax is None and ids.size else vmax,
    )
    cmap_obj = plt.get_cmap(cmap)
    field_colors = cmap_obj(norm(values[ids]))
    field_colors[:, 3] = field_alpha

    if draw_basin:
        bx, by, bz = basin_xyz
        basin_max_points = int(config.get("basin_overlay_max_points", 20000))
        basin_rgba = basin_colors(int(n_states), config)
        basin_alpha = float(config.get("basin_overlay_alpha", 0.22))
        basin_size = float(config.get("basin_overlay_point_size", 10.0))
        basin_x_parts = []
        basin_y_parts = []
        basin_z_parts = []
        basin_color_parts = []
        legend_handles = []
        names = state_names_for_plot(config, int(n_states))
        for state_idx in range(int(n_states)):
            state_ids = np.flatnonzero(basin_state == state_idx)
            if state_ids.size == 0:
                continue
            keep = state_ids[thin_3d_points(state_ids.size, basin_max_points // max(1, int(n_states)))]
            color = to_rgba(basin_rgba[state_idx], alpha=basin_alpha)
            basin_x_parts.append(bx[keep])
            basin_y_parts.append(by[keep])
            basin_z_parts.append(bz[keep])
            basin_color_parts.append(np.tile(color, (keep.size, 1)))
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=to_rgba(basin_rgba[state_idx], alpha=1.0),
                    label=names[state_idx],
                    markersize=5,
                )
            )

        if basin_x_parts:
            all_x = np.concatenate([np.concatenate(basin_x_parts), x[ids]])
            all_y = np.concatenate([np.concatenate(basin_y_parts), y[ids]])
            all_z = np.concatenate([np.concatenate(basin_z_parts), z[ids]])
            all_colors = np.vstack([np.vstack(basin_color_parts), field_colors])
            all_sizes = np.concatenate(
                [
                    np.full(sum(part.size for part in basin_x_parts), basin_size, dtype=np.float64),
                    field_sizes,
                ]
            )
            sc = ax.scatter(
                all_x,
                all_y,
                all_z,
                c=all_colors,
                s=all_sizes,
                linewidths=0,
                depthshade=bool(config.get("3d_depthshade", False)),
            )

            if bool(config.get("basin_overlay_labels", True)):
                label_size = float(config.get("basin_overlay_label_size", 8.0))
                z_span = max(float(np.nanmax(bz) - np.nanmin(bz)), 1e-12)
                label_offset = float(config.get("basin_overlay_label_z_offset", 0.04)) * z_span
                for state_idx in range(int(n_states)):
                    state_ids = np.flatnonzero(basin_state == state_idx)
                    if state_ids.size == 0:
                        continue
                    ax.text(
                        float(np.nanmean(bx[state_ids])),
                        float(np.nanmean(by[state_ids])),
                        float(np.nanmax(bz[state_ids]) + label_offset),
                        names[state_idx],
                        color=basin_rgba[state_idx],
                        fontsize=label_size,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
            if bool(config.get("basin_overlay_legend", True)) and legend_handles:
                ax.legend(handles=legend_handles, frameon=False, fontsize=7, loc="upper left")
        else:
            draw_basin = False

    if not draw_basin:
        sc = ax.scatter(
            x[ids],
            y[ids],
            z[ids],
            c=values[ids],
            s=field_sizes,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=field_alpha,
            linewidths=0,
            depthshade=bool(config.get("3d_depthshade", False)),
        )
    ax.set_xlabel(str(labels[0]))
    ax.set_ylabel(str(labels[1]))
    ax.set_zlabel(str(labels[2]))
    ax.set_title(title)
    if axis_limits is not None:
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_zlim(*axis_limits[2])
    ax.view_init(elev=float(config.get("3d_elev", 24.0)), azim=float(config.get("3d_azim", -60.0)))
    if draw_basin:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array(values[ids])
        cb = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.75)
    else:
        cb = fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.75)
    cb.set_label(color_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_q(config: dict[str, Any], pack, n_states: int) -> np.ndarray:
    q_path = config.get("Q_npy", config.get("q_npy", None))
    if q_path is not None:
        q = np.load(q_path).astype(np.float32)
    else:
        model_path = config.get("model", None)
        if model_path is None:
            raise KeyError("Plot config needs Q_npy or model.")
        device = setup_device(config.get("prediction_device", config.get("device", "cuda:0")))
        model = load_committor_model(model_path, device)
        model_features, _ = select_model_inputs(pack, config)
        q = infer_probabilities(model, model_features.float(), device, batch_size=int(config.get("batch_size", 65536)))
    if q.ndim != 2 or q.shape[1] != n_states:
        raise RuntimeError(f"q shape {q.shape} does not match n_states={n_states}.")
    return q


def load_model_for_center_evaluation(config: dict[str, Any]) -> tuple[torch.nn.Module, torch.device] | None:
    model_path = config.get("model", None)
    if model_path is None:
        return None
    device = setup_device(config.get("prediction_device", config.get("device", "cuda:0")))
    return load_committor_model(model_path, device), device


def predict_q_at_cv_points(
    config: dict[str, Any],
    cv_points: np.ndarray,
    cv_headers: list[str],
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    features, _ = featurize_cv_inputs(
        torch.as_tensor(cv_points, dtype=torch.float32),
        cv_headers,
        cvs_to_use=selected_model_cvs(config, cv_headers),
        periodic_cvs=config.get("periodic_cvs", config.get("periodic", False)),
        periodic_units=str(config.get("periodic_cv_units", config.get("cv_units", "degrees"))),
    )
    return infer_probabilities(model, features.float(), device, batch_size=int(config.get("batch_size", 65536)))


def cv_points_from_plane(
    plane: list[str],
    plane_values: list[np.ndarray],
    cv_headers: list[str],
    config: dict[str, Any],
) -> np.ndarray:
    n_points = int(plane_values[0].shape[0])
    points = np.zeros((n_points, len(cv_headers)), dtype=np.float32)
    fill_values = config.get("cv_center_fill_values", {}) or {}
    for idx, name in enumerate(cv_headers):
        if name in fill_values:
            points[:, idx] = float(fill_values[name])
    for name, values in zip(plane, plane_values):
        points[:, cv_headers.index(name)] = values.astype(np.float32)
    return points


def plot_q_distributions(q: np.ndarray, weights: np.ndarray, out_path: str, state_names=None, bins: int = 60) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for j in range(q.shape[1]):
        label = f"q_{j}" if state_names is None or j >= len(state_names) else str(state_names[j])
        ax.hist(q[:, j], bins=bins, range=(0.0, 1.0), weights=weights, histtype="step", linewidth=1.5, label=label)
    ax.set_xlabel("q_j(z)")
    ax.set_ylabel("weighted density")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_reaction_tube_distributions(
    q: np.ndarray,
    weights: np.ndarray,
    out_path: str,
    state_names=None,
    bins: int = 60,
    plot_threshold: float | None = None,
) -> None:
    pairs = unordered_pairs(q.shape[1])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for i, j in pairs:
        tube = q[:, i] * q[:, j]
        mask = np.ones_like(tube, dtype=bool) if plot_threshold is None else tube > float(plot_threshold)
        if not np.any(mask):
            continue
        if state_names is not None and i < len(state_names) and j < len(state_names):
            label = f"{state_names[i]}-{state_names[j]}"
        else:
            label = f"{i}-{j}"
        ax.hist(
            tube[mask],
            bins=bins,
            range=(float(plot_threshold or 0.0), 0.25),
            weights=weights[mask],
            histtype="step",
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("q_i(z) q_j(z)")
    ax.set_ylabel("weighted density")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_flux_profiles(pairs, thresholds, J, out_dir: str, fmt: str = "png") -> None:
    subdir = ensure_dir(os.path.join(out_dir, "flux_profiles"))
    for p_idx, (i, j) in enumerate(pairs):
        fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=160)
        ax.plot(thresholds, J[p_idx], marker="o")
        ax.axhline(float(np.mean(J[p_idx])), color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("isocommittor threshold c")
        ax.set_ylabel(f"J_{i}_{j}(c)")
        ax.set_title(f"{i} -> {j}")
        fig.tight_layout()
        fig.savefig(os.path.join(subdir, f"J_{i}_{j}.{fmt}"))
        plt.close(fig)

    if len(pairs) <= 20:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
        for p_idx, (i, j) in enumerate(pairs):
            ax.plot(thresholds, J[p_idx], marker="o", linewidth=1.0, label=f"{i}->{j}")
        ax.set_xlabel("isocommittor threshold c")
        ax.set_ylabel("J_ij(c)")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"flux_profiles_all.{fmt}"))
        plt.close(fig)


def plot_cv_fields(config: dict[str, Any], pack, q: np.ndarray, weights: np.ndarray, out_dir: str) -> list[str]:
    planes = normalize_planes(config.get("planes", []))
    if not planes:
        return []
    if pack.cv is None:
        raise RuntimeError("CV projection requested, but dataset has no cv block.")

    cv = pack.cv.numpy()
    cv_headers = pack.meta.get("cv_headers", None)
    if cv_headers is None or len(cv_headers) != cv.shape[1]:
        cv_headers = [f"cv_{i}" for i in range(cv.shape[1])]
    cv_headers = list(cv_headers)

    bins = int(config.get("bins", 60))
    fmt = str(config.get("format", "png"))
    cmap = str(config.get("cmap", "RdBu_r"))
    saved = []
    for plane in planes:
        missing = [name for name in plane if name not in cv_headers]
        if missing:
            raise ValueError(f"CV plane {plane} contains columns not present in dataset headers: {missing}")
        if len(plane) == 3:
            cvx, cvy, cvz = plane
            ix, iy, iz = cv_headers.index(cvx), cv_headers.index(cvy), cv_headers.index(cvz)
            x, y, z = cv[:, ix], cv[:, iy], cv[:, iz]
            axis_limits = resolve_cv_axis_limits([x, y, z], config)
            subdir = ensure_dir(os.path.join(out_dir, f"{cvx}__{cvy}__{cvz}"))

            for j in range(q.shape[1]):
                path = os.path.join(subdir, f"q_{j}__{cvx}__{cvy}__{cvz}.{fmt}")
                scatter_3d_field(
                    x,
                    y,
                    z,
                    q[:, j],
                    weights=weights,
                    labels=(cvx, cvy, cvz),
                    title=f"q_{j}",
                    color_label=f"q_{j}",
                    out_path=path,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                    config=config,
                    axis_limits=axis_limits,
                )
                saved.append(path)

            dest = np.argmax(q, axis=1).astype(float)
            path = os.path.join(subdir, f"destination__{cvx}__{cvy}__{cvz}.{fmt}")
            scatter_3d_field(
                x,
                y,
                z,
                dest,
                weights=weights,
                labels=(cvx, cvy, cvz),
                title="argmax_j q_j",
                color_label="destination",
                out_path=path,
                cmap="tab20",
                vmin=0.0,
                vmax=float(max(1, q.shape[1] - 1)),
                config=config,
                axis_limits=axis_limits,
            )
            saved.append(path)
            continue

        cvx, cvy = plane
        ix, iy = cv_headers.index(cvx), cv_headers.index(cvy)
        x, y = cv[:, ix], cv[:, iy]
        (xmin, xmax), (ymin, ymax) = resolve_cv_axis_limits([x, y], config)
        xedges = np.linspace(xmin, xmax, bins + 1)
        yedges = np.linspace(ymin, ymax, bins + 1)
        subdir = ensure_dir(os.path.join(out_dir, f"{cvx}__{cvy}"))

        for j in range(q.shape[1]):
            field = weighted_mean_2d(x, y, q[:, j], weights, xedges, yedges)
            fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=160)
            pcm = ax.pcolormesh(xedges, yedges, field.T, cmap=cmap, shading="auto", vmin=0.0, vmax=1.0)
            ax.set_xlabel(str(cvx))
            ax.set_ylabel(str(cvy))
            ax.set_title(f"q_{j}")
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label(f"q_{j}")
            fig.tight_layout()
            path = os.path.join(subdir, f"q_{j}__{cvx}__{cvy}.{fmt}")
            fig.savefig(path)
            plt.close(fig)
            saved.append(path)

        dest = np.argmax(q, axis=1).astype(float)
        field = weighted_mean_2d(x, y, dest, weights, xedges, yedges)
        fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=160)
        pcm = ax.pcolormesh(xedges, yedges, field.T, cmap="tab20", shading="auto", vmin=0, vmax=max(1, q.shape[1] - 1))
        ax.set_xlabel(str(cvx))
        ax.set_ylabel(str(cvy))
        ax.set_title("argmax_j q_j")
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label("destination")
        fig.tight_layout()
        path = os.path.join(subdir, f"destination__{cvx}__{cvy}.{fmt}")
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_reaction_tube_fields(
    config: dict[str, Any],
    pack,
    q: np.ndarray,
    weights: np.ndarray,
    out_dir: str,
) -> list[str]:
    if not bool(config.get("plot_reaction_tubes", True)):
        return []

    planes = normalize_planes(config.get("planes", []))
    if not planes:
        return []
    if pack.cv is None:
        raise RuntimeError("Reaction-tube CV projection requested, but dataset has no cv block.")

    cv = pack.cv.numpy()
    cv_headers = cv_headers_for_pack(pack)
    basin_state = pack.state.numpy().astype(np.int64)
    n_states = int(q.shape[1])

    bins = int(config.get("bins", 60))
    fmt = str(config.get("format", "png"))
    cmap = str(config.get("reaction_tube_cmap", "magma"))
    vmax_cfg = config.get("reaction_tube_vmax", 0.25)
    vmax = None if vmax_cfg is None else float(vmax_cfg)
    plot_threshold = reaction_tube_plot_threshold(config)
    reaction_tube_max_points = int(config.get("reaction_tube_max_3d_points", config.get("max_3d_points", 50000)))
    reaction_tube_selection = str(config.get("reaction_tube_3d_point_selection", "top"))
    reaction_tube_3d_alpha = float(
        config.get("reaction_tube_3d_alpha", config.get("reaction_tube_alpha", config.get("3d_alpha", 0.75)))
    )
    plot_network = bool(config.get("plot_reaction_tube_network", True))
    pairs = config.get("reaction_tube_pairs", None)
    if pairs is None:
        pairs = unordered_pairs(q.shape[1])
    else:
        pairs = [(int(i), int(j)) for i, j in pairs]
    center_model = load_model_for_center_evaluation(config)
    center_mode = str(config.get("reaction_tube_center_evaluate", "auto")).lower()
    if center_model is None and center_mode in {"true", "yes", "on", "1"}:
        raise KeyError("reaction_tube_center_evaluate=true requires NEXT_HIT_PLOT.model.")

    saved = []
    for plane in planes:
        missing = [name for name in plane if name not in cv_headers]
        if missing:
            raise ValueError(f"CV plane {plane} contains columns not present in dataset headers: {missing}")
        if len(plane) == 3:
            cvx, cvy, cvz = plane
            ix, iy, iz = cv_headers.index(cvx), cv_headers.index(cvy), cv_headers.index(cvz)
            x, y, z = cv[:, ix], cv[:, iy], cv[:, iz]
            axis_limits = resolve_cv_axis_limits([x, y, z], config)
            subdir = ensure_dir(os.path.join(out_dir, f"{cvx}__{cvy}__{cvz}", "reaction_tubes"))

            center_eval = should_center_evaluate_reaction_tube(config, plane, cv_headers) and center_model is not None
            if center_eval:
                (xmin, xmax), (ymin, ymax), (zmin, zmax) = axis_limits
                xcenters = 0.5 * (np.linspace(xmin, xmax, bins + 1)[:-1] + np.linspace(xmin, xmax, bins + 1)[1:])
                ycenters = 0.5 * (np.linspace(ymin, ymax, bins + 1)[:-1] + np.linspace(ymin, ymax, bins + 1)[1:])
                zcenters = 0.5 * (np.linspace(zmin, zmax, bins + 1)[:-1] + np.linspace(zmin, zmax, bins + 1)[1:])
                Xc, Yc, Zc = np.meshgrid(xcenters, ycenters, zcenters, indexing="ij")
                center_points = cv_points_from_plane(
                    plane,
                    [Xc.ravel(), Yc.ravel(), Zc.ravel()],
                    cv_headers,
                    config,
                )
                model, device = center_model
                q_centers = predict_q_at_cv_points(config, center_points, cv_headers, model, device)
                plot_x, plot_y, plot_z = Xc.ravel(), Yc.ravel(), Zc.ravel()
            else:
                q_centers = q
                plot_x, plot_y, plot_z = x, y, z

            network_points = []
            for i, j in pairs:
                if i == j or i < 0 or j < 0 or i >= q.shape[1] or j >= q.shape[1]:
                    raise ValueError(f"Invalid reaction_tube pair {(i, j)} for n_states={q.shape[1]}.")
                tube = q_centers[:, i] * q_centers[:, j]
                ids = select_reaction_tube_points(
                    tube,
                    threshold=plot_threshold,
                    max_points=reaction_tube_max_points,
                    mode=reaction_tube_selection,
                )
                if ids.size == 0:
                    continue
                if plot_network:
                    network_points.append(((i, j), plot_x[ids], plot_y[ids], plot_z[ids]))
                path = os.path.join(subdir, f"tube_{i}_{j}__{cvx}__{cvy}__{cvz}.{fmt}")
                scatter_3d_field(
                    plot_x[ids],
                    plot_y[ids],
                    plot_z[ids],
                    tube[ids],
                    weights=np.ones(ids.size, dtype=np.float64),
                    labels=(cvx, cvy, cvz),
                    title=f"q_{i} q_{j}",
                    color_label=f"q_{i}(z) q_{j}(z)",
                    out_path=path,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=vmax,
                    config=config,
                    axis_limits=axis_limits,
                    max_points=0,
                    alpha=reaction_tube_3d_alpha,
                    basin_xyz=(x, y, z),
                    basin_state=basin_state,
                    n_states=n_states,
                )
                saved.append(path)
            if plot_network:
                network_path = os.path.join(subdir, f"reactive_network__{cvx}__{cvy}__{cvz}.{fmt}")
                if plot_reaction_tube_network_3d(
                    network_points,
                    labels=(cvx, cvy, cvz),
                    out_path=network_path,
                    config=config,
                    axis_limits=axis_limits,
                    basin_xyz=(x, y, z),
                    basin_state=basin_state,
                    n_states=n_states,
                ):
                    saved.append(network_path)
            continue

        cvx, cvy = plane
        ix, iy = cv_headers.index(cvx), cv_headers.index(cvy)
        x, y = cv[:, ix], cv[:, iy]
        (xmin, xmax), (ymin, ymax) = resolve_cv_axis_limits([x, y], config)
        xedges = np.linspace(xmin, xmax, bins + 1)
        yedges = np.linspace(ymin, ymax, bins + 1)
        subdir = ensure_dir(os.path.join(out_dir, f"{cvx}__{cvy}", "reaction_tubes"))
        center_eval = should_center_evaluate_reaction_tube(config, plane, cv_headers) and center_model is not None
        if center_eval:
            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")
            center_points = cv_points_from_plane(plane, [Xc.ravel(), Yc.ravel()], cv_headers, config)
            model, device = center_model
            q_centers = predict_q_at_cv_points(config, center_points, cv_headers, model, device)

        network_fields = []
        network_pairs = []
        for i, j in pairs:
            if i == j or i < 0 or j < 0 or i >= q.shape[1] or j >= q.shape[1]:
                raise ValueError(f"Invalid reaction_tube pair {(i, j)} for n_states={q.shape[1]}.")
            if center_eval:
                field = (q_centers[:, i] * q_centers[:, j]).reshape((bins, bins))
                if plot_threshold is not None:
                    field = field.copy()
                    field[field <= plot_threshold] = np.nan
            else:
                tube = q[:, i] * q[:, j]
                mask = np.ones_like(tube, dtype=bool) if plot_threshold is None else tube > plot_threshold
                field = weighted_mean_2d(x[mask], y[mask], tube[mask], weights[mask], xedges, yedges)
            if plot_network:
                network_fields.append(field)
                network_pairs.append((i, j))
            fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=160)
            add_basin_overlay_2d(ax, x, y, basin_state, xedges, yedges, n_states, config)
            pcm = ax.pcolormesh(xedges, yedges, field.T, cmap=cmap, shading="auto", vmin=0.0, vmax=vmax)
            ax.set_xlabel(str(cvx))
            ax.set_ylabel(str(cvy))
            ax.set_title(f"q_{i} q_{j}")
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label(f"q_{i}(z) q_{j}(z)")
            fig.tight_layout()
            path = os.path.join(subdir, f"tube_{i}_{j}__{cvx}__{cvy}.{fmt}")
            fig.savefig(path)
            plt.close(fig)
            saved.append(path)
        if plot_network:
            network_path = os.path.join(subdir, f"reactive_network__{cvx}__{cvy}.{fmt}")
            if plot_reaction_tube_network_2d(
                network_fields,
                network_pairs,
                x=x,
                y=y,
                basin_state=basin_state,
                xedges=xedges,
                yedges=yedges,
                n_states=n_states,
                labels=(cvx, cvy),
                out_path=network_path,
                config=config,
            ):
                saved.append(network_path)

    return saved


def run(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(config.get("out_dir", "./next_hit_plots"))
    dataset_path = config.get("dataset", config.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("Plot config needs 'dataset' or 'dataset_path'.")
    dataset_stride = int(config.get("dataset_stride", 1))
    pack = apply_stride(load_dataset(dataset_path), dataset_stride)
    n_states = infer_n_states(pack, config.get("n_states", None))
    q = load_q(config, pack, n_states)

    weights = pack.weights.numpy().astype(np.float64)
    weights = weights / (np.sum(weights) + 1e-300)
    state_names = config.get("state_names", None)
    fmt = str(config.get("format", "png"))
    plot_q_distributions(
        q,
        weights,
        os.path.join(out_dir, f"q_distributions.{fmt}"),
        state_names=state_names,
        bins=int(config.get("distribution_bins", 60)),
    )
    reaction_tube_distribution_path = None
    if bool(config.get("plot_reaction_tubes", True)):
        reaction_tube_distribution_path = os.path.join(out_dir, f"reaction_tube_distributions.{fmt}")
        plot_reaction_tube_distributions(
            q,
            weights,
            reaction_tube_distribution_path,
            state_names=state_names,
            bins=int(config.get("reaction_tube_distribution_bins", config.get("distribution_bins", 60))),
            plot_threshold=reaction_tube_plot_threshold(config),
        )
    cv_paths = plot_cv_fields(config, pack, q, weights, out_dir)
    tube_paths = plot_reaction_tube_fields(config, pack, q, weights, out_dir)

    timing = resolve_lag_timing(config, dataset_stride)
    idx0_t, idx1_t = build_lagged_indices(
        q.shape[0],
        lag=int(timing["lag_index_step"]),
        traj_id=pack.traj_id,
        allow_cross=bool(config.get("allow_cross_traj_pairs", False)),
    )
    thresholds = make_thresholds(
        config.get("thresholds", None),
        n_thresholds=int(config.get("n_thresholds", 9)),
        start=float(config.get("threshold_start", 0.1)),
        stop=float(config.get("threshold_stop", 0.9)),
    ).numpy()
    pairs = resolve_ordered_pairs(
        n_states,
        adjacency=config.get("adjacency", config.get("flux_pairs", None)),
        symmetric_adjacency=bool(config.get("symmetric_adjacency", True)),
    )
    tau = float(timing["tau"])
    J, variance = estimate_flux_profiles(
        q=q,
        weights=pack.weights.numpy().astype(np.float64),
        idx0=idx0_t.numpy(),
        idx1=idx1_t.numpy(),
        pairs=pairs,
        thresholds=thresholds,
        eps=float(config.get("flux_eps", 0.02)),
        tau=tau,
        divide_by_tau=bool(config.get("divide_by_tau", False)),
        surface=str(config.get("flux_surface", "qi_decrease")),
        chunk_size=int(config.get("chunk_size", 20000)),
        weighted=bool(config.get("weighted_flux", True)),
    )
    plot_flux_profiles(pairs, thresholds, J, out_dir, fmt=fmt)

    rows = []
    for p_idx, (i, j) in enumerate(pairs):
        for t_idx, c in enumerate(thresholds):
            rows.append(
                {
                    "state_i": i,
                    "state_j": j,
                    "threshold": float(c),
                    "J_ij": float(J[p_idx, t_idx]),
                    "threshold_variance": float(variance[p_idx]),
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "flux_profiles.csv"), index=False)

    summary = {
        "out_dir": os.path.abspath(out_dir),
        "dataset": os.path.abspath(str(dataset_path)),
        "n_states": int(n_states),
        "q_distribution_plot": os.path.abspath(os.path.join(out_dir, f"q_distributions.{fmt}")),
        "reaction_tube_distribution_plot": (
            os.path.abspath(reaction_tube_distribution_path) if reaction_tube_distribution_path is not None else None
        ),
        "n_cv_projection_plots": int(len(cv_paths)),
        "n_reaction_tube_projection_plots": int(len(tube_paths)),
        "flux_profiles_csv": os.path.abspath(os.path.join(out_dir, "flux_profiles.csv")),
        "ordered_pairs": [[int(i), int(j)] for i, j in pairs],
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    print(f"[PLOT] Saved q distributions, reaction tubes, q_j projections, and J_ij(c) profiles to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot q_j distributions, CV projections, and flux profiles.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "NEXT_HIT_PLOT", "PLOT", "INFER")
    run(cfg)


if __name__ == "__main__":
    main()

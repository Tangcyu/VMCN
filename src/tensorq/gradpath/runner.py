from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, load_dataset, select_model_inputs
from ..next_hit.predict import infer_probabilities, load_committor_model
from .cluster import cluster_paths_with_linkage
from .coordinates import (
    has_periodic_cv_projection,
    model_inputs_to_projected_cv,
    projected_axis_names,
    projected_cv_to_model_inputs,
    selected_cv_points,
)
from .fel_selection import (
    channel_selection_from_fel_result,
    plot_fel_projection,
    save_fel_selection_npz,
    select_fel_kde_centers,
    weighted_average_paths_by_fel_cluster,
)
from .plot import plot_path_dendrogram, plot_paths_2d, plot_paths_3d, plot_selected_points, plot_selected_points_3d
from .selection import ChannelSelection, select_channel_points
from .shooting import GradientPath, build_channel_paths, finalize_stitched_path


def _torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "double": torch.float64,
        "float64": torch.float64,
    }
    value = str(name).lower()
    if value not in aliases:
        raise ValueError("dtype must be 'float32'/'float' or 'float64'/'double'.")
    return aliases[value]


def _maybe_load_q(config: dict[str, Any], model: torch.nn.Module, features: torch.Tensor, device: torch.device) -> np.ndarray:
    q_path = config.get("q_path", config.get("Q_npy", None))
    if q_path:
        q = np.load(str(q_path))
        if q.shape[0] != features.shape[0]:
            raise ValueError("Loaded q_path row count does not match the strided dataset.")
        return np.asarray(q, dtype=np.float64)
    return infer_probabilities(model, features.float(), device, batch_size=int(config.get("batch_size", 65536)))


def _plot_axes(config: dict[str, Any]) -> list[int | str]:
    axes = config.get("plot_axes", config.get("plot_cvs", [0, 1]))
    if not isinstance(axes, (list, tuple)) or len(axes) not in {2, 3}:
        raise ValueError("plot_axes/plot_cvs must contain exactly two or three axes.")
    return list(axes)


def _optional_endpoint(config: dict[str, Any], *keys: str) -> np.ndarray | None:
    for key in keys:
        value = config.get(key, None)
        if value is not None:
            endpoint = np.asarray(value, dtype=np.float64)
            if endpoint.ndim != 1:
                raise ValueError(f"{key} must be a one-dimensional coordinate list.")
            return endpoint
    return None


def _state_basin_from_config(config: dict[str, Any], state: int) -> tuple[np.ndarray | None, float | None]:
    spec = config.get("state_endpoints", None)
    if not isinstance(spec, dict):
        return None, None
    basins = spec.get("basins", [])
    if not isinstance(basins, list):
        return None, None
    for basin in basins:
        if not isinstance(basin, dict) or int(basin.get("label", -1)) != int(state):
            continue
        center = np.asarray(basin.get("center", []), dtype=np.float64)
        if center.ndim != 1 or center.size == 0:
            return None, None
        radius = basin.get("cutoff", basin.get("radius", None))
        if radius is None and "size" in basin:
            size = np.asarray(basin["size"], dtype=np.float64)
            radius = float(np.linalg.norm(size)) if size.ndim > 0 else float(size)
        return center, None if radius is None else float(radius)
    return None, None


def _state_basin_from_dataset_meta(pack, state: int) -> tuple[np.ndarray | None, float | None]:
    details = pack.meta.get("label_details", None) if isinstance(pack.meta, dict) else None
    if not isinstance(details, dict):
        return None, None
    basins = details.get("basins", [])
    if not isinstance(basins, list):
        return None, None
    for basin in basins:
        if not isinstance(basin, dict) or int(basin.get("label", -1)) != int(state):
            continue
        center = np.asarray(basin.get("center", []), dtype=np.float64)
        if center.ndim != 1 or center.size == 0:
            return None, None
        radius = basin.get("cutoff", basin.get("radius", None))
        if radius is None and "size" in basin:
            size = np.asarray(basin["size"], dtype=np.float64)
            radius = float(np.linalg.norm(size)) if size.ndim > 0 else float(size)
        return center, None if radius is None else float(radius)
    return None, None


def _expansion_radius_override(config: dict[str, Any], state_key: str) -> float | None:
    for key in (
        f"expansion_basin_radius_{state_key}",
        f"basin_radius_{state_key}",
        "expansion_basin_radius",
        "basin_radius",
    ):
        value = config.get(key, None)
        if value is not None:
            return float(value)
    return None


def _basin_radius_to_model_input(radius: float | None, input_meta: dict[str, Any]) -> float | None:
    if radius is None:
        return None
    if not has_periodic_cv_projection(input_meta):
        return float(radius)
    units = str(input_meta.get("model_periodic_cv_units", "degrees")).lower()
    if units in {"degrees", "degree", "deg"}:
        return float(radius) * (np.pi / 180.0)
    return float(radius)


def _periodic_projected_periods(input_meta: dict[str, Any]) -> list[float | None]:
    names = projected_axis_names(input_meta)
    periodic_names = set(input_meta.get("model_periodic_cvs", []))
    units = str(input_meta.get("model_periodic_cv_units", "degrees")).lower()
    period_value = 2.0 * np.pi if units in {"radians", "radian", "rad"} else 360.0
    return [period_value if name in periodic_names else None for name in names]


def _align_periodic_point(point: np.ndarray, reference: np.ndarray, periods: list[float | None]) -> np.ndarray:
    out = np.asarray(point, dtype=np.float64).copy()
    ref = np.asarray(reference, dtype=np.float64)
    for dim, period in enumerate(periods):
        if period is None or float(period) <= 0.0:
            continue
        p = float(period)
        out[dim] = ref[dim] + ((out[dim] - ref[dim] + 0.5 * p) % p) - 0.5 * p
    return out


def _wrap_periodic_projected_path(path: np.ndarray, periods: list[float | None]) -> np.ndarray:
    out = np.asarray(path, dtype=np.float64).copy()
    for dim, period in enumerate(periods):
        if period is None or float(period) <= 0.0:
            continue
        p = float(period)
        lower = -0.5 * p
        out[:, dim] = lower + np.remainder(out[:, dim] - lower, p)
    return out


def _project_paths_if_needed(paths: list[GradientPath], input_meta: dict[str, Any]) -> list[GradientPath]:
    if not has_periodic_cv_projection(input_meta):
        return paths
    out: list[GradientPath] = []
    for path in paths:
        projected = model_inputs_to_projected_cv(path.path, input_meta, unwrap=False)
        out.append(
            GradientPath(
                path=projected,
                q_path=path.q_path,
                start_index=path.start_index,
                weight=path.weight,
                channel_score=path.channel_score,
                state_i=path.state_i,
                state_j=path.state_j,
                model_path=path.path,
            )
        )
    return out


def _project_and_finalize_periodic_paths(
    paths: list[GradientPath],
    input_meta: dict[str, Any],
    *,
    num_images: int,
    endpoint_i: np.ndarray | None = None,
    endpoint_j: np.ndarray | None = None,
    smooth_iterations: int = 0,
    smooth_window: int = 3,
    reparameterize_after_smoothing: bool = True,
    wrap_output: bool = True,
) -> list[GradientPath]:
    periods = _periodic_projected_periods(input_meta)
    out: list[GradientPath] = []
    for path in paths:
        projected = model_inputs_to_projected_cv(path.path, input_meta, unwrap=True)
        endpoint_i_local = (
            _align_periodic_point(endpoint_i, projected[0], periods) if endpoint_i is not None else None
        )
        endpoint_j_local = (
            _align_periodic_point(endpoint_j, projected[-1], periods) if endpoint_j is not None else None
        )
        finalized, q_path = finalize_stitched_path(
            projected,
            path.q_path,
            num_images=int(num_images),
            endpoint_i=endpoint_i_local,
            endpoint_j=endpoint_j_local,
            smooth_iterations=int(smooth_iterations),
            smooth_window=int(smooth_window),
            reparameterize_after_smoothing=bool(reparameterize_after_smoothing),
            state_i=path.state_i,
            state_j=path.state_j,
        )
        if bool(wrap_output):
            finalized = _wrap_periodic_projected_path(finalized, periods)
        out.append(
            GradientPath(
                path=finalized,
                q_path=q_path,
                start_index=path.start_index,
                weight=path.weight,
                channel_score=path.channel_score,
                state_i=path.state_i,
                state_j=path.state_j,
                model_path=path.path,
            )
        )
    return out


def _cv_ranges_from_config(config: dict[str, Any], dim: int) -> list[list[float]] | None:
    ranges = config.get("fel_cv_ranges", config.get("cv_ranges", None))
    if ranges is not None:
        return [[float(item[0]), float(item[1])] for item in ranges]
    axis_keys = ["xlim", "ylim", "zlim"]
    axis_ranges = []
    for key in axis_keys[:dim]:
        value = config.get(key, None)
        if value is None:
            return None
        axis_ranges.append([float(value[0]), float(value[1])])
    return axis_ranges


def _kmeans_n_init(value: Any) -> str | int:
    if isinstance(value, str):
        return value
    return int(value)


def run_gradpath(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run direct-channel selection, gradient shooting, weighted clustering, and plots.

    Required config keys:
      dataset, model, state_i, state_j
    Common optional keys:
      threshold, max_points, step_size, max_steps, target_q, num_images,
      cluster_distance_threshold, model_input_space, cvs_to_use.
    """

    state_i = int(config["state_i"])
    state_j = int(config["state_j"])
    out_root = ensure_dir(config.get("out_dir", "./gradpath"))
    channel_name = str(config.get("channel_name", f"state_{state_i}_{state_j}"))
    out_dir = out_root
    if bool(config.get("use_channel_subdir", True)):
        out_dir = ensure_dir(os.path.join(out_root, channel_name))
    paths_dir = ensure_dir(os.path.join(out_dir, "paths"))
    centers_dir = ensure_dir(os.path.join(out_dir, "cluster_centers"))
    device = setup_device(config.get("device", "cuda:0"))
    dtype = _torch_dtype(config.get("dtype", "float32"))
    pack = apply_stride(load_dataset(config["dataset"]), int(config.get("dataset_stride", 1)))
    features, input_meta = select_model_inputs(pack, config)
    reproject_periodic = has_periodic_cv_projection(input_meta) and bool(config.get("reproject_periodic_paths", True))
    model = load_committor_model(config["model"], device)
    q = _maybe_load_q(config, model, features, device)

    weights = pack.weights.detach().cpu().double().numpy()
    selection_mode = str(config.get("selection_mode", "channel")).lower()
    fel_result = None
    if selection_mode in {"fel_kde", "kde_fel", "fel", "kde"}:
        if str(input_meta.get("model_input_space", "")).lower() != "cv":
            raise ValueError("selection_mode='fel_kde' requires model_input_space='cv'.")
        cv_coords = selected_cv_points(pack, input_meta)
        cv_ranges = _cv_ranges_from_config(config, cv_coords.shape[1])
        periodic_names = set(input_meta.get("model_periodic_cvs", []))
        cv_names = projected_axis_names(input_meta)
        periodic_flags = [name in periodic_names for name in cv_names]

        def evaluate_grid_q(points: np.ndarray) -> np.ndarray:
            model_points = projected_cv_to_model_inputs(points, input_meta) if reproject_periodic else points
            tensor = torch.as_tensor(model_points, dtype=torch.float32)
            return infer_probabilities(
                model,
                tensor,
                device,
                batch_size=int(config.get("batch_size", 65536)),
            )

        fel_result = select_fel_kde_centers(
            cv_coords,
            q,
            (state_i, state_j),
            weights=weights,
            bins=config.get("fel_bins", config.get("bins", 40)),
            bandwidth_bins=config.get("fel_bandwidth_bins", 1.0),
            ranges=cv_ranges,
            periodic=periodic_flags,
            periodic_units=str(input_meta.get("model_periodic_cv_units", config.get("periodic_cv_units", "degrees"))),
            threshold=float(config.get("threshold", config.get("fel_threshold", 0.20))),
            min_density=float(config.get("fel_min_density", 0.0)),
            kmin=int(config.get("fel_kmin", config.get("kmin", 1))),
            kmax=int(config.get("fel_kmax", config.get("kmax", 12))),
            n_clusters=config.get("fel_n_clusters", config.get("n_clusters", None)),
            points_per_cluster=int(config.get("fel_points_per_cluster", config.get("points_per_cluster", 5))),
            selection_power=float(config.get("fel_selection_power", 0.0)),
            selection_method=str(config.get("fel_selection_method", config.get("selection_method", "weighted"))),
            q_evaluator=evaluate_grid_q,
            random_state=int(config.get("seed", config.get("kmeans_random_state", 0))),
            kmeans_n_init=_kmeans_n_init(config.get("kmeans_n_init", "auto")),
            eps=float(config.get("eps", 1.0e-12)),
        )
        selected_model_points = (
            projected_cv_to_model_inputs(fel_result.selected_cv_points, input_meta)
            if reproject_periodic
            else fel_result.selected_cv_points
        )
        selection = channel_selection_from_fel_result(
            fel_result,
            selected_model_points,
            (state_i, state_j),
            threshold=float(config.get("threshold", config.get("fel_threshold", 0.20))),
        )
    else:
        selection = select_channel_points(
            features.detach().cpu().double().numpy(),
            q,
            state_i,
            state_j,
            threshold=float(config.get("threshold", 0.20)),
            weights=weights,
            max_points=config.get("max_points", config.get("n_trajs", None)),
            seed=config.get("seed", None),
            sample_with_replacement=bool(config.get("sample_with_replacement", False)),
            selection_power=float(config.get("selection_power", 1.0)),
        )
    np.save(os.path.join(out_dir, "selected_indices.npy"), selection.indices)
    np.save(os.path.join(out_dir, "selected_channel_score.npy"), selection.channel_score)
    np.save(os.path.join(out_dir, "selected_weights.npy"), selection.weights)
    if fel_result is not None:
        save_fel_selection_npz(os.path.join(out_dir, "fel_kde_selection.npz"), fel_result)
        np.savetxt(os.path.join(out_dir, "fel_selected_cv_points.txt"), fel_result.selected_cv_points)
        np.savetxt(os.path.join(out_dir, "fel_selected_cluster_labels.txt"), fel_result.selected_cluster_labels, fmt="%d")

    periodic = config.get("periodic_features", None)
    periodic_arr = None if periodic is None or periodic is False else np.asarray(periodic, dtype=bool)
    endpoint_i_input = _optional_endpoint(config, "endpoint_i", "state_i_endpoint", "manual_endpoint_i")
    endpoint_j_input = _optional_endpoint(config, "endpoint_j", "state_j_endpoint", "manual_endpoint_j")
    expansion = bool(config.get("expansion", False))
    basin_i_config, radius_i_config = _state_basin_from_config(config, state_i)
    basin_j_config, radius_j_config = _state_basin_from_config(config, state_j)
    basin_i_meta, radius_i_meta = _state_basin_from_dataset_meta(pack, state_i)
    basin_j_meta, radius_j_meta = _state_basin_from_dataset_meta(pack, state_j)
    if expansion and endpoint_i_input is None:
        endpoint_i_input = basin_i_config if basin_i_config is not None else basin_i_meta
    if expansion and endpoint_j_input is None:
        endpoint_j_input = basin_j_config if basin_j_config is not None else basin_j_meta
    basin_radius_i = _expansion_radius_override(config, "i")
    basin_radius_j = _expansion_radius_override(config, "j")
    if basin_radius_i is None:
        basin_radius_i = radius_i_config if radius_i_config is not None else radius_i_meta
    if basin_radius_j is None:
        basin_radius_j = radius_j_config if radius_j_config is not None else radius_j_meta
    endpoint_i = (
        projected_cv_to_model_inputs(endpoint_i_input, input_meta)
        if endpoint_i_input is not None and reproject_periodic
        else endpoint_i_input
    )
    endpoint_j = (
        projected_cv_to_model_inputs(endpoint_j_input, input_meta)
        if endpoint_j_input is not None and reproject_periodic
        else endpoint_j_input
    )
    basin_radius_i_model = _basin_radius_to_model_input(basin_radius_i, input_meta)
    basin_radius_j_model = _basin_radius_to_model_input(basin_radius_j, input_meta)
    num_images = int(config.get("num_images", config.get("tmp_images", 50)))
    smooth_iterations = int(config.get("smooth_iterations", config.get("path_smooth_iterations", 0)))
    smooth_window = int(config.get("smooth_window", config.get("path_smooth_window", 3)))
    reparameterize_after_smoothing = bool(config.get("reparameterize_after_smoothing", True))
    model_paths = build_channel_paths(
        model,
        selection,
        step_size=float(config.get("step_size", config.get("sd_step", 0.05))),
        max_steps=int(config.get("max_steps", config.get("max_step", 300))),
        target_q=float(config.get("target_q", 0.98)),
        num_images=None if reproject_periodic else num_images,
        device=device,
        dtype=dtype,
        normalize_gradient=bool(config.get("normalize_gradient", True)),
        expansion=expansion,
        expansion_eps=float(config.get("expansion_eps", 1.0e-6)),
        basin_radius_i=basin_radius_i_model,
        basin_radius_j=basin_radius_j_model,
        noise_scale=float(config.get("noise_scale", 0.0)),
        seed=config.get("seed", None),
        periodic=periodic_arr,
        integration_batch_size=config.get("integration_batch_size", config.get("shooting_batch_size", None)),
        endpoint_i=endpoint_i,
        endpoint_j=endpoint_j,
        smooth_iterations=0 if reproject_periodic else smooth_iterations,
        smooth_window=smooth_window,
        reparameterize_after_smoothing=reparameterize_after_smoothing,
        attach_endpoints=not reproject_periodic,
    )
    paths = (
        _project_and_finalize_periodic_paths(
            model_paths,
            input_meta,
            num_images=num_images,
            endpoint_i=endpoint_i_input,
            endpoint_j=endpoint_j_input,
            smooth_iterations=smooth_iterations,
            smooth_window=smooth_window,
            reparameterize_after_smoothing=reparameterize_after_smoothing,
            wrap_output=bool(config.get("wrap_periodic_projected_paths", True)),
        )
        if reproject_periodic
        else model_paths
    )
    for idx, path in enumerate(paths):
        np.savetxt(os.path.join(paths_dir, f"path_{idx:04d}.txt"), path.path)
        if path.model_path is not None:
            np.savetxt(os.path.join(paths_dir, f"path_{idx:04d}_model_input.txt"), path.model_path)
        np.savetxt(os.path.join(paths_dir, f"path_{idx:04d}_q.txt"), path.q_path)

    cluster_periods = None
    if reproject_periodic:
        units = str(input_meta.get("model_periodic_cv_units", config.get("periodic_cv_units", "degrees"))).lower()
        period_value = 2.0 * np.pi if units in {"radians", "radian", "rad"} else 360.0
        periodic_names = set(input_meta.get("model_periodic_cvs", []))
        names = projected_axis_names(input_meta)
        cluster_periods = [period_value if name in periodic_names else None for name in names]

    path_weights = np.asarray([path.weight for path in paths], dtype=np.float64)
    cluster_threshold = float(config.get("cluster_distance_threshold", config.get("cluster_threshold", 0.25)))
    labels, clusters, distance_matrix, linkage = cluster_paths_with_linkage(
        paths,
        weights=path_weights,
        distance_threshold=cluster_threshold,
        num_images=int(config.get("num_images", config.get("tmp_images", 50))),
        periods=cluster_periods,
    )

    # Discard sparsely populated pathway clusters. In automatic-pair runs the
    # P_jump cutoff also acts as the minimum fraction of max_points that a
    # pathway cluster must contain.
    max_points_for_filter = config.get("max_points", None)
    if max_points_for_filter is None or int(max_points_for_filter) <= 0:
        max_points_for_filter = len(paths)
    cluster_min_fraction = float(config.get("prob_threshold", 0.0))
    if cluster_min_fraction < 0.0:
        raise ValueError("prob_threshold must be nonnegative.")
    cluster_min_members = int(math.ceil(cluster_min_fraction * int(max_points_for_filter)))
    clusters_before_filter = list(clusters)
    discarded_clusters = [
        {
            "label": int(cluster.label),
            "n_members": int(cluster.member_indices.size),
            "total_weight": float(cluster.total_weight),
        }
        for cluster in clusters_before_filter
        if int(cluster.member_indices.size) < cluster_min_members
    ]
    retained_clusters = [
        cluster
        for cluster in clusters_before_filter
        if int(cluster.member_indices.size) >= cluster_min_members
    ]
    labels = np.zeros(len(paths), dtype=np.int64)
    clusters = []
    for new_label, cluster in enumerate(retained_clusters, start=1):
        retained = replace(cluster, label=new_label)
        clusters.append(retained)
        labels[retained.member_indices] = new_label

    if discarded_clusters:
        discarded_members = sum(item["n_members"] for item in discarded_clusters)
        print(
            f"[GRADPATH] discarded {len(discarded_clusters)} cluster(s) "
            f"({discarded_members} paths): n < {cluster_min_members} "
            "= ceil(prob_threshold * max_points)"
        )

    np.save(os.path.join(out_dir, "path_cluster_labels.npy"), labels)
    np.save(os.path.join(out_dir, "path_distance_matrix.npy"), distance_matrix)
    np.save(os.path.join(out_dir, "path_linkage.npy"), linkage)
    for pattern in ("cluster_*_center_path.txt", "cluster_*_medoid_path.txt"):
        for old_path in glob.glob(os.path.join(centers_dir, pattern)):
            os.remove(old_path)
    for cluster in clusters:
        np.savetxt(os.path.join(centers_dir, f"cluster_{cluster.label:02d}_center_path.txt"), cluster.center_path)
        np.savetxt(os.path.join(centers_dir, f"cluster_{cluster.label:02d}_medoid_path.txt"), cluster.medoid_path)

    fel_cluster_center_paths: dict[int, str] = {}
    if fel_result is not None:
        fel_dir = ensure_dir(os.path.join(out_dir, "fel_cluster_centers"))
        averaged = weighted_average_paths_by_fel_cluster(
            [path.path for path in paths],
            fel_result.selected_cluster_labels,
            selection.weights,
            periods=cluster_periods,
        )
        for label, center_path in averaged.items():
            path = os.path.join(fel_dir, f"fel_cluster_{label:02d}_weighted_center_path.txt")
            np.savetxt(path, center_path)
            fel_cluster_center_paths[int(label)] = os.path.abspath(path)

    plot_files: dict[str, str | None] = {"selected_points": None, "paths": None, "dendrogram": None}
    if bool(config.get("save_plots", True)):
        plot_config = dict(config)
        if reproject_periodic:
            plot_config.setdefault("periodic_plot_axes", list(input_meta.get("model_periodic_cvs", [])))
            plot_config.setdefault("periodic_cv_units", input_meta.get("model_periodic_cv_units", "degrees"))
        axis_names = projected_axis_names(input_meta) if reproject_periodic else input_meta.get("model_feature_names", None)
        plot_coords = selected_cv_points(pack, input_meta) if reproject_periodic else features.detach().cpu().double().numpy()
        plot_selection = selection
        if fel_result is not None:
            selected_plot_points = fel_result.selected_cv_points if reproject_periodic else selection.points
            offset = plot_coords.shape[0]
            plot_coords = np.vstack([plot_coords, selected_plot_points])
            plot_selection = ChannelSelection(
                indices=np.arange(offset, offset + selected_plot_points.shape[0], dtype=np.int64),
                points=selected_plot_points.astype(np.float64, copy=True),
                q=selection.q,
                weights=selection.weights,
                channel_score=selection.channel_score,
                state_i=selection.state_i,
                state_j=selection.state_j,
                threshold=selection.threshold,
            )
        axes = _plot_axes(config)
        if fel_result is not None:
            plot_fel_projection(
                fel_result,
                axes=axes,
                axis_names=axis_names,
                save_path=os.path.join(out_dir, "fel_projection_on_plot_axes.png"),
                config=plot_config,
            )
        dendrogram_path = os.path.join(out_dir, "path_dendrogram.png")
        plot_path_dendrogram(linkage, distance_threshold=cluster_threshold, save_path=dendrogram_path)
        if len(axes) == 3:
            plot_selected_points_3d(
                plot_coords,
                plot_selection,
                axes=axes,
                axis_names=axis_names,
                background_weights=np.concatenate([weights, selection.weights]) if fel_result is not None else weights,
                save_path=os.path.join(out_dir, "selected_points.png"),
                config=plot_config,
            )
            plot_paths_3d(
                paths,
                axes=axes,
                axis_names=axis_names,
                clusters=clusters,
                background=plot_coords,
                selected=plot_selection,
                save_path=os.path.join(out_dir, "paths_and_clusters.png"),
                config=plot_config,
            )
        else:
            plot_selected_points(
                plot_coords,
                plot_selection,
                axes=axes,
                axis_names=axis_names,
                background_weights=np.concatenate([weights, selection.weights]) if fel_result is not None else weights,
                save_path=os.path.join(out_dir, "selected_points.png"),
            )
            plot_paths_2d(
                paths,
                axes=axes,
                axis_names=axis_names,
                clusters=clusters,
                background=plot_coords,
                selected=plot_selection,
                save_path=os.path.join(out_dir, "paths_and_clusters.png"),
                config=plot_config,
            )
        plot_files["dendrogram"] = os.path.abspath(dendrogram_path)
        plot_files["selected_points"] = os.path.abspath(os.path.join(out_dir, "selected_points.png"))
        plot_files["paths"] = os.path.abspath(os.path.join(out_dir, "paths_and_clusters.png"))
        if fel_result is not None:
            plot_files["fel_projection"] = os.path.abspath(os.path.join(out_dir, "fel_projection_on_plot_axes.png"))

    summary = {
        "dataset": os.path.abspath(str(config["dataset"])),
        "model": os.path.abspath(str(config["model"])),
        "out_root": os.path.abspath(out_root),
        "out_dir": os.path.abspath(out_dir),
        "channel_name": channel_name,
        "model_input": input_meta,
        "state_i": state_i,
        "state_j": state_j,
        "selection_mode": selection_mode,
        "threshold": float(selection.threshold),
        "coordinate_names": projected_axis_names(input_meta) if reproject_periodic else input_meta.get("model_feature_names", None),
        "reprojected_periodic_paths": bool(reproject_periodic),
        "periodic_path_postprocess_space": "projected_cv" if reproject_periodic else "model_input",
        "wrap_periodic_projected_paths": bool(config.get("wrap_periodic_projected_paths", True)) if reproject_periodic else None,
        "endpoint_i": endpoint_i_input.tolist() if endpoint_i_input is not None else None,
        "endpoint_j": endpoint_j_input.tolist() if endpoint_j_input is not None else None,
        "endpoint_i_model_input": endpoint_i.tolist() if endpoint_i is not None else None,
        "endpoint_j_model_input": endpoint_j.tolist() if endpoint_j is not None else None,
        "expansion": bool(expansion),
        "expansion_eps": float(config.get("expansion_eps", 1.0e-6)),
        "expansion_basin_radius_i": None if basin_radius_i is None else float(basin_radius_i),
        "expansion_basin_radius_j": None if basin_radius_j is None else float(basin_radius_j),
        "expansion_basin_radius_i_model_input": None if basin_radius_i_model is None else float(basin_radius_i_model),
        "expansion_basin_radius_j_model_input": None if basin_radius_j_model is None else float(basin_radius_j_model),
        "num_images": int(num_images),
        "smooth_iterations": int(smooth_iterations),
        "smooth_window": int(smooth_window),
        "n_selected": int(selection.indices.size),
        "n_paths": int(len(paths)),
        "n_clusters": int(len(clusters)),
        "n_clusters_before_population_filter": int(len(clusters_before_filter)),
        "cluster_min_member_fraction": float(cluster_min_fraction),
        "cluster_filter_max_points": int(max_points_for_filter),
        "cluster_min_members": int(cluster_min_members),
        "discarded_clusters": discarded_clusters,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "integration_batch_size": config.get("integration_batch_size", config.get("shooting_batch_size", None)),
        "cluster_total_weights": [float(cluster.total_weight) for cluster in clusters],
        "selected_indices_npy": os.path.abspath(os.path.join(out_dir, "selected_indices.npy")),
        "cluster_labels_npy": os.path.abspath(os.path.join(out_dir, "path_cluster_labels.npy")),
        "linkage_npy": os.path.abspath(os.path.join(out_dir, "path_linkage.npy")),
        "fel_kde_selection_npz": (
            os.path.abspath(os.path.join(out_dir, "fel_kde_selection.npz")) if fel_result is not None else None
        ),
        "fel_n_clusters": int(fel_result.n_clusters) if fel_result is not None else None,
        "fel_cluster_center_paths": fel_cluster_center_paths if fel_cluster_center_paths else None,
        "plot_files": plot_files,
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    print(
        f"[GRADPATH] selected {selection.indices.size} points, built {len(paths)} paths, "
        f"found {len(clusters)} weighted clusters"
    )
    print(f"[GRADPATH] outputs saved to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weighted gradient pathways from next-hit q-vectors.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "GRADPATH", "GradPath")
    run_gradpath(cfg)


if __name__ == "__main__":
    main()

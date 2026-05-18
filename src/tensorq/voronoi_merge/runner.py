from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from ..common.config import ensure_dir, load_yaml, select_section, write_yaml
from .core import kl_divergence, voronoi_assignment
from .io import load_images, load_samples
from .iterative import run_iterative_pathway_expansion


def _get_path(config: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return str(value)
    return None


def _periods_from_config(config: dict[str, Any], ndim: int) -> list[float | None] | None:
    periods = config.get("periods", config.get("periodic_periods", None))
    if periods is not None and periods is not False:
        return [None if item is None else float(item) for item in periods]
    periodic = config.get("periodic", config.get("periodic_features", None))
    if periodic is None or periodic is False:
        return None
    if periodic is True:
        period = float(config.get("period", 360.0))
        return [period] * int(ndim)
    flags = list(periodic)
    if len(flags) != int(ndim):
        raise ValueError(f"periodic must have length {int(ndim)}.")
    period = float(config.get("period", 360.0))
    return [period if bool(flag) else None for flag in flags]


def _wrap_bounds_from_config(config: dict[str, Any], ndim: int) -> list[list[float] | None] | None:
    pathway_config = dict(config.get("pathway_iteration", {}) or {})
    bounds = pathway_config.get("wrap_bounds", config.get("wrap_bounds", None))
    if bounds is not None:
        if len(bounds) != int(ndim):
            raise ValueError(f"wrap_bounds must have length {int(ndim)}.")
        return [None if item is None else [float(item[0]), float(item[1])] for item in bounds]
    periods = _periods_from_config(config, ndim)
    if periods is None:
        return None
    return [None if period is None else [-0.5 * float(period), 0.5 * float(period)] for period in periods]


def _load_previous_probabilities(config: dict[str, Any]) -> np.ndarray | None:
    path = _get_path(config, "previous_probabilities", "previous_probability_file", "p_previous")
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    delimiter = "," if ext == ".csv" else None
    return np.asarray(np.loadtxt(path, delimiter=delimiter), dtype=np.float64).reshape(-1)


def run_voronoi_merge(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(config.get("out_dir", config.get("output_dir", "./voronoi_merge_out")))
    image_paths = config.get("images", config.get("image_files", config.get("paths", None)))
    if image_paths is None:
        raise ValueError("Voronoi segment alignment config requires 'images' (file, directory, or list).")
    pathway_config = dict(config.get("pathway_iteration", {}) or {})
    pathway_iteration_enabled = bool(pathway_config.get("enabled", config.get("pathway_iteration_enabled", False)))
    centers_initial = load_images(
        image_paths,
        kind=str(config.get("image_kind", "center")),
        key=config.get("image_key", None),
        flatten=False if pathway_iteration_enabled else bool(config.get("flatten_images", True)),
        image_stride=int(config.get("image_stride", config.get("path_image_stride", 1))),
        max_images_per_path=config.get("max_images_per_path", None),
        num_images_per_path=config.get("num_images_per_path", None),
    )
    sample_path = _get_path(config, "samples", "sample_file", "dataset")
    if sample_path is None:
        raise ValueError("Voronoi segment alignment config requires 'samples' or 'dataset'.")
    samples = load_samples(
        sample_path,
        config,
        key=config.get("sample_key", None),
        stride=int(config.get("sample_stride", config.get("dataset_stride", 1))),
        weights_path=config.get("weights", config.get("sample_weights", None)),
        weights_key=config.get("weights_key", None),
        traj_id_path=config.get("traj_id", config.get("sample_traj_id", None)),
        traj_id_key=config.get("traj_id_key", None),
    )
    if isinstance(centers_initial, np.ndarray):
        image_dim = int(centers_initial.shape[-1])
        centers_list = [np.asarray(p, dtype=np.float64) for p in (centers_initial if centers_initial.ndim == 3 else [centers_initial])] if not pathway_iteration_enabled else None
    else:
        image_dim = int(centers_initial[0].shape[-1])
        centers_list = centers_initial
    if samples.points.shape[1] != image_dim:
        raise ValueError(
            f"samples dimension {samples.points.shape[1]} does not match image dimension {image_dim}."
        )
    periods = _periods_from_config(config, samples.points.shape[1])
    wrap_bounds = _wrap_bounds_from_config(config, samples.points.shape[1])
    chunk_size = int(config.get("chunk_size", 65536))
    pseudocount = float(config.get("pseudocount", 0.0))
    device = config.get("device", None)
    dtype = str(config.get("dtype", "float32"))
    periodic_geometry = str(pathway_config.get("periodic_geometry", config.get("periodic_geometry", "minimum_image")))

    iterative_result = None
    if pathway_iteration_enabled:
        if isinstance(centers_initial, np.ndarray) and centers_initial.ndim != 3 and centers_initial.ndim != 2:
            raise ValueError("pathway_iteration requires non-flattened images with shape (n_paths, n_images, n_dim).")
        if isinstance(centers_initial, list) and len(centers_initial) == 0:
            raise ValueError("pathway_iteration requires at least one pathway.")
        iterative_result = run_iterative_pathway_expansion(
            centers_initial,
            samples.points,
            weights=samples.weights,
            traj_id=samples.traj_id,
            periods=periods,
            lag=int(pathway_config.get("lag", config.get("exchange_lag", 1))),
            terminal_image_margin=int(pathway_config.get("terminal_image_margin", 1)),
            min_exchange_count=float(pathway_config.get("min_exchange_count", 1.0)),
            min_exchange_probability=float(pathway_config.get("min_exchange_probability", 0.0)),
            exchange_weight_mode=str(pathway_config.get("exchange_weight_mode", "count")),
            max_cell_distance=pathway_config.get("max_cell_distance", None),
            max_iterations=int(pathway_config.get("max_iterations", 10)),
            convergence_tol=float(pathway_config.get("convergence_tol", 1.0e-6)),
            num_images=pathway_config.get("num_images", None),
            image_spacing=pathway_config.get("image_spacing", None),
            final_image_spacing=pathway_config.get("final_image_spacing", None),
            smooth_iterations=int(pathway_config.get("smooth_iterations", config.get("smooth_iterations", 2))),
            smooth_window=int(pathway_config.get("smooth_window", config.get("smooth_window", 5))),
            fixed_endpoints=bool(pathway_config.get("fixed_endpoints", True)),
            cell_relaxation=float(pathway_config.get("cell_relaxation", 1.0)),
            wrap_bounds=wrap_bounds,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
            periodic_geometry=periodic_geometry,
            out_dir=out_dir,
            plot_config={
                **dict(config.get("plotting", {}) or {}),
                "cvs_to_use": config.get("cvs_to_use", None),
                "xlim": config.get("xlim", (config.get("plotting", {}) or {}).get("xlim", None)),
                "ylim": config.get("ylim", (config.get("plotting", {}) or {}).get("ylim", None)),
            },
        )
        centers = np.vstack(iterative_result.paths)
    else:
        centers = np.vstack(centers_list) if isinstance(centers_initial, list) else centers_initial

    current = voronoi_assignment(
        samples.points,
        centers,
        weights=samples.weights,
        periods=periods,
        periodic_geometry=periodic_geometry,
        pseudocount=pseudocount,
        chunk_size=chunk_size,
        device=device,
        dtype=dtype,
    )

    previous_prob = _load_previous_probabilities(config)
    previous_assignment = None
    previous_path = _get_path(config, "previous_samples", "previous_sample_file", "previous_dataset")
    if previous_prob is None and previous_path is not None:
        previous = load_samples(
            previous_path,
            config,
            key=config.get("previous_sample_key", config.get("sample_key", None)),
            stride=int(config.get("previous_sample_stride", config.get("sample_stride", 1))),
            weights_path=config.get("previous_weights", None),
            weights_key=config.get("previous_weights_key", config.get("weights_key", None)),
            traj_id_path=config.get("previous_traj_id", None),
            traj_id_key=config.get("previous_traj_id_key", config.get("traj_id_key", None)),
        )
        previous_assignment = voronoi_assignment(
            previous.points,
            centers,
            weights=previous.weights,
            periods=periods,
            periodic_geometry=periodic_geometry,
            pseudocount=pseudocount,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
        )
        previous_prob = previous_assignment.probabilities

    kld = None
    if previous_prob is not None:
        kld = kl_divergence(current.probabilities, previous_prob, eps=float(config.get("eps", 1.0e-12)))

    np.savetxt(os.path.join(out_dir, "voronoi_images.txt"), centers)
    if iterative_result is not None:
        np.save(
            os.path.join(out_dir, "final_pathways.npy"),
            np.asarray(iterative_result.paths, dtype=object),
            allow_pickle=True,
        )
        for path_idx, path in enumerate(iterative_result.paths, start=1):
            np.savetxt(os.path.join(out_dir, f"final_pathway_{path_idx:02d}.txt"), path)
    np.savetxt(os.path.join(out_dir, "current_probabilities.txt"), current.probabilities)
    np.save(os.path.join(out_dir, "current_assignments.npy"), current.labels)
    np.save(os.path.join(out_dir, "current_distances.npy"), current.distances)
    if previous_prob is not None:
        np.savetxt(os.path.join(out_dir, "previous_probabilities.txt"), previous_prob)
    if previous_assignment is not None:
        np.save(os.path.join(out_dir, "previous_assignments.npy"), previous_assignment.labels)
        np.save(os.path.join(out_dir, "previous_distances.npy"), previous_assignment.distances)
    if isinstance(centers_initial, list):
        n_init_pathways = len(centers_initial)
        n_init_images = sum(p.shape[0] for p in centers_initial)
        init_images_per_path = [int(p.shape[0]) for p in centers_initial]
    elif isinstance(centers_initial, np.ndarray) and centers_initial.ndim == 3:
        n_init_pathways = int(centers_initial.shape[0])
        n_init_images = int(np.prod(centers_initial.shape[:2]))
        init_images_per_path = [int(centers_initial.shape[1])] * n_init_pathways
    elif isinstance(centers_initial, np.ndarray):
        n_init_pathways = None
        n_init_images = int(centers_initial.shape[0])
        init_images_per_path = None
    else:
        n_init_pathways = None
        n_init_images = None
        init_images_per_path = None
    summary = {
        "out_dir": os.path.abspath(out_dir),
        "images": image_paths,
        "samples": os.path.abspath(sample_path),
        "previous_samples": os.path.abspath(previous_path) if previous_path is not None else None,
        "n_initial_pathways": n_init_pathways,
        "n_initial_images": n_init_images,
        "initial_images_per_path": init_images_per_path,
        "n_voronoi_cells": int(centers.shape[0]),
        "n_samples": int(samples.points.shape[0]),
        "dimension": int(samples.points.shape[1]),
        "periods": periods,
        "wrap_bounds": wrap_bounds,
        "device": str(device) if device is not None else None,
        "dtype": dtype,
        "periodic_geometry": periodic_geometry,
        "image_stride": int(config.get("image_stride", config.get("path_image_stride", 1))),
        "max_images_per_path": config.get("max_images_per_path", None),
        "num_images_per_path": config.get("num_images_per_path", None),
        "num_images": pathway_config.get("num_images", None),
        "image_spacing": pathway_config.get("image_spacing", None),
        "pathway_iteration_enabled": pathway_iteration_enabled,
        "pathway_iteration_converged": None if iterative_result is None else bool(iterative_result.converged),
        "pathway_iteration_count": None if iterative_result is None else int(len(iterative_result.history)),
        "exchange_weight_mode": None if not pathway_iteration_enabled else str(pathway_config.get("exchange_weight_mode", "count")),
        "n_final_pathways": None if iterative_result is None else len(iterative_result.paths),
        "final_images_per_path": None if iterative_result is None else [int(p.shape[0]) for p in iterative_result.paths],
        "kld": kld,
        "current_probabilities": os.path.abspath(os.path.join(out_dir, "current_probabilities.txt")),
        "previous_probabilities": (
            os.path.abspath(os.path.join(out_dir, "previous_probabilities.txt")) if previous_prob is not None else None
        ),
        "voronoi_images": os.path.abspath(os.path.join(out_dir, "voronoi_images.txt")),
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    if kld is None:
        print(f"[VORONOI_SEGMENTS] built {centers.shape[0]} Voronoi cells; no previous probabilities were provided")
    else:
        print(f"[VORONOI_SEGMENTS] built {centers.shape[0]} Voronoi cells; KLD={kld:.8g}")
    print(f"[VORONOI_SEGMENTS] outputs saved to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Voronoi shared-segment alignment and iterative KLD diagnostics.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--images", help="Override Voronoi image file/directory")
    parser.add_argument("--samples", help="Override current-iteration samples")
    parser.add_argument("--previous-samples", help="Override previous-iteration samples")
    parser.add_argument("--out-dir", help="Override output directory")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "VORONOI_MERGE", "voronoi_merge", "GRADPATH_MERGY", "gradpath_mergy")
    if args.images is not None:
        cfg["images"] = args.images
    if args.samples is not None:
        cfg["samples"] = args.samples
    if args.previous_samples is not None:
        cfg["previous_samples"] = args.previous_samples
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    run_voronoi_merge(cfg)


if __name__ == "__main__":
    main()

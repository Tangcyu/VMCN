#!/usr/bin/env python
"""Reparameterize pathways and export with committor scores.

Usage:
  python scripts/export_paths.py --config export_paths.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorq.common.config import ensure_dir, load_yaml, select_section
from tensorq.common.data import apply_stride, load_dataset, select_model_inputs
from tensorq.gradpath.coordinates import (
    has_periodic_cv_projection,
    projected_cv_to_model_inputs,
)
from tensorq.gradpath.plot import apply_plot_style, cm2inch, plot_colored_paths_2d
from tensorq.gradpath.runner import _plot_axes, _torch_dtype
from tensorq.next_hit.predict import load_committor_model
from tensorq.voronoi_merge.iterative import (
    smooth_reparameterize_paths_independently,
    wrap_periodic_points,
)
from tensorq.voronoi_merge.core import normalize_periods


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cpu")
    requested = torch.device(str(device_str))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _load_paths_npy(path: str) -> list[np.ndarray]:
    paths = np.load(path, allow_pickle=True)
    if isinstance(paths, np.ndarray) and paths.dtype == object:
        return [np.asarray(p, dtype=np.float64) for p in paths]
    if isinstance(paths, np.ndarray) and paths.ndim == 3:
        return [np.asarray(p, dtype=np.float64) for p in paths]
    if isinstance(paths, np.ndarray) and paths.ndim == 2:
        return [np.asarray(paths, dtype=np.float64)]
    return [np.asarray(p, dtype=np.float64) for p in paths]


def _parse_periods(value, ndim: int) -> list[float | None]:
    if value is None or value is False:
        return [None] * ndim
    if isinstance(value, (int, float)):
        return [float(value)] * ndim
    if isinstance(value, (list, tuple)):
        return [None if v is None else float(v) for v in value]
    return [None] * ndim


def _parse_wrap_bounds(value, periods: list[float | None]) -> list[tuple[float, float] | None] | None:
    if value is None or value is False:
        return None
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if item is None:
                out.append(None)
            else:
                out.append((float(item[0]), float(item[1])))
        return out
    return None


def compute_path_q(
    model: torch.nn.Module,
    paths: list[np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    input_meta: dict[str, Any] | None = None,
) -> list[np.ndarray]:
    """Run committor model on every image of every path, return per-path q arrays."""
    model = model.to(device=device, dtype=dtype)
    model.eval()
    sizes = [p.shape[0] for p in paths]
    points = np.vstack(paths)
    if input_meta is not None and has_periodic_cv_projection(input_meta):
        points = projected_cv_to_model_inputs(points, input_meta)
    q_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, points.shape[0], int(batch_size)):
            end = min(points.shape[0], start + int(batch_size))
            x = torch.as_tensor(points[start:end], dtype=dtype, device=device)
            q_chunks.append(model(x).detach().cpu().double().numpy())
    q_all = np.vstack(q_chunks)
    out: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        out.append(q_all[offset: offset + size])
        offset += size
    return out


def run_export_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Main entry point: reparameterize, score, export, and optionally plot pathways."""

    paths_npy = config["paths_npy"]
    if not os.path.exists(paths_npy):
        raise FileNotFoundError(f"paths_npy not found: {paths_npy}")
    out_dir = ensure_dir(config.get("out_dir", os.path.dirname(paths_npy)))

    # --- load pathways ---
    path_list = _load_paths_npy(paths_npy)
    ndim = path_list[0].shape[1]

    # --- reparameterize ---
    spacing = config.get("image_spacing", None)
    num_images = config.get("num_images", None)
    smooth_iters = int(config.get("smooth_iterations", 2))
    smooth_window = int(config.get("smooth_window", 5))
    periodic_geometry = str(config.get("periodic_geometry", "minimum_image"))
    periods = _parse_periods(config.get("periods"), ndim)
    wrap_bounds = _parse_wrap_bounds(config.get("wrap_bounds"), periods)

    if spacing is not None or num_images is not None:
        path_list = smooth_reparameterize_paths_independently(
            path_list,
            periods=periods,
            num_images=int(num_images) if spacing is None and num_images is not None else None,
            image_spacing=float(spacing) if spacing is not None else None,
            smooth_iterations=smooth_iters,
            smooth_window=smooth_window,
            endpoints=None,
            wrap_bounds=wrap_bounds,
            periodic_geometry=periodic_geometry,
        )
    elif wrap_bounds is not None:
        path_list = [wrap_periodic_points(p, periods=periods, wrap_bounds=wrap_bounds) for p in path_list]

    # --- export per-path txt ---
    for idx, path in enumerate(path_list, start=1):
        np.savetxt(os.path.join(out_dir, f"path_{idx}.txt"), path, fmt="%.12g")
        print(f"Wrote path_{idx}.txt  ({path.shape[0]} images)")

    # --- committor scoring ---
    q_paths = None
    if "model" in config:
        device = _resolve_device(config.get("device", "cpu"))
        dtype = _torch_dtype(config.get("dtype", "float32"))
        batch_size = int(config.get("batch_size", 65536))
        model = load_committor_model(config["model"], device)
        input_meta = None
        if "dataset" in config:
            pack = apply_stride(load_dataset(config["dataset"]), int(config.get("dataset_stride", 1)))
            _, input_meta = select_model_inputs(pack, config)
        q_paths = compute_path_q(
            model, path_list, device=device, dtype=dtype,
            batch_size=batch_size, input_meta=input_meta,
        )
        state_i = int(config.get("state_i", 0))
        state_j = int(config.get("state_j", 1))
        for idx, q in enumerate(q_paths, start=1):
            q_ij = q[:, state_i] * q[:, state_j]
            out_txt = os.path.join(out_dir, f"path_{idx}_qij.txt")
            np.savetxt(out_txt, np.column_stack([q, q_ij]), fmt="%.12g",
                       header="q_0 q_1 ... q_ij")
            print(f"Wrote path_{idx}_qij.txt  (q shape {q.shape})")

    # --- plot ---
    if bool(config.get("plot", True)):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        apply_plot_style(config)

        if q_paths is not None:
            state_i = int(config.get("state_i", 0))
            state_j = int(config.get("state_j", 1))
            scores = [q[:, state_i] * q[:, state_j] for q in q_paths]
            plot_colored_paths_2d(
                path_list, scores,
                axes=config.get("plot_axes", [0, 1]),
                save_path=os.path.join(out_dir, "paths.png"),
                cmap=str(config.get("colored_path_cmap", "magma")),
                vmin=0.0,
                vmax=config.get("colored_path_vmax", config.get("separatrix_vmax", 0.25)),
                linewidth=float(config.get("path_linewidth", 2.35)),
                background_point_size=float(config.get("background_point_size", 2.0)),
                background_point_alpha=float(config.get("background_point_alpha", 0.12)),
                title=f"Pathways colored by $q_{{{state_i}}} q_{{{state_j}}}$",
                label_paths=True,
                config=config,
            )
        else:
            from tensorq.voronoi_merge.plot import _break_periodic_line as _bpl

            plot_periods = None if periods is None else [periods[0], periods[1]] if ndim >= 2 else None
            cmap = plt.get_cmap("tab10")
            fig, ax = plt.subplots(
                figsize=(cm2inch(config.get("figsize_cm", 3.5)),
                         cm2inch(config.get("figsize_cm", 3.5)) / config.get("fig_aspect", 1.2)),
            )
            for idx, path in enumerate(path_list):
                c = cmap(idx % 10)
                if ndim >= 2:
                    xy = _bpl(path[:, [0, 1]], plot_periods)
                    ax.plot(xy[:, 0], xy[:, 1], color=c, linewidth=2.0, label=f"path {idx+1}")
                    ax.scatter(path[:, 0], path[:, 1], s=12, color=c, edgecolors="black", linewidths=0.25, zorder=3)
            ax.set_xlabel("axis 0")
            ax.set_ylabel("axis 1")
            ax.set_title(f"{len(path_list)} pathways  ({path_list[0].shape[0]} images)")
            ax.legend(frameon=False, fontsize=config.get("legend_font_size", config.get("font_size", 7)))
            ax.tick_params(labelsize=config.get("font_size", 7), width=config.get("line_width", 0.4))
            fig.savefig(os.path.join(out_dir, "paths.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        print(f"Wrote {os.path.join(out_dir, 'paths.png')}")

    n_images = sum(p.shape[0] for p in path_list)
    print(f"\nExported {len(path_list)} pathways ({n_images} total images) to {out_dir}")
    return {
        "out_dir": os.path.abspath(out_dir),
        "n_pathways": len(path_list),
        "total_images": n_images,
        "images_per_path": [int(p.shape[0]) for p in path_list],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reparameterize pathways and export with committor scores.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "EXPORT_PATHS", "export_paths", "PATH_EXPORT")
    run_export_paths(cfg)


if __name__ == "__main__":
    main()

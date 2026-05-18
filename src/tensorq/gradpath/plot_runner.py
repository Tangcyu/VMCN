from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Any

import numpy as np
import torch

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, load_dataset, select_model_inputs
from ..next_hit.predict import load_committor_model
from .coordinates import (
    has_periodic_cv_projection,
    projected_axis_names,
    projected_cv_to_model_inputs,
    selected_cv_points,
)
from .plot import apply_plot_style, plot_colored_paths_2d, plot_colored_paths_3d
from .runner import _plot_axes, _torch_dtype


def _resolve_channel_dir(config: dict[str, Any]) -> str:
    state_i = int(config["state_i"])
    state_j = int(config["state_j"])
    out_root = ensure_dir(config.get("out_dir", "./gradpath"))
    channel_name = str(config.get("channel_name", f"state_{state_i}_{state_j}"))
    if bool(config.get("use_channel_subdir", True)):
        return os.path.join(out_root, channel_name)
    return out_root


_STATE_DIR_RE = re.compile(r"^state_(\d+)_(\d+)$")


def find_state_pairs(out_dir: str) -> list[tuple[int, int]]:
    """Scan *out_dir* for ``state_i_j`` subdirectories and return sorted (i, j) pairs."""
    if not os.path.isdir(out_dir):
        return []
    pairs: list[tuple[int, int]] = []
    for entry in sorted(os.listdir(out_dir)):
        m = _STATE_DIR_RE.match(entry)
        if m and os.path.isdir(os.path.join(out_dir, entry)):
            pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def _path_glob(channel_dir: str, kind: str) -> str:
    if kind == "center":
        return os.path.join(channel_dir, "cluster_centers", "cluster_*_center_path.txt")
    if kind == "medoid":
        return os.path.join(channel_dir, "cluster_centers", "cluster_*_medoid_path.txt")
    if kind == "all":
        return os.path.join(channel_dir, "paths", "path_[0-9][0-9][0-9][0-9].txt")
    raise ValueError("plot_path_kind must be one of: center, medoid, all.")


def _load_paths(channel_dir: str, kind: str) -> tuple[list[str], list[np.ndarray]]:
    files = sorted(glob.glob(_path_glob(channel_dir, kind)))
    if not files:
        raise RuntimeError(f"No {kind!r} path files found under {channel_dir}. Run gradpath first.")
    paths = [np.loadtxt(path, dtype=np.float64) for path in files]
    paths = [path.reshape(1, -1) if path.ndim == 1 else path for path in paths]
    return files, paths


def _load_background_points(
    config: dict[str, Any],
    axis_dim: int,
    *,
    input_meta: dict[str, Any] | None = None,
) -> np.ndarray | None:
    if not bool(config.get("plot_dataset_points", True)):
        return None
    if "dataset" not in config:
        return None
    pack = apply_stride(load_dataset(config["dataset"]), int(config.get("dataset_stride", 1)))
    if input_meta is not None and has_periodic_cv_projection(input_meta):
        points = selected_cv_points(pack, input_meta)
    else:
        features, _ = select_model_inputs(pack, config)
        points = features.detach().cpu().double().numpy()
    if points.shape[1] <= axis_dim:
        raise ValueError("Dataset point coordinates do not contain the requested plot axes.")
    max_points = int(config.get("plot_dataset_max_points", config.get("background_max_points", 0)) or 0)
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(config.get("seed", None))
        ids = rng.choice(points.shape[0], size=max_points, replace=False)
        ids.sort()
        points = points[ids]
    return points


def _infer_path_q(
    model: torch.nn.Module,
    paths: list[np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    input_meta: dict[str, Any] | None = None,
) -> list[np.ndarray]:
    model = model.to(device=device, dtype=dtype)
    model.eval()
    sizes = [path.shape[0] for path in paths]
    points = np.vstack(paths)
    if input_meta is not None and has_periodic_cv_projection(input_meta):
        points = projected_cv_to_model_inputs(points, input_meta)
    q_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, points.shape[0], int(batch_size)):
            end = min(points.shape[0], start + int(batch_size))
            x = torch.as_tensor(points[start:end], dtype=dtype, device=device)
            q = model(x)
            q_chunks.append(q.detach().cpu().double().numpy())
    q_all = np.vstack(q_chunks)
    out: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        out.append(q_all[offset : offset + size])
        offset += size
    return out


def run_gradpath_plot(config: dict[str, Any]) -> dict[str, Any]:
    """Plot saved gradpath pathways colored by q_i*q_j using the same YAML."""

    apply_plot_style(config)

    state_i = int(config["state_i"])
    state_j = int(config["state_j"])
    channel_dir = _resolve_channel_dir(config)
    kind = str(config.get("plot_path_kind", config.get("colored_path_kind", "center"))).lower()
    path_files, paths = _load_paths(channel_dir, kind)
    device = setup_device(config.get("device", "cuda:0"))
    dtype = _torch_dtype(config.get("dtype", "float32"))
    model = load_committor_model(config["model"], device)
    input_meta = None
    axis_names = None
    if "dataset" in config:
        pack = apply_stride(load_dataset(config["dataset"]), int(config.get("dataset_stride", 1)))
        _, input_meta = select_model_inputs(pack, config)
        axis_names = projected_axis_names(input_meta) if has_periodic_cv_projection(input_meta) else input_meta.get(
            "model_feature_names", None
        )
    q_paths = _infer_path_q(
        model,
        paths,
        device=device,
        dtype=dtype,
        batch_size=int(config.get("plot_batch_size", config.get("batch_size", 65536))),
        input_meta=input_meta,
    )
    scores = [q[:, state_i] * q[:, state_j] for q in q_paths]

    axes = _plot_axes(config)
    axis_ids = []
    for axis in axes:
        if isinstance(axis, str):
            if axis_names is None or axis not in axis_names:
                raise ValueError(f"Unknown CV/feature axis {axis!r}.")
            axis_ids.append(axis_names.index(axis))
        else:
            axis_ids.append(int(axis))
    background_points = _load_background_points(config, max(axis_ids), input_meta=input_meta)
    save_path = os.path.join(
        channel_dir,
        str(config.get("colored_path_plot", f"{kind}_paths_qiqj_colored.png")),
    )
    plot_kwargs = dict(
        axes=axes,
        axis_names=axis_names,
        background_points=background_points,
        save_path=save_path,
        cmap=str(config.get("colored_path_cmap", "magma")),
        vmin=config.get("colored_path_vmin", 0.0),
        vmax=config.get("colored_path_vmax", config.get("separatrix_vmax", 0.25)),
        linewidth=float(config.get("colored_path_linewidth", 2.35)),
        background_point_size=float(config.get("dataset_point_size", config.get("background_point_size", 2.0))),
        background_point_alpha=float(config.get("dataset_point_alpha", config.get("background_point_alpha", 0.12))),
        label_paths=bool(config.get("label_colored_paths", True)),
        title=f"State {state_i}-{state_j} pathways colored by $q_{{{state_i}}} q_{{{state_j}}}$",
    )
    if len(axes) == 3:
        plot_colored_paths_3d(paths, scores, config=config, **plot_kwargs)
    else:
        plot_colored_paths_2d(paths, scores, config=config, **plot_kwargs)
    summary = {
        "channel_dir": os.path.abspath(channel_dir),
        "state_i": state_i,
        "state_j": state_j,
        "plot_path_kind": kind,
        "n_paths": len(paths),
        "n_dataset_points_plotted": int(0 if background_points is None else background_points.shape[0]),
        "path_files": [os.path.abspath(path) for path in path_files],
        "colored_path_plot": os.path.abspath(save_path),
    }
    write_yaml(summary, os.path.join(channel_dir, f"{kind}_paths_qiqj_plot_summary.yaml"))
    print(f"[GRADPATH_PLOT] plotted {len(paths)} {kind} paths to {save_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved gradpath pathways colored by q_i*q_j.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Scan out_dir for all state_i_j subdirectories and plot each one.",
    )
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "GRADPATH", "GradPath")

    if args.auto:
        out_dir = ensure_dir(cfg.get("out_dir", "./gradpath"))
        pairs = find_state_pairs(out_dir)
        if not pairs:
            print(f"[GRADPATH_PLOT] No state_*_* subdirectories found under {out_dir}")
            return
        print(f"[GRADPATH_PLOT] Found {len(pairs)} state pair(s) under {out_dir}")
        for state_i, state_j in pairs:
            pair_cfg = dict(cfg)
            pair_cfg["state_i"] = state_i
            pair_cfg["state_j"] = state_j
            pair_cfg["out_dir"] = out_dir
            pair_cfg["channel_name"] = f"state_{state_i}_{state_j}"
            pair_cfg["use_channel_subdir"] = True
            try:
                run_gradpath_plot(pair_cfg)
            except Exception as exc:
                print(f"[GRADPATH_PLOT] ERROR plotting state_{state_i}_{state_j}: {exc}")
    else:
        run_gradpath_plot(cfg)


if __name__ == "__main__":
    main()

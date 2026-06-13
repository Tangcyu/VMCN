from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from ..common.config import ensure_dir
from ..common.data import (
    cv_headers_for_pack,
)


def _entropy_confidence(q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    q = np.clip(q_values, eps, 1.0)
    q_max = np.max(q, axis=1)
    q_argmax = np.argmax(q, axis=1).astype(np.int64)
    entropy = -np.sum(q * np.log(q), axis=1)
    entropy_norm = entropy / np.log(q.shape[1])
    return q_max, q_argmax, entropy, entropy_norm


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="") as fh:
            return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _thin_indices(indices: np.ndarray, max_points: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    max_points = int(max_points)
    if max_points <= 0 or indices.size <= max_points:
        return indices
    keep = np.linspace(0, indices.size - 1, max_points, dtype=np.int64)
    return indices[keep]


def _resolve_cv_plane(cv_data, cv_headers, config):
    plane = config.get("planes", config.get("cv_plane", []))
    if not plane:
        if len(cv_headers) >= 2:
            plane = [cv_headers[0], cv_headers[1]]
        else:
            raise ValueError("No CV plane specified and fewer than 2 CV columns available.")
    if isinstance(plane, (list, tuple)) and plane and isinstance(plane[0], (list, tuple)):
        plane = plane[0]
    if len(plane) not in {2, 3}:
        raise ValueError("CV plane must list exactly 2 or 3 CV column names.")
    missing = [name for name in plane if name not in cv_headers]
    if missing:
        raise ValueError(f"CV plane columns not in dataset: {missing}")
    indices = [cv_headers.index(name) for name in plane]
    return plane, indices, cv_data[:, indices]


def _apply_axis_limits(ax, config, is_3d):
    for setter, key in [(ax.set_xlim, "xlim"), (ax.set_ylim, "ylim")]:
        vals = config.get(key)
        if isinstance(vals, (list, tuple)) and len(vals) == 2:
            setter(float(vals[0]), float(vals[1]))
    if is_3d:
        vals = config.get("zlim")
        if isinstance(vals, (list, tuple)) and len(vals) == 2:
            ax.set_zlim(float(vals[0]), float(vals[1]))


def _state_names(config, n_states):
    names = config.get("state_names")
    if names is None:
        return [f"state {i}" for i in range(n_states)]
    names = [str(name) for name in names]
    if len(names) < n_states:
        names.extend(f"state {i}" for i in range(len(names), n_states))
    return names


def plot_relabel_cv(pack, old_state, new_state, proposal, config, output_dir):
    if pack.cv is None:
        print("[RELABEL] No CV data available; skipping relabel CV plots.")
        return []

    import matplotlib.pyplot as plt

    ensure_dir(output_dir)
    cv = pack.cv.detach().cpu().numpy()
    cv_headers = cv_headers_for_pack(pack)
    plane, indices, cv_plane = _resolve_cv_plane(cv, cv_headers, config)
    is_3d = len(indices) == 3
    n_states = int(max(np.max(new_state[new_state >= 0]) + 1 if np.any(new_state >= 0) else 0, 1))
    fmt = str(config.get("format", "png"))
    max_points = int(config.get("max_scatter_points", 20000))
    max_highlight = int(config.get("relabel", {}).get("max_plot_highlight_points", max_points))
    alpha = float(config.get("scatter_alpha", 0.35))
    size = float(config.get("scatter_size", 3.0))
    cmap = plt.get_cmap(str(config.get("basin_cmap", "tab10")), max(n_states, 1))
    names = _state_names(config, n_states)
    saved = []

    show_unlabeled = bool(config.get("relabel", {}).get("plot_unlabeled", False))
    plot_mask = np.ones(cv_plane.shape[0], dtype=bool) if show_unlabeled else (new_state >= 0)
    ids = _thin_indices(np.flatnonzero(plot_mask), max_points)
    if is_3d:
        fig = plt.figure(figsize=(6.5, 5.5), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        for state in range(n_states):
            ids_i = np.intersect1d(np.flatnonzero(new_state == state), ids)
            if ids_i.size:
                ax.scatter(
                    cv_plane[ids_i, 0], cv_plane[ids_i, 1], cv_plane[ids_i, 2],
                    s=size + 4, color=cmap(state), alpha=alpha, linewidths=0,
                    label=names[state],
                )
        ids_u = np.intersect1d(np.flatnonzero(new_state == -1), ids)
        if show_unlabeled and ids_u.size:
            ax.scatter(
                cv_plane[ids_u, 0], cv_plane[ids_u, 1], cv_plane[ids_u, 2],
                s=size, color="gray", alpha=alpha * 0.5, linewidths=0,
                label="unlabeled",
            )
        ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1]); ax.set_zlabel(plane[2])
        ax.view_init(elev=float(config.get("3d_elev", 24)), azim=float(config.get("3d_azim", -60)))
        _apply_axis_limits(ax, config, True)
        ax.set_title("Relabeled State Labels")
        ax.legend(frameon=False, fontsize=7)
        path = os.path.join(output_dir, f"relabeled_state_labels_3d.{fmt}")
    else:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
        for state in range(n_states):
            ids_i = np.intersect1d(np.flatnonzero(new_state == state), ids)
            if ids_i.size:
                ax.scatter(
                    cv_plane[ids_i, 0], cv_plane[ids_i, 1],
                    s=size + 4, color=cmap(state), alpha=alpha, linewidths=0,
                    label=names[state], rasterized=True,
                )
        ids_u = np.intersect1d(np.flatnonzero(new_state == -1), ids)
        if show_unlabeled and ids_u.size:
            ax.scatter(
                cv_plane[ids_u, 0], cv_plane[ids_u, 1],
                s=size, color="gray", alpha=alpha * 0.5, linewidths=0,
                label="unlabeled", rasterized=True,
            )
        ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1])
        _apply_axis_limits(ax, config, False)
        ax.set_title("Relabeled State Labels")
        ax.legend(frameon=False, fontsize=7)
        path = os.path.join(output_dir, f"relabeled_state_labels_2d.{fmt}")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    changed = _thin_indices(np.flatnonzero(proposal["changed_mask"]), max_highlight)
    review_mixed = _thin_indices(
        np.flatnonzero(proposal["review_mixed_mask"] & ~proposal["changed_mask"]),
        max_highlight,
    )
    review_missing = _thin_indices(
        np.flatnonzero(proposal["review_missing_mask"] & ~proposal["changed_mask"]),
        max_highlight,
    )
    if changed.size or review_mixed.size or review_missing.size:
        if is_3d:
            fig = plt.figure(figsize=(6.5, 5.5), dpi=160)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                cv_plane[ids, 0], cv_plane[ids, 1], cv_plane[ids, 2],
                s=max(size - 1, 1.0), color="lightgray", alpha=0.18, linewidths=0,
            )
            if review_mixed.size:
                ax.scatter(
                    cv_plane[review_mixed, 0], cv_plane[review_mixed, 1], cv_plane[review_mixed, 2],
                    s=size + 12, color="orange", marker="x", alpha=0.75, linewidths=0.8,
                    label="review mixed",
                )
            if review_missing.size:
                ax.scatter(
                    cv_plane[review_missing, 0], cv_plane[review_missing, 1], cv_plane[review_missing, 2],
                    s=size + 12, color="purple", marker="x", alpha=0.75, linewidths=0.8,
                    label="review missing signal",
                )
            if changed.size:
                ax.scatter(
                    cv_plane[changed, 0], cv_plane[changed, 1], cv_plane[changed, 2],
                    s=size + 18, color="red", alpha=0.85, linewidths=0,
                    label="changed",
                )
            ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1]); ax.set_zlabel(plane[2])
            ax.view_init(elev=float(config.get("3d_elev", 24)), azim=float(config.get("3d_azim", -60)))
            _apply_axis_limits(ax, config, True)
            path = os.path.join(output_dir, f"relabel_changed_review_3d.{fmt}")
        else:
            fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
            ax.scatter(
                cv_plane[ids, 0], cv_plane[ids, 1],
                s=max(size - 1, 1.0), color="lightgray", alpha=0.18,
                linewidths=0, rasterized=True,
            )
            if review_mixed.size:
                ax.scatter(
                    cv_plane[review_mixed, 0], cv_plane[review_mixed, 1],
                    s=size + 12, color="orange", marker="x", alpha=0.75,
                    linewidths=0.8, label="review mixed", rasterized=True,
                )
            if review_missing.size:
                ax.scatter(
                    cv_plane[review_missing, 0], cv_plane[review_missing, 1],
                    s=size + 12, color="purple", marker="x", alpha=0.75,
                    linewidths=0.8, label="review missing signal", rasterized=True,
                )
            if changed.size:
                ax.scatter(
                    cv_plane[changed, 0], cv_plane[changed, 1],
                    s=size + 18, color="red", alpha=0.85, linewidths=0,
                    label="changed", rasterized=True,
                )
            ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1])
            _apply_axis_limits(ax, config, False)
            path = os.path.join(output_dir, f"relabel_changed_review_2d.{fmt}")

        ax.set_title("Relabel Changes and Review Signals")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    return saved


def _rows_for_frames(indices, old_labels, new_labels, proposal, traj_id, frame_index, max_rows=None):
    if max_rows is not None and len(indices) > int(max_rows):
        indices = indices[: int(max_rows)]
    rows = []
    for idx in indices:
        row = {
            "frame_global_index": int(idx),
            "trajectory_index": int(traj_id[idx]),
            "frame_index": int(frame_index[idx]),
            "old_state": int(old_labels[idx]),
            "new_state": int(new_labels[idx]),
            "q_argmax": int(proposal["q_argmax"][idx]),
            "q_max": float(proposal["q_max"][idx]),
            "label_consistency": float(proposal["label_consistency"][idx])
            if np.isfinite(proposal["label_consistency"][idx]) else np.nan,
            "q_entropy_norm": float(proposal["entropy_norm"][idx]),
        }
        if "mean_lagged_entropy_norm" in proposal:
            value = proposal["mean_lagged_entropy_norm"][idx]
            row["mean_lagged_entropy_norm"] = float(value) if np.isfinite(value) else np.nan
        if "lagged_high_entropy_fraction" in proposal:
            value = proposal["lagged_high_entropy_fraction"][idx]
            row["lagged_high_entropy_fraction"] = float(value) if np.isfinite(value) else np.nan
        rows.append(row)
    return rows


def _compact_nonnegative_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    compact = labels.copy()
    valid_labels = sorted(int(label) for label in np.unique(labels[labels >= 0]))
    mapping = {old: new for new, old in enumerate(valid_labels)}
    for old, new in mapping.items():
        compact[labels == old] = new
    rows = [{"old_label": int(old), "new_label": int(new)} for old, new in mapping.items()]
    return compact, rows


def _save_dataset_like_input(dataset_path, output_path, pack, new_state, config, stride):
    import torch
    import yaml

    dataset_path = str(dataset_path)
    output_path = str(output_path)
    ext = os.path.splitext(dataset_path)[1].lower()
    out_ext = os.path.splitext(output_path)[1].lower() or ext
    ensure_dir(os.path.dirname(output_path) or ".")

    meta = dict(pack.meta)
    section_cfg = config.get("relabel", config.get("radical", {}))
    compact_labels = bool(section_cfg.get("compact_labels", config.get("compact_labels", True)))
    save_state = np.asarray(new_state, dtype=np.int64)
    label_mapping = []
    if compact_labels:
        save_state, label_mapping = _compact_nonnegative_labels(save_state)
    valid_labeled = save_state[save_state >= 0]
    relabeled_n_states = int(np.max(valid_labeled) + 1) if valid_labeled.size else 0
    if relabeled_n_states > 0:
        meta["k_selected"] = relabeled_n_states
        meta["n_states"] = relabeled_n_states
    meta["relabel"] = {
        "source_dataset": os.path.abspath(dataset_path),
        "dataset_stride": int(stride),
        "config": section_cfg,
        "compact_labels": compact_labels,
        "label_mapping": label_mapping,
        "n_states": relabeled_n_states,
    }

    if out_ext in {".pt", ".pth"}:
        out = {
            "features": pack.features.detach().cpu().float(),
            "weights": pack.weights.detach().cpu().float(),
            "meta_state": torch.as_tensor(save_state, dtype=torch.long),
            "meta": meta,
        }
        if pack.cv is not None:
            out["cv"] = pack.cv.detach().cpu().float()
        if pack.traj_id is not None:
            out["traj_id"] = pack.traj_id.detach().cpu().long()
        torch.save(out, output_path)
        return output_path

    if out_ext == ".npz":
        out = {
            "features": pack.features.detach().cpu().numpy().astype(np.float32),
            "weights": pack.weights.detach().cpu().numpy().astype(np.float32),
            "meta_state": save_state.astype(np.int64, copy=False),
            "meta_yaml": np.array([yaml.safe_dump(meta, sort_keys=False)], dtype=object),
        }
        if pack.cv is not None:
            out["cv"] = pack.cv.detach().cpu().numpy().astype(np.float32)
        if pack.traj_id is not None:
            out["traj_id"] = pack.traj_id.detach().cpu().numpy().astype(np.int64)
        np.savez_compressed(output_path, **out)
        return output_path

    raise ValueError("Relabeled dataset output must end in .pt, .pth, or .npz.")

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..common.config import ensure_dir
from ..common.data import cv_headers_for_pack


def state_names_for_plot(config, n_states):
    names = config.get("state_names", None)
    if names is None:
        return [f"state {i}" for i in range(n_states)]
    out = [str(n) for n in names]
    if len(out) < n_states:
        out.extend(f"state {i}" for i in range(len(out), n_states))
    return out


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
    missing = [n for n in plane if n not in cv_headers]
    if missing:
        raise ValueError(f"CV plane columns not in dataset: {missing}")
    indices = [cv_headers.index(n) for n in plane]
    return plane, indices, cv_data[:, indices]


def _finite_limits(*arrays):
    all_vals = np.concatenate([a.ravel() for a in arrays])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def plot_state_labels_cv(pack, q_values, results, config, out_dir):
    """Scatter plot of existing state labels in CV space."""
    if pack.cv is None:
        print("[PLOT] No CV data available; skipping state-label CV plot.")
        return []

    cv_headers = cv_headers_for_pack(pack)
    cv = pack.cv.numpy()
    plane, indices, cv_plane = _resolve_cv_plane(cv, cv_headers, config)
    state = pack.state.numpy().astype(np.int64)
    n_states = int(q_values.shape[1])
    fmt = str(config.get("format", "png"))
    max_points = int(config.get("max_scatter_points", 20000))
    alpha = float(config.get("scatter_alpha", 0.35))
    size = float(config.get("scatter_size", 4.0))
    names = state_names_for_plot(config, n_states)
    cmap = plt.get_cmap(str(config.get("basin_cmap", "tab10")), max(n_states, 1))
    saved = []

    if len(indices) == 3:
        x, y, z = cv_plane[:, 0], cv_plane[:, 1], cv_plane[:, 2]
        fig = plt.figure(figsize=(6.5, 5.5), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        ids = _thin_points(len(x), max_points)
        for i in range(n_states):
            mask = state == i
            ids_i = np.intersect1d(np.flatnonzero(mask), ids)
            if ids_i.size == 0:
                continue
            ax.scatter(x[ids_i], y[ids_i], z[ids_i], s=size + 6, color=cmap(i), alpha=alpha, linewidths=0, label=names[i])
        unlabeled = state == -1
        ids_u = np.intersect1d(np.flatnonzero(unlabeled), ids)
        if ids_u.size > 0:
            ax.scatter(x[ids_u], y[ids_u], z[ids_u], s=size, color="gray", alpha=alpha * 0.5, linewidths=0, label="unlabeled")
        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])
        ax.set_zlabel(plane[2])
        ax.set_title("State Labels (CV space)")
        ax.legend(frameon=False, fontsize=7)
        ax.view_init(elev=float(config.get("3d_elev", 24)), azim=float(config.get("3d_azim", -60)))
        path = os.path.join(out_dir, f"state_labels_3d.{fmt}")
    else:
        x, y = cv_plane[:, 0], cv_plane[:, 1]
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
        ids = _thin_points(len(x), max_points)
        for i in range(n_states):
            mask = state == i
            ids_i = np.intersect1d(np.flatnonzero(mask), ids)
            if ids_i.size == 0:
                continue
            ax.scatter(x[ids_i], y[ids_i], s=size + 6, color=cmap(i), alpha=alpha, linewidths=0, label=names[i], rasterized=True)
        unlabeled = state == -1
        ids_u = np.intersect1d(np.flatnonzero(unlabeled), ids)
        if ids_u.size > 0:
            ax.scatter(x[ids_u], y[ids_u], s=size, color="gray", alpha=alpha * 0.5, linewidths=0, label="unlabeled", rasterized=True)
        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])
        ax.set_title("State Labels (CV space)")
        ax.legend(frameon=False, fontsize=7)
        path = os.path.join(out_dir, f"state_labels_2d.{fmt}")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def plot_committor_confidence_cv(pack, q_values, results, config, out_dir):
    """Plot committor confidence and entropy in CV space."""
    if pack.cv is None:
        print("[PLOT] No CV data; skipping committor confidence CV plot.")
        return []

    cv_headers = cv_headers_for_pack(pack)
    cv = pack.cv.numpy()
    plane, indices, cv_plane = _resolve_cv_plane(cv, cv_headers, config)
    fmt = str(config.get("format", "png"))
    max_points = int(config.get("max_scatter_points", 20000))
    alpha = float(config.get("scatter_alpha", 0.4))
    size = float(config.get("scatter_size", 3.0))
    saved = []

    entropy_denominator = np.log(q_values.shape[1]) if q_values.shape[1] > 1 else 1.0
    confidence = 1.0 - (-np.sum(np.clip(q_values, 1e-12, 1.0) * np.log(np.clip(q_values, 1e-12, 1.0)), axis=1) / entropy_denominator)
    entropy_norm = 1.0 - confidence
    per_frame = results.get("_per_frame", {})

    fields = [
        ("committor_confidence", confidence, "viridis", "confidence c(x)"),
        ("committor_entropy_norm", entropy_norm, "inferno", "normalized entropy H_norm(x)"),
    ]
    if "mean_lagged_q_entropy_norm" in per_frame:
        fields.append((
            "lagged_committor_entropy_norm",
            per_frame["mean_lagged_q_entropy_norm"],
            "magma",
            "lagged normalized entropy H_norm(x+tau)",
        ))

    for field_name, field, cmap_name, cbar_label in fields:
        ids = _thin_points(len(cv_plane), max_points)
        if len(indices) == 3:
            x, y, z = cv_plane[ids, 0], cv_plane[ids, 1], cv_plane[ids, 2]
            fig = plt.figure(figsize=(6.5, 5.5), dpi=160)
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(x, y, z, c=field[ids], s=size, cmap=cmap_name, alpha=alpha, linewidths=0, vmin=0, vmax=1)
            ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1]); ax.set_zlabel(plane[2])
            ax.set_title(cbar_label)
            ax.view_init(elev=float(config.get("3d_elev", 24)), azim=float(config.get("3d_azim", -60)))
            path = os.path.join(out_dir, f"{field_name}_3d.{fmt}")
        else:
            x, y = cv_plane[ids, 0], cv_plane[ids, 1]
            fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
            sc = ax.scatter(x, y, c=field[ids], s=size, cmap=cmap_name, alpha=alpha, linewidths=0, vmin=0, vmax=1, rasterized=True)
            ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1])
            ax.set_title(cbar_label)
            path = os.path.join(out_dir, f"{field_name}_2d.{fmt}")

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cbar_label)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_uncertainty_categories_cv(pack, q_values, results, config, out_dir):
    """Highlight lagged-entropy uncertainty classes in CV space."""
    per_frame = results.get("_per_frame", {})
    if pack.cv is None or "uncertainty_category" not in per_frame:
        return []

    cv_headers = cv_headers_for_pack(pack)
    cv = pack.cv.numpy()
    plane, indices, cv_plane = _resolve_cv_plane(cv, cv_headers, config)
    fmt = str(config.get("format", "png"))
    max_points = int(config.get("max_scatter_points", 20000))
    alpha = float(config.get("scatter_alpha", 0.35))
    size = float(config.get("scatter_size", 3.0))
    category = np.asarray(per_frame["uncertainty_category"], dtype=np.int64)
    ids = _thin_points(cv_plane.shape[0], max_points)
    is_2d = len(indices) == 2

    styles = [
        (1, "mislabeled metastate", "tab:blue", "o"),
        (2, "missed metastate", "tab:red", "x"),
        (3, "transition state", "tab:orange", "^"),
        (4, "unresolved uncertain", "tab:purple", "s"),
    ]

    if is_2d:
        fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=160)
        ax.scatter(
            cv_plane[ids, 0],
            cv_plane[ids, 1],
            s=size,
            color="lightgray",
            alpha=alpha * 0.45,
            linewidths=0,
            rasterized=True,
        )
        for code, label, color, marker in styles:
            ids_c = ids[category[ids] == code]
            if ids_c.size == 0:
                continue
            ax.scatter(
                cv_plane[ids_c, 0],
                cv_plane[ids_c, 1],
                s=size + 12,
                color=color,
                alpha=min(alpha + 0.25, 1.0),
                linewidths=0.5,
                marker=marker,
                label=label,
                rasterized=True,
            )
        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])
        ax.set_title("Lagged-Entropy Uncertainty Classes")
        path = os.path.join(out_dir, f"uncertainty_categories_2d.{fmt}")
    else:
        fig = plt.figure(figsize=(6.7, 5.6), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            cv_plane[ids, 0],
            cv_plane[ids, 1],
            cv_plane[ids, 2],
            s=size,
            color="lightgray",
            alpha=alpha * 0.35,
            linewidths=0,
        )
        for code, label, color, marker in styles:
            ids_c = ids[category[ids] == code]
            if ids_c.size == 0:
                continue
            ax.scatter(
                cv_plane[ids_c, 0],
                cv_plane[ids_c, 1],
                cv_plane[ids_c, 2],
                s=size + 18,
                color=color,
                alpha=min(alpha + 0.3, 1.0),
                linewidths=0.5,
                marker=marker,
                label=label,
            )
        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])
        ax.set_zlabel(plane[2])
        ax.set_title("Lagged-Entropy Uncertainty Classes")
        ax.view_init(elev=float(config.get("3d_elev", 24)), azim=float(config.get("3d_azim", -60)))
        path = os.path.join(out_dir, f"uncertainty_categories_3d.{fmt}")

    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def plot_candidate_summary_cv(pack, q_values, results, config, out_dir):
    """Highlight candidate regions on CV scatter."""
    if pack.cv is None:
        return []

    cv_headers = cv_headers_for_pack(pack)
    cv = pack.cv.numpy()
    plane, indices, cv_plane = _resolve_cv_plane(cv, cv_headers, config)
    state = pack.state.numpy().astype(np.int64)
    fmt = str(config.get("format", "png"))
    max_points = int(config.get("max_scatter_points", 20000))
    alpha = float(config.get("scatter_alpha", 0.3))
    size = float(config.get("scatter_size", 3.0))
    saved = []
    n_states = int(q_values.shape[1])

    split_states = {c["state"] for c in results.get("split_candidates", [])}
    merge_states = set()
    for c in results.get("merge_candidates", []):
        merge_states.add(c.get("state_i", c.get("states", [None, None])[0]) if isinstance(c.get("states"), list) else c.get("state_i"))
        merge_states.add(c.get("state_j", c.get("states", [None, None])[1]) if isinstance(c.get("states"), list) else c.get("state_j"))

    is_2d = len(indices) == 2
    for candidate_type, highlight_states, color, label in [
        ("split", split_states, "red", "split candidate"),
        ("merge", merge_states, "orange", "merge candidate"),
    ]:
        if not highlight_states:
            continue

        ids = _thin_points(cv_plane.shape[0], max_points)
        if is_2d:
            x, y = cv_plane[ids, 0], cv_plane[ids, 1]
            fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
            ax.scatter(x, y, s=size, color="lightgray", alpha=alpha * 0.5, linewidths=0, rasterized=True)
            for si in sorted(highlight_states):
                mask = state == si
                ids_i = np.intersect1d(np.flatnonzero(mask), ids)
                if ids_i.size == 0:
                    continue
                ax.scatter(cv_plane[ids_i, 0], cv_plane[ids_i, 1], s=size + 8, color=color, alpha=alpha + 0.2, linewidths=0, label=f"state {si}", rasterized=True)
            ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1])
            ax.set_title(f"{label}s")
            ax.legend(frameon=False, fontsize=7)
            path = os.path.join(out_dir, f"candidates_{candidate_type}_2d.{fmt}")
        else:
            x, y, z = cv_plane[ids, 0], cv_plane[ids, 1], cv_plane[ids, 2]
            fig = plt.figure(figsize=(6.5, 5.5), dpi=160)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x, y, z, s=size, color="lightgray", alpha=alpha * 0.3, linewidths=0)
            for si in sorted(highlight_states):
                mask = state == si
                ids_i = np.intersect1d(np.flatnonzero(mask), ids)
                if ids_i.size == 0:
                    continue
                ax.scatter(cv_plane[ids_i, 0], cv_plane[ids_i, 1], cv_plane[ids_i, 2], s=size + 15, color=color, alpha=alpha + 0.3, linewidths=0, label=f"state {si}")
            ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1]); ax.set_zlabel(plane[2])
            ax.set_title(f"{label}s")
            ax.legend(frameon=False, fontsize=7)
            ax.view_init(elev=float(config.get("3d_elev", 24)), azim=float(config.get("3d_azim", -60)))
            path = os.path.join(out_dir, f"candidates_{candidate_type}_3d.{fmt}")

        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_missing_candidates_cv(pack, q_values, results, config, out_dir):
    """Highlight missing-state candidate regions in CV space."""
    if pack.cv is None:
        return []

    missing = results.get("missing_state_candidates", [])
    if not missing:
        return []

    cv_headers = cv_headers_for_pack(pack)
    cv = pack.cv.numpy()
    plane, indices, cv_plane = _resolve_cv_plane(cv, cv_headers, config)
    core_labels = pack.state.numpy().astype(np.int64)
    fmt = str(config.get("format", "png"))
    max_points = int(config.get("max_scatter_points", 20000))
    alpha = float(config.get("scatter_alpha", 0.3))
    size = float(config.get("scatter_size", 3.0))
    n_states = int(q_values.shape[1])
    cmap = plt.get_cmap(str(config.get("basin_cmap", "tab10")), max(n_states, 1))

    is_2d = len(indices) == 2
    ids = _thin_points(cv_plane.shape[0], max_points)

    noncore = core_labels == -1
    ids_nc = np.intersect1d(np.flatnonzero(noncore), ids)

    if is_2d:
        x, y = cv_plane[:, 0], cv_plane[:, 1]
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=160)
        for i in range(n_states):
            mask = core_labels == i
            ids_i = np.intersect1d(np.flatnonzero(mask), ids)
            if ids_i.size == 0:
                continue
            ax.scatter(x[ids_i], y[ids_i], s=size, color=cmap(i), alpha=alpha, linewidths=0, rasterized=True)
        if ids_nc.size > 0:
            ax.scatter(x[ids_nc], y[ids_nc], s=size + 10, color="red", alpha=alpha + 0.2, linewidths=0, marker="x", label="non-core (missing candidate source)", rasterized=True)
        ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1])
        ax.set_title("Missing-State Candidate Regions")
        ax.legend(frameon=False, fontsize=7)
        path = os.path.join(out_dir, f"missing_candidates_2d.{fmt}")
    else:
        x, y, z = cv_plane[:, 0], cv_plane[:, 1], cv_plane[:, 2]
        fig = plt.figure(figsize=(6.5, 5.5), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        for i in range(n_states):
            mask = core_labels == i
            ids_i = np.intersect1d(np.flatnonzero(mask), ids)
            if ids_i.size == 0:
                continue
            ax.scatter(x[ids_i], y[ids_i], z[ids_i], s=size, color=cmap(i), alpha=alpha, linewidths=0)
        if ids_nc.size > 0:
            ax.scatter(x[ids_nc], y[ids_nc], z[ids_nc], s=size + 20, color="red", alpha=alpha + 0.3, linewidths=0, marker="x", label="non-core")
        ax.set_xlabel(plane[0]); ax.set_ylabel(plane[1]); ax.set_zlabel(plane[2])
        ax.set_title("Missing-State Candidate Regions")
        ax.legend(frameon=False, fontsize=7)
        ax.view_init(elev=float(config.get("3d_elev", 24)), azim=float(config.get("3d_azim", -60)))
        path = os.path.join(out_dir, f"missing_candidates_3d.{fmt}")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def plot_basin_kinetic_groups_cv(pack, q_values, results, config, out_dir):
    """Plot spectral metastability groups inside each label core."""
    per_frame = results.get("_per_frame", {})
    group_labels = per_frame.get("basin_kinetic_group")
    group_rows = results.get("basin_kinetic_groups", [])
    state_rows = results.get("basin_kinetic_state_stats", [])
    if pack.cv is None or group_labels is None or not group_rows:
        return []

    cv_headers = cv_headers_for_pack(pack)
    cv = pack.cv.numpy()
    plane, indices, cv_plane = _resolve_cv_plane(cv, cv_headers, config)
    state = pack.state.numpy().astype(np.int64)
    group_labels = np.asarray(group_labels, dtype=np.int64)
    fmt = str(config.get("format", "png"))
    max_points = int(config.get("max_scatter_points", 20000))
    alpha = float(config.get("scatter_alpha", 0.35))
    size = float(config.get("scatter_size", 3.0))
    saved = []
    is_2d = len(indices) == 2
    states_to_plot = sorted({int(row["state"]) for row in group_rows})
    group_cmap = plt.get_cmap(str(config.get("basin_group_cmap", "tab20")))
    state_meta = {int(row["state"]): row for row in state_rows}

    for state_id in states_to_plot:
        groups = [row for row in group_rows if int(row["state"]) == state_id]
        group_ids = [int(row["kinetic_group"]) for row in groups]
        if not group_ids:
            continue
        state_ids = np.flatnonzero(state == state_id)
        if state_ids.size == 0:
            continue
        ids_bg = _thin_points(cv_plane.shape[0], max_points)
        ids_bg = np.intersect1d(ids_bg, state_ids)
        meta = state_meta.get(state_id, {})
        title = (
            f"State {state_id} Spectral Groups: "
            f"k={meta.get('suggested_groups', len(group_ids))}, "
            f"eigengap={float(meta.get('eigengap_score', 0.0)):.3f}, "
            f"{meta.get('split_confidence', 'unknown')}"
        )

        if is_2d:
            fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=160)
            ax.scatter(
                cv_plane[ids_bg, 0],
                cv_plane[ids_bg, 1],
                s=size,
                color="lightgray",
                alpha=alpha * 0.45,
                linewidths=0,
                rasterized=True,
            )
            for idx, group_id in enumerate(group_ids):
                ids_g = np.flatnonzero(group_labels == group_id)
                ids_g = _thin_points(ids_g.size, max_points // max(1, len(group_ids)))
                frame_ids = np.flatnonzero(group_labels == group_id)[ids_g]
                if frame_ids.size == 0:
                    continue
                ax.scatter(
                    cv_plane[frame_ids, 0],
                    cv_plane[frame_ids, 1],
                    s=size + 12,
                    color=group_cmap(idx % group_cmap.N),
                    alpha=min(alpha + 0.3, 1.0),
                    linewidths=0,
                    label=f"group {idx}",
                    rasterized=True,
                )
            ax.set_xlabel(plane[0])
            ax.set_ylabel(plane[1])
            path = os.path.join(out_dir, f"basin_kinetic_groups_state_{state_id}_2d.{fmt}")
        else:
            fig = plt.figure(figsize=(6.7, 5.6), dpi=160)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                cv_plane[ids_bg, 0],
                cv_plane[ids_bg, 1],
                cv_plane[ids_bg, 2],
                s=size,
                color="lightgray",
                alpha=alpha * 0.35,
                linewidths=0,
            )
            for idx, group_id in enumerate(group_ids):
                frame_ids_all = np.flatnonzero(group_labels == group_id)
                keep = _thin_points(frame_ids_all.size, max_points // max(1, len(group_ids)))
                frame_ids = frame_ids_all[keep]
                if frame_ids.size == 0:
                    continue
                ax.scatter(
                    cv_plane[frame_ids, 0],
                    cv_plane[frame_ids, 1],
                    cv_plane[frame_ids, 2],
                    s=size + 18,
                    color=group_cmap(idx % group_cmap.N),
                    alpha=min(alpha + 0.3, 1.0),
                    linewidths=0,
                    label=f"group {idx}",
                )
            ax.set_xlabel(plane[0])
            ax.set_ylabel(plane[1])
            ax.set_zlabel(plane[2])
            ax.view_init(elev=float(config.get("3d_elev", 24)), azim=float(config.get("3d_azim", -60)))
            path = os.path.join(out_dir, f"basin_kinetic_groups_state_{state_id}_3d.{fmt}")

        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_basin_kinetic_eigenvalues(results, config, out_dir):
    """Plot local MSM eigenvalue spectra used for split diagnosis."""
    rows = results.get("basin_kinetic_state_stats", [])
    rows = [row for row in rows if row.get("eigenvalues")]
    if not rows:
        return []

    fmt = str(config.get("format", "png"))
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=160)
    for row in rows:
        vals = [float(item) for item in str(row.get("eigenvalues", "")).split(";") if item]
        if not vals:
            continue
        x = np.arange(len(vals), dtype=np.int64)
        label = f"state {row['state']} k={row.get('suggested_groups', 1)}"
        ax.plot(x, vals, marker="o", linewidth=1.2, markersize=3.5, label=label)
    ax.axhline(float(config.get("basin_kinetic_groups", {}).get("min_slow_eigenvalue", 0.80)), color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("|lambda|")
    ax.set_ylim(0, 1.02)
    ax.set_title("Within-Label Local MSM Eigenvalues")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, f"basin_kinetic_eigenvalues.{fmt}")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def plot_confidence_histograms(per_frame, results, config, out_dir):
    """Histograms of committor confidence, label consistency, and entropy."""
    fmt = str(config.get("format", "png"))
    bins = int(config.get("histogram_bins", 50))
    saved = []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=160)

    if "committor_confidence" in per_frame:
        axes[0].hist(per_frame["committor_confidence"], bins=bins, range=(0, 1), color="steelblue", edgecolor="white", linewidth=0.3)
        axes[0].set_xlabel("committor confidence c(x)")
        axes[0].set_ylabel("count")
        axes[0].set_title("Committor Confidence")

    if "label_consistency" in per_frame:
        valid = per_frame["label_consistency"][~np.isnan(per_frame["label_consistency"])]
        if valid.size > 0:
            axes[1].hist(valid, bins=bins, range=(0, 1), color="darkgreen", edgecolor="white", linewidth=0.3)
        axes[1].set_xlabel("label consistency")
        axes[1].set_ylabel("count")
        axes[1].set_title("Label-Committor Consistency")

    if "q_entropy_norm" in per_frame:
        axes[2].hist(per_frame["q_entropy_norm"], bins=bins, range=(0, 1), color="darkred", edgecolor="white", linewidth=0.3)
        axes[2].set_xlabel("normalized entropy H_norm")
        axes[2].set_ylabel("count")
        axes[2].set_title("Committor Entropy")

    fig.tight_layout()
    path = os.path.join(out_dir, f"confidence_histograms.{fmt}")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def plot_state_confidence_bars(results, config, out_dir):
    """Bar chart of per-state mean q_own and fraction low consistency."""
    state_conf = results.get("state_confidence", [])
    if not state_conf:
        return []

    fmt = str(config.get("format", "png"))
    states = [row["state"] for row in state_conf]
    mean_q = [row.get("mean_q_own", 0) for row in state_conf]
    frac_low = [row.get("fraction_low_consistency", 0) for row in state_conf]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    colors = [plt.get_cmap("tab10")(s % 10) for s in states]

    ax1.bar(states, mean_q, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("state")
    ax1.set_ylabel("mean q_own")
    ax1.set_title("Mean Own-State Committor")
    ax1.set_ylim(0, 1)

    ax2.bar(states, frac_low, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("state")
    ax2.set_ylabel("fraction low consistency")
    ax2.set_title("Fraction with q_own < cutoff")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    path = os.path.join(out_dir, f"state_confidence_bars.{fmt}")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def _thin_points(n_points, max_points):
    if n_points <= max_points:
        return np.arange(n_points, dtype=np.int64)
    return np.linspace(0, n_points - 1, max_points, dtype=np.int64)


def run_plots(pack, q_values, results, config, out_dir):
    ensure_dir(out_dir)
    saved = []
    saved.extend(plot_state_labels_cv(pack, q_values, results, config, out_dir))
    saved.extend(plot_committor_confidence_cv(pack, q_values, results, config, out_dir))
    saved.extend(plot_uncertainty_categories_cv(pack, q_values, results, config, out_dir))
    saved.extend(plot_candidate_summary_cv(pack, q_values, results, config, out_dir))
    saved.extend(plot_missing_candidates_cv(pack, q_values, results, config, out_dir))
    saved.extend(plot_basin_kinetic_groups_cv(pack, q_values, results, config, out_dir))
    saved.extend(plot_basin_kinetic_eigenvalues(results, config, out_dir))

    per_frame = _build_per_frame_from_results(results)
    saved.extend(plot_confidence_histograms(per_frame, results, config, out_dir))
    saved.extend(plot_state_confidence_bars(results, config, out_dir))

    print(f"[PLOT] Saved {len(saved)} diagnostic plots to {out_dir}")
    for p in saved:
        print(f"  {p}")
    return saved


def _build_per_frame_from_results(results):
    return results.get("_per_frame", {})

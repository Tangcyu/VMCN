from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import (
    apply_stride,
    cv_headers_for_pack,
    infer_n_states,
    load_dataset,
    select_model_inputs,
)
from ..next_hit.predict import infer_probabilities, load_committor_model


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
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
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


def propose_relabeling(q_values, state_labels, config):
    confidence_cfg = config.get("confidence", {})
    relabel_cfg = config.get("relabel", {})

    q_label_cutoff = float(confidence_cfg.get("q_label_cutoff", 0.7))
    frac_low_cutoff = float(confidence_cfg.get("state_fraction_low_cutoff", 0.2))
    mean_q_cutoff = float(confidence_cfg.get("state_mean_q_cutoff", 0.75))
    entropy_cutoff = float(confidence_cfg.get("entropy_cutoff_ambiguous", 0.5))
    dominance_cutoff = float(confidence_cfg.get("reassign_dominance_cutoff", 0.65))
    reassign_q_cutoff = float(relabel_cfg.get("reassign_q_cutoff", q_label_cutoff))
    missing_qmax_cutoff = float(relabel_cfg.get("missing_qmax_cutoff", q_label_cutoff))
    mark_ambiguous = bool(relabel_cfg.get("mark_ambiguous_as_unlabeled", False))

    state_labels = np.asarray(state_labels, dtype=np.int64)
    proposed = state_labels.copy()
    n_states = int(q_values.shape[1])
    q_max, q_argmax, _, entropy_norm = _entropy_confidence(q_values)

    label_consistency = np.full(state_labels.shape[0], np.nan, dtype=np.float64)
    valid = (state_labels >= 0) & (state_labels < n_states)
    idx = np.flatnonzero(valid)
    label_consistency[valid] = q_values[idx, state_labels[valid]]

    changed_mask = np.zeros(state_labels.shape[0], dtype=bool)
    review_mixed_mask = np.zeros(state_labels.shape[0], dtype=bool)
    review_missing_mask = (entropy_norm >= entropy_cutoff) & (q_max < missing_qmax_cutoff)
    if mark_ambiguous:
        proposed[review_missing_mask] = -1
        changed_mask |= review_missing_mask & (state_labels != -1)

    action_rows = []
    state_rows = []

    for state in range(n_states):
        mask = state_labels == state
        n_state = int(np.sum(mask))
        if n_state == 0:
            continue

        own_q = q_values[mask, state]
        low_mask = mask & np.isfinite(label_consistency) & (label_consistency < q_label_cutoff)
        n_low = int(np.sum(low_mask))
        frac_low = float(n_low / max(1, n_state))
        mean_q_own = float(np.mean(own_q))
        mean_entropy = float(np.mean(entropy_norm[mask]))

        dominant_alt = -1
        dominant_alt_fraction = 0.0
        n_reassigned = 0
        issue = "no_confidence_issue"
        action = "keep"

        if n_low:
            alt = q_argmax[low_mask]
            alt = alt[alt != state]
            if alt.size:
                counts = np.bincount(alt, minlength=n_states)
                dominant_alt = int(np.argmax(counts))
                dominant_alt_fraction = float(counts[dominant_alt] / n_low)

        unreliable = frac_low >= frac_low_cutoff or mean_q_own < mean_q_cutoff
        ambiguous = mean_entropy >= entropy_cutoff

        if unreliable and dominant_alt >= 0 and dominant_alt_fraction >= dominance_cutoff:
            reassign_mask = (
                low_mask
                & (q_argmax == dominant_alt)
                & (q_values[:, dominant_alt] >= reassign_q_cutoff)
            )
            proposed[reassign_mask] = dominant_alt
            changed_mask |= reassign_mask & (state_labels != dominant_alt)
            n_reassigned = int(np.sum(reassign_mask))
            issue = "possible_reassignment_or_merge"
            action = "reassign_low_consistency_frames"
        elif unreliable:
            review_mixed_mask |= low_mask
            issue = "ambiguous_or_missing_coordinate" if ambiguous else "possible_mislabel_or_broad_state"
            action = "review_for_split_or_descriptor_issue"
        elif ambiguous:
            issue = "transition_like_or_low_confidence"
            action = "review_entropy"

        state_rows.append({
            "state": state,
            "n_frames": n_state,
            "n_low_consistency": n_low,
            "fraction_low_consistency": frac_low,
            "mean_q_own": mean_q_own,
            "mean_entropy_norm": mean_entropy,
            "dominant_alternative_state": dominant_alt,
            "dominant_alternative_fraction": dominant_alt_fraction,
            "issue": issue,
            "action": action,
            "n_reassigned": n_reassigned,
        })

        if action != "keep":
            action_rows.append(state_rows[-1])

    return {
        "proposed_labels": proposed,
        "changed_mask": changed_mask,
        "review_mixed_mask": review_mixed_mask,
        "review_missing_mask": review_missing_mask,
        "state_actions": state_rows,
        "action_rows": action_rows,
        "q_max": q_max,
        "q_argmax": q_argmax,
        "entropy_norm": entropy_norm,
        "label_consistency": label_consistency,
    }


def _rows_for_frames(indices, old_labels, new_labels, proposal, traj_id, frame_index, max_rows=None):
    if max_rows is not None and len(indices) > int(max_rows):
        indices = indices[: int(max_rows)]
    rows = []
    for idx in indices:
        rows.append({
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
        })
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


def run_apply_relabel(dataset_path, model_path, config, device="cuda:0", batch_size=65536, dataset_stride=1):
    from .label_diagnostics import _compute_frame_index

    device_obj = setup_device(device)
    stride = int(dataset_stride)
    pack = apply_stride(load_dataset(dataset_path), stride)
    n_states = infer_n_states(pack, config.get("n_states", None))
    model_features, _ = select_model_inputs(pack, config)

    model = load_committor_model(model_path, device_obj)
    q_values = infer_probabilities(model, model_features.float(), device_obj, batch_size=int(batch_size))
    if q_values.ndim != 2 or q_values.shape[1] != n_states:
        raise RuntimeError(f"Model returned q shape {q_values.shape}, expected (_, {n_states}).")

    state = pack.state.detach().cpu().numpy().astype(np.int64)
    traj_id = (
        pack.traj_id.detach().cpu().numpy().astype(np.int64)
        if pack.traj_id is not None
        else np.zeros(state.shape[0], dtype=np.int64)
    )
    frame_index = _compute_frame_index(traj_id)

    proposal = propose_relabeling(q_values, state, config)
    new_state = proposal["proposed_labels"]
    changed = np.flatnonzero(proposal["changed_mask"])
    review_mixed = np.flatnonzero(proposal["review_mixed_mask"] & ~proposal["changed_mask"])
    review_missing = np.flatnonzero(proposal["review_missing_mask"] & ~proposal["changed_mask"])

    output_dir = ensure_dir(config.get("output_dir", "relabel_out"))
    relabel_cfg = config.get("relabel", {})
    default_output = os.path.join(output_dir, f"relabeled_dataset{Path(str(dataset_path)).suffix or '.pt'}")
    output_dataset = relabel_cfg.get("output_dataset", default_output)

    if bool(relabel_cfg.get("write_relabel_dataset", True)):
        saved_dataset = _save_dataset_like_input(dataset_path, output_dataset, pack, new_state, config, stride)
    else:
        saved_dataset = None

    max_review = relabel_cfg.get("max_review_frames", 20000)
    _write_csv(os.path.join(output_dir, "relabel_actions.csv"), proposal["action_rows"])
    _write_csv(
        os.path.join(output_dir, "relabel_changed_frames.csv"),
        _rows_for_frames(changed, state, new_state, proposal, traj_id, frame_index),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_review_mixed_frames.csv"),
        _rows_for_frames(review_mixed, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    _write_csv(
        os.path.join(output_dir, "relabel_review_missing_signal_frames.csv"),
        _rows_for_frames(review_missing, state, new_state, proposal, traj_id, frame_index, max_review),
    )
    saved_plots = []
    if bool(config.get("make_plots", True)) and bool(relabel_cfg.get("make_relabel_plots", True)):
        saved_plots = plot_relabel_cv(pack, state, new_state, proposal, config, output_dir)

    summary = {
        "dataset": os.path.abspath(str(dataset_path)),
        "model": os.path.abspath(str(model_path)),
        "output_dataset": None if saved_dataset is None else os.path.abspath(saved_dataset),
        "dataset_stride": stride,
        "n_frames": int(state.shape[0]),
        "n_changed_frames": int(changed.size),
        "n_review_mixed_frames": int(review_mixed.size),
        "n_review_missing_signal_frames": int(review_missing.size),
        "plots": [os.path.abspath(path) for path in saved_plots],
        "state_actions": proposal["state_actions"],
        "notes": [
            "Automatic changes are limited to dominant q-argmax reassignment among low-consistency frames.",
            "Mixed-destination and high-entropy/no-strong-destination frames are review signals only.",
            "If dataset_stride > 1, the output dataset contains the strided analysis frames only.",
        ],
    }
    write_yaml(summary, os.path.join(output_dir, "relabel_apply_summary.yaml"))
    print(f"[RELABEL] Changed frames: {changed.size}")
    print(f"[RELABEL] Review mixed frames: {review_mixed.size}")
    print(f"[RELABEL] Review missing-signal frames: {review_missing.size}")
    if saved_dataset is not None:
        print(f"[RELABEL] Saved relabeled dataset: {saved_dataset}")
    if saved_plots:
        print("[RELABEL] Saved relabel plots:")
        for path in saved_plots:
            print(f"  {path}")
    print(f"[RELABEL] Summary: {os.path.join(output_dir, 'relabel_apply_summary.yaml')}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Apply conservative confidence-based relabel proposals."
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = select_section(raw, "RELABEL", "TENSORQ_RELABEL")
    dataset_path = cfg.get("dataset", cfg.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("Relabel config needs 'dataset' or 'dataset_path'.")
    model_path = cfg.get("model")
    if model_path is None:
        raise KeyError("Relabel config needs 'model' (path to trained checkpoint).")

    cfg["output_dir"] = ensure_dir(cfg.get("output_dir", cfg.get("out_dir", "relabel")))
    run_apply_relabel(
        dataset_path=dataset_path,
        model_path=model_path,
        config=cfg,
        device=str(cfg.get("device", "cuda:0")),
        batch_size=int(cfg.get("batch_size", 65536)),
        dataset_stride=int(cfg.get("dataset_stride", 1)),
    )


if __name__ == "__main__":
    main()

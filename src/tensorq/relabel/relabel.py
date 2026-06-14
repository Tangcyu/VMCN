from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from .config_utils import _relabel_cfg, _sample_indices, _select_graph_features, _standardize_features
from .density import _knn_missing_components, _select_weighted_density_core
from .entropy import _classify_lagged_entropy_candidates, _remove_inconsistent_states
from .kinetics import (
    _iteratively_merge_kinetically_duplicate_labels,
    _iteratively_merge_mixed_new_labels,
    _reshape_existing_basins_from_kinetic_groups,
    _split_labels_by_final_knn_kinetics,
)
from .utils import _entropy_confidence, _save_dataset_like_input
from .visualization import plot_relabel_diagnostics
from .settings import analysis_settings


def _mask_count(mask):
    return int(np.sum(np.asarray(mask, dtype=bool)))


def _elapsed_seconds(start):
    return round(float(time.perf_counter() - start), 6)


def _relabel_diagnostics(config, graph_idx, masks, tables, lagged_entropy_classification, relabel_cfg):
    counts = {name: _mask_count(mask) for name, mask in masks.items()}
    return {
        "pipeline": ["entropy", "density", "kinetics"],
        "counts": counts,
        "candidate_frames": counts["missing_metastate_candidate"],
        "graph_frames": int(np.asarray(graph_idx, dtype=np.int64).size),
        "lagged_entropy_classification": lagged_entropy_classification,
        "knn_backend_config": {
            "knn_backend": str(relabel_cfg.get("knn_backend", "auto")),
            "knn_device": str(relabel_cfg.get("knn_device", relabel_cfg.get("device", "cuda:0"))),
            "torch_knn_auto_max_pairs": int(relabel_cfg.get("torch_knn_auto_max_pairs", 1_000_000_000)),
            "torch_knn_query_batch": int(relabel_cfg.get("torch_knn_query_batch", 4096)),
            "torch_knn_reference_batch": int(relabel_cfg.get("torch_knn_reference_batch", 32768)),
        },
        "n_new_core_components": len(tables["new_core_components"]),
        "n_removed_states": len(tables["removed_states"]),
        "removed_state_ids": [
            int(row["removed_state"]) for row in tables["removed_states"]
        ],
        "n_merged_new_states": len(tables["merged_new_states"]),
        "n_reshaped_basin_groups": len(tables["reshaped_basin_groups"]),
        "n_final_kinetic_merges": len(tables["final_kinetic_merges"]),
        "n_final_kinetic_splits": len(tables["final_kinetic_splits"]),
        "promote_missing_metastate_candidates": bool(
            relabel_cfg.get("promote_missing_metastate_candidates", True)
        ),
    }


def propose_relabeling(
    q_values,
    state_labels,
    graph_features,
    weights,
    config,
    trajectory_index=None,
    frame_index=None,
):
    """Propose labels with the compact entropy -> density -> kinetics pipeline."""
    proposal_start = time.perf_counter()
    stage_start = proposal_start
    timings = {}

    def mark(stage):
        nonlocal stage_start
        timings[stage] = _elapsed_seconds(stage_start)
        stage_start = time.perf_counter()

    settings = analysis_settings(config)
    relabel_cfg = _relabel_cfg(config)
    if "knn_device" not in relabel_cfg and "device" in config:
        relabel_cfg["knn_device"] = str(config["device"])

    q_label_cutoff = float(settings["q_cutoff"])
    entropy_cutoff = float(settings["entropy_cutoff"])
    candidate_qmax_cutoff = float(relabel_cfg.get("candidate_current_qmax_cutoff", q_label_cutoff))
    candidate_require_low_qmax = bool(relabel_cfg.get("candidate_require_current_low_qmax", False))

    state_labels = np.asarray(state_labels, dtype=np.int64)
    q_values = np.asarray(q_values, dtype=np.float64)
    new_state = state_labels.copy()
    n_frames = int(state_labels.shape[0])
    n_states = int(q_values.shape[1])
    q_max, q_argmax, _, entropy_norm = _entropy_confidence(q_values)
    weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    trajectory_arr = None if trajectory_index is None else np.asarray(trajectory_index, dtype=np.int64)
    frame_arr = None if frame_index is None else np.asarray(frame_index, dtype=np.int64)

    label_consistency = np.full(n_frames, np.nan, dtype=np.float64)
    valid = (state_labels >= 0) & (state_labels < n_states)
    valid_idx = np.flatnonzero(valid)
    label_consistency[valid] = q_values[valid_idx, state_labels[valid]]
    mark("proposal_prepare_entropy_scores")

    new_state, removed_mask, removed_states, removed_state_ids = _remove_inconsistent_states(
        new_state,
        state_labels,
        q_values,
        label_consistency,
        entropy_norm,
        config,
    )
    mark("entropy_remove_inconsistent_states")

    current_entropy_candidate_mask = entropy_norm >= entropy_cutoff
    if candidate_require_low_qmax:
        current_entropy_candidate_mask &= q_max <= candidate_qmax_cutoff

    ambiguous_to_unlabeled_mask = current_entropy_candidate_mask & (new_state >= 0)
    new_state[ambiguous_to_unlabeled_mask] = -1
    mark("entropy_unlabel_high_H")

    current_entropy_unlabeled = current_entropy_candidate_mask & (new_state == -1)
    (
        missing_metastate_h_tau_mask,
        transition_like_candidate_mask,
        unresolved_lagged_candidate_mask,
        mean_lagged_entropy,
        lagged_high_entropy_fraction,
        lagged_entropy_classification,
    ) = _classify_lagged_entropy_candidates(
        current_entropy_unlabeled,
        entropy_norm,
        trajectory_arr,
        frame_arr,
        config,
    )
    mark("entropy_lagged_H_tau_classification")

    z = _standardize_features(graph_features)
    mark("feature_standardization_for_graph")
    raw_candidate_idx = np.flatnonzero(missing_metastate_h_tau_mask)
    raw_lagged_score = np.nan_to_num(
        lagged_high_entropy_fraction[raw_candidate_idx],
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    raw_candidate_scores = (
        entropy_norm[raw_candidate_idx]
        * np.clip(raw_lagged_score, 0.0, 1.0)
        * (1.0 - np.clip(q_max[raw_candidate_idx], 0.0, 1.0))
        * (weights[raw_candidate_idx] if weights is not None else 1.0)
    )
    sampled_candidate_idx = _sample_indices(
        raw_candidate_idx,
        raw_candidate_scores,
        int(relabel_cfg.get("max_graph_frames", 20000)),
        int(relabel_cfg.get("random_seed", 0)),
    )
    candidate_mask = np.zeros(n_frames, dtype=bool)
    candidate_mask[sampled_candidate_idx] = True
    candidate_idx = np.flatnonzero(candidate_mask)
    mark("density_candidate_sampling")

    graph_idx = candidate_idx
    persistent_candidate_review_mask = np.zeros(n_frames, dtype=bool)
    if not bool(relabel_cfg.get("promote_missing_metastate_candidates", True)):
        persistent_candidate_review_mask = candidate_mask.copy()
        graph_idx = np.zeros(0, dtype=np.int64)

    component_rows = []
    component_frames = []
    if graph_idx.size:
        component_rows, component_frames = _knn_missing_components(
            z,
            graph_idx,
            q_values,
            q_max,
            q_argmax,
            entropy_norm,
            weights,
            relabel_cfg,
        )
    mark("density_knn_missing_components")

    next_label = int(np.max(state_labels[state_labels >= 0]) + 1) if np.any(state_labels >= 0) else 0
    new_core_mask = np.zeros(n_frames, dtype=bool)
    density_shell_mask = np.zeros(n_frames, dtype=bool)
    provisional_new_labels = []
    for row, frame_idx in zip(component_rows, component_frames):
        row["new_state"] = int(next_label)
        row["provisional_new_state"] = int(next_label)
        row["component_candidate_frames"] = int(frame_idx.size)
        row["component_candidate_weight"] = (
            float(np.sum(weights[frame_idx])) if weights is not None else float(frame_idx.size)
        )
        new_state[frame_idx] = next_label
        provisional_new_labels.append(next_label)
        next_label += 1
    mark("density_provisional_label_assignment")

    new_state, merged_new_state_rows = _iteratively_merge_mixed_new_labels(
        new_state,
        provisional_new_labels,
        trajectory_arr,
        frame_arr,
        weights,
        config,
    )
    mark("kinetics_merge_mixed_new_labels")

    component_final_labels = {
        int(row["provisional_new_state"]): int(new_state[component_frames[idx][0]])
        for idx, row in enumerate(component_rows)
        if len(component_frames[idx]) > 0
    }
    preselected_core_by_label = {}
    for idx, row in enumerate(component_rows):
        if len(component_frames[idx]) == 0:
            continue
        provisional = int(row["provisional_new_state"])
        final_label = component_final_labels.get(provisional, provisional)
        preselected_core_by_label[int(final_label)] = (
            preselected_core_by_label.get(int(final_label), False)
            or bool(row.get("preselected_density_core", False))
        )

    final_density = {}
    active_final_labels = sorted({label for label in component_final_labels.values() if label >= 0})
    for final_label in active_final_labels:
        final_idx = np.flatnonzero(new_state == final_label)
        if preselected_core_by_label.get(int(final_label), False):
            core_idx = final_idx
            shell_idx = np.zeros(0, dtype=np.int64)
            density_meta = {
                "dense_core_enabled": False,
                "dense_core_fraction": 1.0,
                "dense_core_frames": int(core_idx.size),
                "dense_core_shell_frames": 0,
                "dense_core_weight": float(np.sum(weights[core_idx])) if weights is not None else float(core_idx.size),
                "dense_core_shell_weight": 0.0,
                "dense_core_weight_fraction": 1.0,
                "dense_core_density_cutoff": np.nan,
                "dense_core_density_min": np.nan,
                "dense_core_density_median": np.nan,
                "dense_core_density_max": np.nan,
                "density_k_neighbors": 0,
                "density_radius_power": np.nan,
                "density_reason": "candidate components were already built from the dense core",
            }
        else:
            core_idx, shell_idx, density_meta = _select_weighted_density_core(
                z, final_idx, weights, relabel_cfg
            )
        if shell_idx.size:
            new_state[shell_idx] = -1
            density_shell_mask[shell_idx] = True
        if core_idx.size:
            new_core_mask[core_idx] = True
            new_state[core_idx] = final_label
        final_density[int(final_label)] = {
            **density_meta,
            "final_basin_candidate_frames": int(final_idx.size),
            "final_basin_candidate_weight": float(np.sum(weights[final_idx])) if weights is not None else float(final_idx.size),
            "mean_q_max": float(np.mean(q_max[core_idx])) if core_idx.size else np.nan,
            "mean_entropy_norm": float(np.mean(entropy_norm[core_idx])) if core_idx.size else np.nan,
        }

    for idx, row in enumerate(component_rows):
        provisional = int(row["provisional_new_state"])
        final_label = component_final_labels.get(provisional, provisional)
        row["new_state"] = int(final_label)
        component_frame_idx = component_frames[idx]
        row["component_dense_core_frames"] = int(np.sum(new_state[component_frame_idx] == final_label))
        row["component_density_shell_frames"] = int(np.sum(density_shell_mask[component_frame_idx]))
        row.update(final_density.get(int(final_label), {}))
    mark("density_core_selection")

    next_label = int(np.max(new_state[new_state >= 0]) + 1) if np.any(new_state >= 0) else int(next_label)
    (
        new_state,
        basin_kinetic_state_stats,
        reshaped_basin_groups,
        basin_kinetic_group_labels,
        reshaped_basin_core_mask,
        reshaped_basin_shell_mask,
        next_label,
    ) = _reshape_existing_basins_from_kinetic_groups(
        new_state,
        new_state.copy(),
        q_values,
        graph_features,
        trajectory_arr,
        frame_arr,
        weights,
        config,
        next_label,
    )
    mark("kinetics_reshape_existing_basins")

    original_label_set = set(int(label) for label in np.unique(state_labels[state_labels >= 0]))
    new_state, final_kinetic_merge_rows = _iteratively_merge_kinetically_duplicate_labels(
        new_state,
        original_label_set,
        trajectory_arr,
        frame_arr,
        weights,
        config,
    )
    for row in final_kinetic_merge_rows:
        row["phase"] = "before_final_split"
    mark("kinetics_final_merge_before_split")

    next_label = int(np.max(new_state[new_state >= 0]) + 1) if np.any(new_state >= 0) else int(next_label)
    (
        new_state,
        final_kinetic_split_rows,
        final_kinetic_split_mask,
        next_label,
    ) = _split_labels_by_final_knn_kinetics(
        new_state,
        z,
        original_label_set,
        trajectory_arr,
        frame_arr,
        weights,
        config,
        next_label,
    )
    mark("kinetics_final_knn_split")

    new_state, final_post_split_merge_rows = _iteratively_merge_kinetically_duplicate_labels(
        new_state,
        original_label_set,
        trajectory_arr,
        frame_arr,
        weights,
        config,
    )
    for row in final_post_split_merge_rows:
        row["phase"] = "after_final_split"
    final_kinetic_merge_rows.extend(final_post_split_merge_rows)
    mark("kinetics_final_merge_after_split")

    for idx, row in enumerate(component_rows):
        component_frame_idx = component_frames[idx]
        if len(component_frame_idx) == 0:
            row["final_checked_state"] = -1
            row["component_final_checked_frames"] = 0
            row["component_final_unlabeled_frames"] = 0
            continue
        assigned = new_state[component_frame_idx]
        nonnegative = assigned[assigned >= 0]
        if nonnegative.size:
            labels, counts = np.unique(nonnegative, return_counts=True)
            dominant = int(labels[int(np.argmax(counts))])
            dominant_frames = int(np.max(counts))
        else:
            dominant = -1
            dominant_frames = 0
        row["new_state"] = dominant
        row["final_checked_state"] = dominant
        row["component_final_checked_frames"] = dominant_frames
        row["component_final_unlabeled_frames"] = int(np.sum(assigned < 0))
    mark("proposal_component_final_annotation")

    changed_mask = new_state != state_labels
    review_mask = (
        transition_like_candidate_mask
        | unresolved_lagged_candidate_mask
        | persistent_candidate_review_mask
        | (missing_metastate_h_tau_mask & ~new_core_mask)
    )

    masks = {
        "changed": changed_mask,
        "removed": removed_mask,
        "current_entropy_candidate": current_entropy_candidate_mask,
        "ambiguous_to_unlabeled": ambiguous_to_unlabeled_mask,
        "missing_metastate_h_tau": missing_metastate_h_tau_mask,
        "transition_like_candidate": transition_like_candidate_mask,
        "unresolved_lagged_candidate": unresolved_lagged_candidate_mask,
        "missing_metastate_candidate": candidate_mask,
        "persistent_candidate_review": persistent_candidate_review_mask,
        "new_core": new_core_mask,
        "density_shell": density_shell_mask,
        "reshaped_basin_core": reshaped_basin_core_mask,
        "reshaped_basin_shell": reshaped_basin_shell_mask,
        "final_kinetic_split": final_kinetic_split_mask,
        "review": review_mask,
    }
    tables = {
        "removed_states": removed_states,
        "new_core_components": component_rows,
        "merged_new_states": merged_new_state_rows,
        "basin_kinetic_state_stats": basin_kinetic_state_stats,
        "reshaped_basin_groups": reshaped_basin_groups,
        "final_kinetic_merges": final_kinetic_merge_rows,
        "final_kinetic_splits": final_kinetic_split_rows,
    }
    diagnostics = _relabel_diagnostics(
        config,
        graph_idx,
        masks,
        tables,
        lagged_entropy_classification,
        relabel_cfg,
    )
    timings["proposal_total"] = _elapsed_seconds(proposal_start)
    diagnostics["timings"] = timings
    scores = {
        "q_max": q_max,
        "q_argmax": q_argmax,
        "entropy_norm": entropy_norm,
        "label_consistency": label_consistency,
        "mean_lagged_entropy_norm": mean_lagged_entropy,
        "lagged_high_entropy_fraction": lagged_high_entropy_fraction,
        "basin_kinetic_group_labels": basin_kinetic_group_labels,
    }
    return {
        "proposed_labels": new_state,
        "changed_mask": changed_mask,
        "masks": masks,
        "tables": tables,
        "diagnostics": diagnostics,
        "scores": scores,
    }

def _remove_stale_relabel_csv(output_dir):
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if name.startswith("relabel_") and name.endswith(".csv"):
            os.remove(os.path.join(output_dir, name))


def _label_population_rows(labels):
    labels = np.asarray(labels, dtype=np.int64)
    rows = []
    for label in sorted(int(x) for x in np.unique(labels)):
        rows.append({"label": label, "n_frames": int(np.sum(labels == label))})
    return rows


def run_relabel(dataset_path, model_path, config, device="cuda:0", batch_size=65536, dataset_stride=1):
    from ..common.data import apply_stride, infer_n_states, load_dataset, select_model_inputs
    from ..next_hit.predict import infer_probabilities, load_committor_model
    from .label_diagnostics import _compute_frame_index

    run_start = time.perf_counter()
    stage_start = run_start
    run_timings = {}

    def mark(stage):
        nonlocal stage_start
        run_timings[stage] = _elapsed_seconds(stage_start)
        stage_start = time.perf_counter()

    device_obj = setup_device(device)
    mark("setup_device")
    stride = int(dataset_stride)
    pack = apply_stride(load_dataset(dataset_path), stride)
    mark("load_dataset_apply_stride")
    n_states = infer_n_states(pack, config.get("n_states", None))
    model_features, _ = select_model_inputs(pack, config)
    mark("select_model_inputs")

    model = load_committor_model(model_path, device_obj)
    mark("load_model")
    q_values = infer_probabilities(model, model_features.float(), device_obj, batch_size=int(batch_size))
    if q_values.ndim != 2 or q_values.shape[1] != n_states:
        raise RuntimeError(f"Model returned q shape {q_values.shape}, expected (_, {n_states}).")
    mark("infer_q_values")

    state = pack.state.detach().cpu().numpy().astype(np.int64)
    weights = pack.weights.detach().cpu().numpy().astype(np.float64)
    traj_id = (
        pack.traj_id.detach().cpu().numpy().astype(np.int64)
        if pack.traj_id is not None
        else np.zeros(state.shape[0], dtype=np.int64)
    )
    frame_index = _compute_frame_index(traj_id)
    graph_features, graph_space = _select_graph_features(pack, model_features, config)
    mark("extract_arrays_and_graph_features")

    proposal = propose_relabeling(
        q_values,
        state,
        graph_features,
        weights,
        config,
        trajectory_index=traj_id,
        frame_index=frame_index,
    )
    mark("propose_relabeling")
    new_state = proposal["proposed_labels"]
    masks = proposal["masks"]
    tables = proposal["tables"]
    diagnostics = proposal["diagnostics"]

    output_dir = ensure_dir(config.get("output_dir", "relabel_out"))
    _remove_stale_relabel_csv(output_dir)
    mark("prepare_output_dir")

    relabel_cfg = _relabel_cfg(config)
    default_output = os.path.join(output_dir, f"relabeled_dataset{Path(str(dataset_path)).suffix or '.pt'}")
    output_dataset = relabel_cfg.get("output_dataset", default_output)
    saved_dataset = None
    if bool(relabel_cfg.get("write_relabel_dataset", True)):
        saved_dataset = _save_dataset_like_input(dataset_path, output_dataset, pack, new_state, config, stride)
    mark("save_relabeled_dataset")
    saved_plots = plot_relabel_diagnostics(
        pack,
        graph_features,
        state,
        proposal,
        config,
        output_dir,
    )
    mark("plot_relabel_diagnostics")

    run_timings["run_total_before_summary_write"] = _elapsed_seconds(run_start)
    timing_summary = {
        "run": run_timings,
        "proposal": diagnostics.get("timings", {}),
    }

    summary = {
        "dataset": os.path.abspath(str(dataset_path)),
        "model": os.path.abspath(str(model_path)),
        "output_dataset": None if saved_dataset is None else os.path.abspath(saved_dataset),
        "dataset_stride": stride,
        "graph_space": graph_space,
        "n_frames": int(state.size),
        "pipeline": diagnostics["pipeline"],
        "counts": diagnostics["counts"],
        "old_label_population": _label_population_rows(state),
        "new_label_population": _label_population_rows(new_state),
        "removed_states": tables["removed_states"],
        "new_core_components": tables["new_core_components"],
        "merged_new_states": tables["merged_new_states"],
        "basin_kinetic_state_stats": tables["basin_kinetic_state_stats"],
        "reshaped_basin_groups": tables["reshaped_basin_groups"],
        "final_kinetic_merges": tables["final_kinetic_merges"],
        "final_kinetic_splits": tables["final_kinetic_splits"],
        "lagged_entropy_classification": diagnostics["lagged_entropy_classification"],
        "knn_backend_config": diagnostics["knn_backend_config"],
        "timings": timing_summary,
        "plots": [os.path.abspath(path) for path in saved_plots],
        "notes": [
            "Relabeling now follows one path: entropy -> density -> kinetics.",
            "Entropy stage can remove a whole label when most of its frames are low-consistency or high-entropy.",
            "High current entropy marks unreliable labeled frames as unlabeled before retraining.",
            "Lagged entropy keeps transition-like frames in review and sends persistent uncertain frames to density-core detection.",
            "Density selection keeps only the weighted local-density core of each proposed new basin.",
            "Kinetic checks reshape existing labels and merge or split labels only when lagged transition evidence supports it.",
            "CSV frame dumps are intentionally not written; this YAML is the diagnostic record.",
        ],
    }
    summary_path = os.path.join(output_dir, "relabel_summary.yaml")
    write_yaml(summary, summary_path)
    mark("write_summary_yaml")
    run_timings["run_total"] = _elapsed_seconds(run_start)
    summary["timings"]["run"] = run_timings
    write_yaml(summary, summary_path)

    counts = diagnostics["counts"]
    print(f"[RELABEL] Removed states: {len(tables['removed_states'])}")
    print(f"[RELABEL] Changed frames: {counts['changed']}")
    print(f"[RELABEL] New core components: {len(tables['new_core_components'])}")
    print(f"[RELABEL] Reshaped basin groups: {len(tables['reshaped_basin_groups'])}")
    print(f"[RELABEL] Final kinetic merges: {len(tables['final_kinetic_merges'])}")
    print(f"[RELABEL] Final kinetic splits: {len(tables['final_kinetic_splits'])}")
    if saved_dataset is not None:
        print(f"[RELABEL] Saved dataset: {saved_dataset}")
    if saved_plots:
        print("[RELABEL] Saved plots:")
        for path in saved_plots:
            print(f"  {path}")
    profile_timing = bool(
        config.get("diagnostics", {}).get(
            "profile_timing",
            relabel_cfg.get("profile_timing", False),
        )
    )
    if profile_timing:
        print("[RELABEL] Timings (seconds):")
        for name, elapsed in run_timings.items():
            print(f"  run.{name}: {elapsed:.3f}")
        for name, elapsed in diagnostics.get("timings", {}).items():
            print(f"  proposal.{name}: {elapsed:.3f}")
    print(f"[RELABEL] Summary: {summary_path}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Apply confidence/kNN-graph relabeling.")
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
    run_relabel(
        dataset_path=dataset_path,
        model_path=model_path,
        config=cfg,
        device=str(cfg.get("device", "cuda:0")),
        batch_size=int(cfg.get("batch_size", 65536)),
        dataset_stride=int(cfg.get("dataset_stride", 1)),
    )


if __name__ == "__main__":
    main()

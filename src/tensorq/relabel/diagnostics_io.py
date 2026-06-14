from __future__ import annotations

import os

import numpy as np
import yaml


def _yaml_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_yaml_safe(item) for item in obj.tolist()]
    if isinstance(obj, dict):
        return {str(key): _yaml_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(item) for item in obj]
    return obj


def _remove_legacy_outputs(output_dir: str) -> None:
    legacy_names = [
        "diagnostic_summary.json",
        "state_confidence_summary.csv",
        "relabel_hints.csv",
        "basin_kinetic_state_summary.csv",
        "basin_kinetic_groups.csv",
        "split_candidates.csv",
        "split_candidates_details.json",
        "merge_candidates.csv",
        "missing_state_candidates.csv",
        "cluster_statistics.csv",
        "label_consistency_by_frame.csv",
        "label_consistency_by_frame.npz",
    ]
    for name in legacy_names:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def save_results(
    results: dict,
    output_dir: str,
    per_frame: np.ndarray | dict | None = None,
    q_values: np.ndarray | None = None,
) -> None:
    """Write the diagnostic result as one YAML file.

    ``per_frame`` and ``q_values`` are accepted for API compatibility but are
    intentionally not serialized. The relabel package now keeps diagnostics in
    a compact summary instead of producing many CSV/NPZ side files.
    """
    os.makedirs(output_dir, exist_ok=True)
    _remove_legacy_outputs(output_dir)

    summary = dict(results.get("summary", {}))
    summary["n_split_candidates"] = len(results.get("split_candidates", []))
    summary["n_merge_candidates"] = len(results.get("merge_candidates", []))
    summary["n_missing_state_candidates"] = len(results.get("missing_state_candidates", []))
    summary["split_candidates"] = results.get("split_candidates", [])
    summary["merge_candidates"] = results.get("merge_candidates", [])
    summary["missing_state_candidates"] = results.get("missing_state_candidates", [])
    summary["relabel_hints"] = results.get("relabel_hints", [])
    summary["state_confidence"] = results.get("state_confidence", [])
    summary["basin_kinetic_state_stats"] = results.get("basin_kinetic_state_stats", [])
    summary["basin_kinetic_groups"] = results.get("basin_kinetic_groups", [])

    path = os.path.join(output_dir, "diagnostic_summary.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(_yaml_safe(summary), fh, sort_keys=False)

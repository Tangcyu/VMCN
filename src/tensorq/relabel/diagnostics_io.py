from __future__ import annotations

import csv
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ensure_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _ensure_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_ensure_json_serializable(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Internal writers
# ---------------------------------------------------------------------------

def _write_csv(filepath: str, rows: list[dict]) -> None:
    """Write a list of dicts to a CSV file.  Handles empty lists gracefully."""
    if not rows:
        # Write a file with only the header row so downstream readers do not
        # fail on a missing file.
        with open(filepath, "w", newline="") as fh:
            pass
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(filepath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _remove_if_exists(*paths: str) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _per_frame_to_rows(
    per_frame: np.ndarray | dict,
) -> list[dict]:
    """Convert *per_frame* into a list of dicts suitable for CSV writing.

    Supports both structured numpy arrays and dicts of 1-D arrays.
    """
    if isinstance(per_frame, np.ndarray):
        # Structured array – each dtype field becomes a column.
        rows: list[dict] = []
        for record in per_frame:
            row = {}
            for name in record.dtype.names or ():
                val = record[name]
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                elif isinstance(val, np.bytes_):
                    val = val.decode("utf-8")
                row[name] = val
            rows.append(row)
        return rows

    if isinstance(per_frame, dict):
        # Dict of 1-D arrays – zip them together.
        keys = list(per_frame.keys())
        if not keys:
            return []
        n = min(len(v) for v in per_frame.values() if hasattr(v, "__len__"))
        rows = []
        for i in range(n):
            row = {}
            for k in keys:
                arr = per_frame[k]
                val = arr[i] if hasattr(arr, "__getitem__") else arr
                if isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                elif isinstance(val, np.bytes_):
                    val = val.decode("utf-8")
                row[k] = val
            rows.append(row)
        return rows

    # Unknown type – return as-is if it is already a list of dicts.
    if isinstance(per_frame, list) and all(isinstance(r, dict) for r in per_frame):
        return per_frame

    return []


def _write_per_frame_csv(filepath: str, per_frame: np.ndarray | dict) -> None:
    try:
        import pandas as pd

        if isinstance(per_frame, dict):
            pd.DataFrame(per_frame).to_csv(filepath, index=False)
            return
        if isinstance(per_frame, np.ndarray):
            pd.DataFrame.from_records(per_frame).to_csv(filepath, index=False)
            return
    except Exception:
        pass

    _write_csv(filepath, _per_frame_to_rows(per_frame))


def _write_per_frame_npz(filepath: str, per_frame: np.ndarray | dict) -> None:
    if isinstance(per_frame, dict):
        np.savez(filepath, **{key: np.asarray(value) for key, value in per_frame.items()})
        return
    if isinstance(per_frame, np.ndarray) and per_frame.dtype.names:
        np.savez(filepath, **{name: per_frame[name] for name in per_frame.dtype.names})
        return
    np.savez(filepath, per_frame=np.asarray(per_frame))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_results(
    results: dict,
    output_dir: str,
    per_frame: np.ndarray | dict | None = None,
    q_values: np.ndarray | None = None,
) -> None:
    """Write all diagnostic outputs to CSV and JSON files in *output_dir*.

    Parameters
    ----------
    results : dict
        Dictionary with keys documented in the module docstring.
    output_dir : str
        Directory into which output files are written (created if needed).
    per_frame : np.ndarray or dict or None
        Structured array or dict of per-frame diagnostic data.
    q_values : np.ndarray or None
        Shape ``(n_frames, n_states)`` array of raw Q-values (not written
        in this version but accepted for API compatibility).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. diagnostic_summary.json ---------------------------------------
    summary = dict(results.get("summary", {}))
    summary["n_split_candidates"] = len(results.get("split_candidates", []))
    summary["n_merge_candidates"] = len(results.get("merge_candidates", []))
    summary["n_missing_state_candidates"] = len(
        results.get("missing_state_candidates", [])
    )
    summary["split_candidates"] = results.get("split_candidates", [])
    summary["merge_candidates"] = results.get("merge_candidates", [])
    summary["missing_state_candidates"] = results.get("missing_state_candidates", [])
    summary["relabel_hints"] = results.get("relabel_hints", [])
    summary["basin_kinetic_state_stats"] = results.get("basin_kinetic_state_stats", [])
    summary["basin_kinetic_groups"] = results.get("basin_kinetic_groups", [])

    with open(os.path.join(output_dir, "diagnostic_summary.json"), "w") as fh:
        json.dump(_ensure_json_serializable(summary), fh, indent=2)

    # ---- 2. state_confidence_summary.csv ----------------------------------
    _write_csv(
        os.path.join(output_dir, "state_confidence_summary.csv"),
        results.get("state_confidence", []),
    )

    # ---- 2b. relabel_hints.csv -------------------------------------------
    _write_csv(
        os.path.join(output_dir, "relabel_hints.csv"),
        results.get("relabel_hints", []),
    )

    _write_csv(
        os.path.join(output_dir, "basin_kinetic_state_summary.csv"),
        results.get("basin_kinetic_state_stats", []),
    )
    _write_csv(
        os.path.join(output_dir, "basin_kinetic_groups.csv"),
        results.get("basin_kinetic_groups", []),
    )

    # These files were produced by the old clustering-based candidate detector.
    # The current workflow keeps those decisions as manual follow-up, so remove
    # stale files instead of writing misleading empty artifacts.
    _remove_if_exists(
        os.path.join(output_dir, "split_candidates.csv"),
        os.path.join(output_dir, "split_candidates_details.json"),
        os.path.join(output_dir, "merge_candidates.csv"),
        os.path.join(output_dir, "missing_state_candidates.csv"),
        os.path.join(output_dir, "cluster_statistics.csv"),
    )

    # ---- 7. label_consistency_by_frame.csv ---------------------------------
    if per_frame is not None:
        config = results.get("summary", {}).get("config", {})
        output_cfg = config.get("output", {})
        save_csv = bool(output_cfg.get("save_per_frame_csv", True))
        save_npz = bool(output_cfg.get("save_per_frame_npz", False))

        if save_csv:
            _write_per_frame_csv(
                os.path.join(output_dir, "label_consistency_by_frame.csv"),
                per_frame,
            )
        else:
            csv_path = os.path.join(output_dir, "label_consistency_by_frame.csv")
            if os.path.exists(csv_path):
                os.remove(csv_path)
        if save_npz:
            _write_per_frame_npz(
                os.path.join(output_dir, "label_consistency_by_frame.npz"),
                per_frame,
            )
        else:
            npz_path = os.path.join(output_dir, "label_consistency_by_frame.npz")
            if os.path.exists(npz_path):
                os.remove(npz_path)

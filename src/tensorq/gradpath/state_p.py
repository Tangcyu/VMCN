from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import ensure_dir, write_yaml
from .runner import run_gradpath


def parse_state_endpoints(config: dict[str, Any]) -> dict[int, np.ndarray]:
    """Build a ``{label: center}`` mapping from a *state_endpoints* config section.

    The config format mirrors ``label.py``'s ``user_defined_basins``::

        state_endpoints:
          cvs_to_label: [cv_0, cv_1, ...]   # informational only
          basins:
            - label: 0
              center: [-70.0, -70.0, -70.0]
            - label: 1
              center: [ 60.0, -70.0, -70.0]

    Returns a dict keyed by integer label whose values are 1-D numpy arrays
    suitable for the ``endpoint_i``/``endpoint_j`` config keys consumed by
    ``run_gradpath``.
    """
    spec = config.get("state_endpoints", None)
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise ValueError("state_endpoints must be a mapping.")
    basins = spec.get("basins", [])
    if not isinstance(basins, list) or not basins:
        raise ValueError("state_endpoints.basins must be a non-empty list.")
    mapping: dict[int, np.ndarray] = {}
    for idx, basin in enumerate(basins):
        if not isinstance(basin, dict):
            raise ValueError(f"Basin #{idx} in state_endpoints must be a dict.")
        label = int(basin.get("label", idx))
        center = np.asarray(basin["center"], dtype=np.float64)
        if center.ndim != 1:
            raise ValueError(f"Basin #{idx} center must be a 1-D list.")
        mapping[label] = center
    return mapping


def load_p_jump(path: str) -> np.ndarray:
    """Load P_jump matrix from CSV or NPY file."""
    if str(path).endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
        return df.values.astype(np.float64)
    return np.asarray(np.load(path), dtype=np.float64)


def find_transitions_above_threshold(
    P_jump: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return (i, j, prob) sorted by descending probability for undirected pairs above threshold.

    i -> j and j -> i are treated as the same channel; only the direction with
    the larger P_jump is kept.  Each unordered pair appears at most once.
    """
    n_states = P_jump.shape[0]
    seen: set[tuple[int, int]] = set()
    transitions: list[tuple[int, int, float]] = []
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            if (min(i, j), max(i, j)) in seen:
                continue
            seen.add((min(i, j), max(i, j)))
            p_ij = float(P_jump[i, j])
            p_ji = float(P_jump[j, i])
            best = max(p_ij, p_ji)
            if best > threshold:
                if p_ij >= p_ji:
                    transitions.append((i, j, best))
                else:
                    transitions.append((j, i, best))
    transitions.sort(key=lambda x: x[2], reverse=True)
    return transitions


def run_gradpath_for_state_pairs(config: dict[str, Any]) -> dict[str, Any]:
    """Run gradpath for every state pair whose P_jump exceeds the configured threshold.

    Reads the following keys from *config* (in addition to the usual GRADPATH keys):

    - ``p_jump`` / ``p_jump_path``: path to P_jump.csv / P_jump.npy from ``rate_constant.py``
    - ``prob_threshold`` / ``p_jump_threshold``: minimum P_jump[i,j] (default 0.01)
    - ``max_pairs``: limit to top-N by probability (default: no limit)

    For each qualifying pair ``(i, j)``, ``run_gradpath`` is called with
    ``out_dir/state_i_j/`` so each transition gets its own subdirectory.
    Identical pair ``(i, j)`` is only ever processed once.
    """
    p_jump_path = config.get("p_jump", config.get("p_jump_path", None))
    if p_jump_path is None:
        raise KeyError("GRADPATH config must contain 'p_jump' or 'p_jump_path' pointing to P_jump.csv / P_jump.npy.")
    prob_threshold = float(config.get("prob_threshold", config.get("p_jump_threshold", 0.01)))
    max_pairs = config.get("max_pairs", None)

    P_jump = load_p_jump(str(p_jump_path))
    transitions = find_transitions_above_threshold(P_jump, prob_threshold)

    if max_pairs is not None:
        transitions = transitions[: int(max_pairs)]

    endpoints = parse_state_endpoints(config)
    if endpoints:
        print(f"[STATE_P] Loaded {len(endpoints)} state endpoint basin(s): {sorted(endpoints.keys())}")

    out_root = ensure_dir(config.get("out_dir", "./gradpath_state_p"))
    results: list[dict[str, Any]] = []

    for i, j, prob in transitions:
        pair_config = dict(config)
        pair_config["state_i"] = i
        pair_config["state_j"] = j
        pair_config["out_dir"] = out_root
        pair_config["channel_name"] = f"state_{i}_{j}"
        pair_config["use_channel_subdir"] = True
        pair_config.setdefault("selection_mode", "fel_kde")

        if endpoints:
            center_i = endpoints.get(i)
            center_j = endpoints.get(j)
            if center_i is None:
                print(f"[STATE_P] WARNING: state {i} has no endpoint center; skipping pair ({i} -> {j})")
                continue
            if center_j is None:
                print(f"[STATE_P] WARNING: state {j} has no endpoint center; skipping pair ({i} -> {j})")
                continue
            pair_config["endpoint_i"] = center_i.tolist()
            pair_config["endpoint_j"] = center_j.tolist()
            print(f"[STATE_P]   endpoint_i = {center_i.tolist()}")
            print(f"[STATE_P]   endpoint_j = {center_j.tolist()}")

        print(f"\n[STATE_P] state pair ({i} -> {j})  P_jump = {prob:.4f}")
        try:
            result = run_gradpath(pair_config)
            result["p_jump_probability"] = float(prob)
            results.append(result)
        except Exception as exc:
            print(f"[STATE_P] ERROR ({i} -> {j}): {exc}")
            results.append(
                {
                    "state_i": int(i),
                    "state_j": int(j),
                    "p_jump_probability": float(prob),
                    "error": str(exc),
                }
            )

    summary: dict[str, Any] = {
        "out_root": os.path.abspath(out_root),
        "p_jump_path": os.path.abspath(str(p_jump_path)),
        "prob_threshold": float(prob_threshold),
        "max_pairs": int(max_pairs) if max_pairs is not None else None,
        "n_pairs_above_threshold": len(transitions),
        "n_pairs_processed": len(results),
        "results": results,
    }
    write_yaml(summary, os.path.join(out_root, "state_p_summary.yaml"))
    print(f"\n[STATE_P] done — {len(results)} pair(s) processed, outputs in {out_root}")
    return summary

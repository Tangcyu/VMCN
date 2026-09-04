"""Macrostate merging from next-hit jump probabilities and MFPTs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _matrix_path(config: Mapping[str, Any], *keys: str) -> Path:
    for key in keys:
        value = config.get(key)
        if value is not None:
            path = Path(str(value)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Rate-merge input {key} is missing: {path}. Run the rate "
                    "estimation step (vmcn_rates in Gen-COMPAS) first."
                )
            return path
    raise KeyError(f"Rate merging requires one of: {', '.join(keys)}")


def merge_macrostates_from_rates(
    labels: np.ndarray,
    config: Mapping[str, Any],
    n_model_states: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, int]]]:
    """Merge connected labels when P and MFPT jointly qualify by direction."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    relabel = config.get("relabel", config)
    if not bool(relabel.get("rate_merge_enabled", False)):
        return labels, [], [], []

    probability_cutoff = float(relabel.get("rate_merge_probability_cutoff", 0.95))
    if not 0.0 <= probability_cutoff <= 1.0:
        raise ValueError("relabel.rate_merge_probability_cutoff must lie in [0, 1].")
    mfpt_cutoff = float(
        relabel.get(
            "rate_merge_mfpt_cutoff",
            relabel.get("rate_merge_mfpt_cutoff_frames", 100.0),
        )
    )
    if not np.isfinite(mfpt_cutoff) or mfpt_cutoff <= 0.0:
        raise ValueError("The rate-merge MFPT cutoff must be finite and positive.")

    probability_path = _matrix_path(
        relabel,
        "rate_merge_probability_path",
        "rate_merge_jump_probability_path",
    )
    mfpt_path = _matrix_path(relabel, "rate_merge_mfpt_path")
    probability = np.asarray(np.load(probability_path), dtype=np.float64)
    mfpt = np.asarray(np.load(mfpt_path), dtype=np.float64)
    expected = (int(n_model_states), int(n_model_states))
    if probability.shape != expected:
        raise ValueError(f"P_jump shape {probability.shape} does not match {expected}.")
    if mfpt.shape != expected:
        raise ValueError(f"MFPT shape {mfpt.shape} does not match {expected}.")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"P_jump contains non-finite values: {probability_path}")
    if np.any(np.isnan(mfpt)) or np.any(mfpt < 0.0):
        raise ValueError(f"MFPT contains NaN or negative values: {mfpt_path}")

    require_bidirectional = bool(relabel.get("rate_merge_require_bidirectional", False))
    parent = np.arange(int(n_model_states), dtype=np.int64)

    def find(state: int) -> int:
        while parent[state] != state:
            parent[state] = parent[parent[state]]
            state = int(parent[state])
        return int(state)

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            keep, remove = sorted((root_i, root_j))
            parent[remove] = keep

    edges: list[dict[str, Any]] = []
    for i in range(int(n_model_states)):
        for j in range(i + 1, int(n_model_states)):
            p_ij, p_ji = float(probability[i, j]), float(probability[j, i])
            t_ij, t_ji = float(mfpt[i, j]), float(mfpt[j, i])
            qualifies_ij = p_ij > probability_cutoff and t_ij < mfpt_cutoff
            qualifies_ji = p_ji > probability_cutoff and t_ji < mfpt_cutoff
            qualifies = (
                qualifies_ij and qualifies_ji
                if require_bidirectional
                else qualifies_ij or qualifies_ji
            )
            if qualifies:
                union(i, j)
                edges.append(
                    {
                        "state_i": i,
                        "state_j": j,
                        "P_ij": p_ij,
                        "P_ji": p_ji,
                        "MFPT_ij": t_ij,
                        "MFPT_ji": t_ji,
                        "MFPT_ij_frames": t_ij,
                        "MFPT_ji_frames": t_ji,
                        "qualifies_ij": bool(qualifies_ij),
                        "qualifies_ji": bool(qualifies_ji),
                    }
                )

    connected: dict[int, list[int]] = {}
    for state in range(int(n_model_states)):
        connected.setdefault(find(state), []).append(state)
    merged = labels.copy()
    groups: list[dict[str, Any]] = []
    for members in connected.values():
        if len(members) < 2:
            continue
        representative = min(members)
        for state in members:
            merged[labels == state] = representative
        groups.append(
            {
                "states": [int(x) for x in members],
                "representative_before_compaction": int(representative),
                "n_labeled_frames": int(np.sum(np.isin(labels, members))),
                "probability_cutoff": probability_cutoff,
                "mfpt_cutoff": mfpt_cutoff,
                "mfpt_cutoff_frames": mfpt_cutoff,
                "require_bidirectional": require_bidirectional,
            }
        )

    present = sorted(int(x) for x in np.unique(merged[merged >= 0]))
    mapping = {old: new for new, old in enumerate(present)}
    compact = merged.copy()
    for old, new in mapping.items():
        compact[merged == old] = new
    for row in groups:
        row["representative"] = int(mapping[row["representative_before_compaction"]])
    mapping_rows = [
        {"label_before_compaction": old, "label": new}
        for old, new in mapping.items()
    ]
    return compact, groups, edges, mapping_rows

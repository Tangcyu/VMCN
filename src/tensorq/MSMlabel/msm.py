from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eig

from .checkpoint import exists, load_npz, save_npz
from .config import ensure_dir, stage_path


def trajectory_label_weight_arrays(
    table: pd.DataFrame,
    labels: np.ndarray,
    weight_column: str = "weight",
    use_weights: bool = True,
    mask_zero_weight_origins: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if weight_column in table.columns:
        raw_weights = pd.to_numeric(table[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    else:
        raw_weights = np.ones(len(table), dtype=np.float64)

    if use_weights:
        origin_factor = raw_weights.copy()
    else:
        origin_factor = np.ones(len(table), dtype=np.float64)
    if mask_zero_weight_origins:
        origin_factor = origin_factor * (raw_weights > 0.0)

    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape[0] != len(table):
        raise SystemExit(f"microstate label count ({labels.shape[0]}) != frame table rows ({len(table)}).")

    traj_codes, _ = pd.factorize(table["traj_id"], sort=False)
    frame = pd.to_numeric(table["frame_in_traj"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    order = np.lexsort((frame, traj_codes.astype(np.int64)))
    sorted_traj = traj_codes[order]
    sorted_labels = labels[order]
    sorted_weights = origin_factor[order]
    split = np.flatnonzero(np.diff(sorted_traj) != 0) + 1
    starts = np.r_[0, split]
    stops = np.r_[split, len(sorted_labels)]

    label_arrays: List[np.ndarray] = []
    weight_arrays: List[np.ndarray] = []
    for start, stop in zip(starts, stops):
        if stop - start > 1:
            label_arrays.append(sorted_labels[start:stop])
            weight_arrays.append(sorted_weights[start:stop])
    return label_arrays, weight_arrays


def trajectory_label_arrays(table: pd.DataFrame, labels: np.ndarray) -> List[np.ndarray]:
    seqs, _ = trajectory_label_weight_arrays(table, labels, use_weights=False, mask_zero_weight_origins=False)
    return seqs


def count_matrix(
    seqs: Iterable[np.ndarray],
    lag: int,
    n_states: int,
    weight_seqs: Iterable[np.ndarray] | None = None,
) -> np.ndarray:
    C = np.zeros((n_states, n_states), dtype=np.float64)
    if weight_seqs is None:
        iterator = ((seq, None) for seq in seqs)
    else:
        iterator = zip(seqs, weight_seqs)
    for seq, weights in iterator:
        if len(seq) <= lag:
            continue
        origins = np.asarray(seq[:-lag], dtype=np.int64)
        targets = np.asarray(seq[lag:], dtype=np.int64)
        origin_weights = np.ones(len(origins), dtype=np.float64) if weights is None else np.asarray(weights[:-lag], dtype=np.float64)
        valid = (
            (origins >= 0)
            & (origins < n_states)
            & (targets >= 0)
            & (targets < n_states)
            & (origin_weights > 0.0)
        )
        if np.any(valid):
            flat = origins[valid] * int(n_states) + targets[valid]
            C += np.bincount(flat, weights=origin_weights[valid], minlength=n_states * n_states).reshape(n_states, n_states)
    return C


def count_matrices(
    seqs: Iterable[np.ndarray],
    lags: Iterable[int],
    n_states: int,
    weight_seqs: Iterable[np.ndarray] | None = None,
) -> Dict[int, np.ndarray]:
    lag_list = sorted({int(lag) for lag in lags})
    counts = {lag: np.zeros((n_states, n_states), dtype=np.float64) for lag in lag_list}
    iterator = ((seq, None) for seq in seqs) if weight_seqs is None else zip(seqs, weight_seqs)
    for seq, weights in iterator:
        seq = np.asarray(seq, dtype=np.int64)
        weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
        for lag in lag_list:
            if len(seq) <= lag:
                continue
            origins = seq[:-lag]
            targets = seq[lag:]
            origin_weights = np.ones(len(origins), dtype=np.float64) if weights_arr is None else weights_arr[:-lag]
            valid = (
                (origins >= 0)
                & (origins < n_states)
                & (targets >= 0)
                & (targets < n_states)
                & (origin_weights > 0.0)
            )
            if np.any(valid):
                flat = origins[valid] * int(n_states) + targets[valid]
                counts[lag] += np.bincount(
                    flat,
                    weights=origin_weights[valid],
                    minlength=n_states * n_states,
                ).reshape(n_states, n_states)
    return counts


def transition_matrix(C: np.ndarray, reversible: bool = True, pseudocount: float = 1e-8, inactive: str = "self") -> np.ndarray:
    C2 = C.astype(np.float64)
    if reversible:
        C2 = 0.5 * (C2 + C2.T)
    inactive_rows = C2.sum(axis=1) <= 0.0
    C2 = C2 + float(pseudocount)
    if inactive == "self" and np.any(inactive_rows):
        C2[inactive_rows, :] = 0.0
        C2[inactive_rows, np.where(inactive_rows)[0]] = 1.0
    row_sum = C2.sum(axis=1, keepdims=True)
    T = C2 / np.maximum(row_sum, 1e-300)
    return T


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    vals, vecs = eig(T.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi)
    return pi / max(float(pi.sum()), 1e-300)


def implied_timescales(T: np.ndarray, lag: int, n_timescales: int) -> np.ndarray:
    vals = np.linalg.eigvals(T)
    vals = np.sort(np.abs(vals))[::-1]
    vals = vals[vals < 1.0 - 1e-10]
    vals = vals[:n_timescales]
    vals = np.clip(vals, 1e-15, 1.0 - 1e-12)
    return -float(lag) / np.log(vals)


def active_origin_microstates(C: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(C, dtype=np.float64).sum(axis=1) > 0.0)[0].astype(np.int64)


def active_transition_matrix_for_spectrum(
    C: np.ndarray,
    active: np.ndarray,
    pseudocount: float,
    reversible: bool,
) -> np.ndarray:
    C_active = np.asarray(C, dtype=np.float64)[np.ix_(active, active)]
    return transition_matrix(C_active, reversible=reversible, pseudocount=pseudocount, inactive="uniform")


def expand_stationary_distribution(pi_active: np.ndarray, active: np.ndarray, n_states: int) -> np.ndarray:
    pi = np.zeros(n_states, dtype=np.float64)
    pi[active] = pi_active
    return pi / max(float(pi.sum()), 1e-300)


def build_msms(cfg: Dict, table: pd.DataFrame, micro: Dict[str, np.ndarray]) -> Dict[int, Dict[str, np.ndarray]]:
    force = bool(cfg["project"].get("force", False))
    labels = np.asarray(micro["labels"], dtype=np.int64)
    n_states = int(np.max(labels)) + 1
    use_weights = bool(cfg["msm"].get("use_weights", True))
    mask_zero_weight_origins = bool(cfg["msm"].get("mask_zero_weight_origins", True))
    weight_column = cfg["data"].get("weight_column", "weight")
    seqs, weight_seqs = trajectory_label_weight_arrays(
        table,
        labels,
        weight_column=weight_column,
        use_weights=use_weights,
        mask_zero_weight_origins=mask_zero_weight_origins,
    )
    out: Dict[int, Dict[str, np.ndarray]] = {}
    ensure_dir(stage_path(cfg, "03_msms"))
    requested_reversible = bool(cfg["msm"].get("reversible", True))
    has_zero_weight_frames = bool(
        weight_column in table.columns
        and np.any(pd.to_numeric(table[weight_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) <= 0.0)
    )
    effective_reversible = requested_reversible
    if mask_zero_weight_origins and has_zero_weight_frames and requested_reversible:
        effective_reversible = False
        print("[warn] msm.reversible=true is disabled because msm.mask_zero_weight_origins=true and zero-weight frames exist.")

    requested_lags = [int(x) for x in cfg["msm"]["lags"]]
    missing_lags: List[int] = []
    for lag in requested_lags:
        out_npz = stage_path(cfg, "03_msms", f"lag_{lag}", "msm.npz")
        if exists(out_npz, force=force):
            print(f"[reuse] MSM lag={lag}: {out_npz}")
            out[lag] = load_npz(out_npz)
        else:
            missing_lags.append(lag)

    counted = count_matrices(seqs, missing_lags, n_states=n_states, weight_seqs=weight_seqs) if missing_lags else {}
    for lag in missing_lags:
        out_npz = stage_path(cfg, "03_msms", f"lag_{lag}", "msm.npz")
        C = counted[lag]
        active = active_origin_microstates(C)
        if active.size <= 1:
            raise SystemExit(f"MSM lag={lag} has only {active.size} active origin microstates; cannot compute implied timescales.")
        T = transition_matrix(
            C,
            reversible=effective_reversible,
            pseudocount=float(cfg["msm"].get("pseudocount", 1e-8)),
        )
        T_active = active_transition_matrix_for_spectrum(
            C,
            active=active,
            reversible=effective_reversible,
            pseudocount=float(cfg["msm"].get("pseudocount", 1e-8)),
        )
        pi_active = stationary_distribution(T_active)
        pi = expand_stationary_distribution(pi_active, active, n_states)
        its = implied_timescales(T_active, lag=lag, n_timescales=int(cfg["msm"].get("n_timescales", 10)))
        result = {
            "count_matrix": C,
            "transition_matrix": T,
            "active_transition_matrix": T_active,
            "active_microstates": active,
            "stationary_distribution": pi,
            "active_stationary_distribution": pi_active,
            "implied_timescales": its,
            "lag": np.asarray(lag),
        }
        save_npz(
            out_npz,
            manifest={
                "stage": "msm",
                "lag": lag,
                "n_states": n_states,
                "n_active_origin_microstates": int(active.size),
                "n_trajectories": len(seqs),
                "use_weights": use_weights,
                "mask_zero_weight_origins": mask_zero_weight_origins,
                "weight_column": weight_column if use_weights else None,
                "requested_reversible": requested_reversible,
                "effective_reversible": effective_reversible,
            },
            **result,
        )
        print(f"[ok] MSM lag={lag}: {out_npz}")
        out[lag] = result
    return out

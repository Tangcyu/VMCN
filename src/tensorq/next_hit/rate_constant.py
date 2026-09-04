from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, torch_load, write_yaml
from ..common.data import apply_stride, build_lagged_indices, infer_n_states, load_dataset, select_model_inputs
from ..common.flux import make_thresholds, resolve_ordered_pairs
from .predict import check_probability_rows, infer_probabilities, load_committor_model


def trajectory_burn_in_mask(
    traj_id: torch.Tensor | None,
    n_frames: int,
    discard_first_n_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask the first N saved frames of every consecutive trajectory block."""

    n_frames = int(n_frames)
    discard = int(discard_first_n_frames)
    if n_frames < 0:
        raise ValueError("n_frames must be nonnegative.")
    if discard < 0:
        raise ValueError("discard_first_n_frames must be nonnegative.")

    if traj_id is None:
        ids = np.zeros(n_frames, dtype=np.int64)
        has_trajectory_ids = False
    else:
        ids = traj_id.detach().cpu().numpy().reshape(-1)
        if ids.shape[0] != n_frames:
            raise ValueError(f"traj_id has {ids.shape[0]} frames but the dataset has {n_frames}.")
        has_trajectory_ids = True

    keep = np.ones(n_frames, dtype=bool)
    if n_frames:
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.r_[0, boundaries].astype(np.int64)
        stops = np.r_[boundaries, n_frames].astype(np.int64)
    else:
        starts = np.asarray([], dtype=np.int64)
        stops = np.asarray([], dtype=np.int64)

    per_trajectory: list[dict[str, Any]] = []
    for block_index, (start, stop) in enumerate(zip(starts, stops)):
        block_size = int(stop - start)
        n_discard = min(discard, block_size)
        keep[start : start + n_discard] = False
        label = ids[start]
        if isinstance(label, np.generic):
            label = label.item()
        per_trajectory.append(
            {
                "block_index": int(block_index),
                "trajectory_id": label,
                "n_frames": block_size,
                "n_discarded": int(n_discard),
                "n_retained": int(block_size - n_discard),
            }
        )

    n_discarded = int(np.count_nonzero(~keep))
    return keep, {
        "discard_first_n_frames": int(discard),
        "reference": "original_saved_dataset_frames",
        "has_trajectory_ids": bool(has_trajectory_ids),
        "n_trajectory_blocks": int(len(per_trajectory)),
        "n_frames_before_discard": int(n_frames),
        "n_frames_discarded": n_discarded,
        "n_frames_retained": int(n_frames - n_discarded),
        "per_trajectory": per_trajectory,
    }


def resolve_lag_timing(config: dict[str, Any], dataset_stride: int) -> dict[str, Any]:
    """
    Resolve configured lag to the index step used after dataset_stride.

    By default, lag is interpreted in original saved dataset frames. After
    apply_stride(), the lagged-index step is lag / dataset_stride. Set
    lag_reference: "strided_frames" to preserve the old post-stride meaning.
    """
    dataset_stride = int(dataset_stride)
    if dataset_stride < 1:
        raise ValueError("dataset_stride must be >= 1.")
    lag = int(config.get("lag", config.get("time_shift", 1)))
    if lag < 1:
        raise ValueError("lag must be >= 1.")

    reference = str(config.get("lag_reference", config.get("lag_unit", "original_frames"))).lower()
    original_aliases = {"original", "original_frame", "original_frames", "saved", "saved_frame", "saved_frames"}
    strided_aliases = {
        "strided",
        "strided_frame",
        "strided_frames",
        "post_stride",
        "post_stride_frame",
        "post_stride_frames",
    }
    if reference in original_aliases:
        if lag % dataset_stride != 0:
            raise ValueError(
                "RATE_CONSTANT lag is interpreted in original saved frames by default, "
                f"but lag={lag} is not divisible by dataset_stride={dataset_stride}. "
                "Choose a divisible lag, reduce dataset_stride, or set "
                "lag_reference: 'strided_frames' to interpret lag after striding."
            )
        lag_index_step = lag // dataset_stride
        lag_original_frames = lag
        lag_reference = "original_frames"
    elif reference in strided_aliases:
        lag_index_step = lag
        lag_original_frames = lag * dataset_stride
        lag_reference = "strided_frames"
    else:
        raise ValueError("lag_reference must be 'original_frames' or 'strided_frames'.")

    if lag_index_step < 1:
        raise ValueError(
            f"lag={lag} original frame(s) is smaller than dataset_stride={dataset_stride}; "
            "no lagged pair can represent that physical separation after striding."
        )

    frame_time = config.get("frame_time", None)
    tau = float(lag_original_frames) if frame_time is None else float(lag_original_frames) * float(frame_time)
    time_unit = str(config.get("time_unit", "frame" if frame_time is None else "time"))
    return {
        "lag": int(lag),
        "lag_reference": lag_reference,
        "lag_index_step": int(lag_index_step),
        "lag_original_frames": int(lag_original_frames),
        "dataset_stride": int(dataset_stride),
        "frame_time": None if frame_time is None else float(frame_time),
        "tau": float(tau),
        "time_unit": time_unit,
    }


def read_checkpoint_model_input(path: str | os.PathLike[str]) -> dict[str, Any]:
    """
    Return model-input metadata saved by train.py checkpoints.

    TorchScript models do not expose this metadata, so unreadable paths simply
    return an empty dict and the caller decides whether explicit config is
    required.
    """
    try:
        checkpoint = torch_load(path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(checkpoint, dict):
        return {}
    raw = checkpoint.get("model_input", {})
    return dict(raw) if isinstance(raw, dict) else {}


def model_input_space_configured(config: dict[str, Any]) -> bool:
    return any(config.get(key, None) is not None for key in ("model_input_space", "input_space", "feature_space"))


def resolve_flux_dtype(value: Any, device: torch.device) -> torch.dtype:
    if value is None:
        return torch.float32 if device.type == "cuda" else torch.float64
    text = str(value).lower()
    if text in {"float32", "fp32", "single"}:
        return torch.float32
    if text in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError("flux_dtype must be 'float32' or 'float64'.")


def apply_checkpoint_model_input_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fill missing model-input options from checkpoint metadata.

    The train step stores select_model_inputs() metadata as "model_input".
    This maps those saved keys back to the config keys consumed by
    select_model_inputs(), so rate estimation uses the same CV/feature space as
    training.
    """
    effective = dict(config)
    if config.get("Q_npy", config.get("q_npy", None)) is not None:
        return effective, {"model_input_source": "not_used_Q_npy"}

    model_path = config.get("model", None)
    if model_path is None:
        return effective, {"model_input_source": "not_used_no_model"}

    meta = read_checkpoint_model_input(model_path)
    mapping = {
        "model_input_space": meta.get("model_input_space", None),
        "cvs_to_use": meta.get("model_cvs_to_use", None),
        "periodic_cvs": meta.get("model_periodic_cvs", None),
        "periodic_cv_units": meta.get("model_periodic_cv_units", None),
    }

    applied: dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None:
            continue
        if key not in effective or effective.get(key) is None:
            effective[key] = value
            applied[key] = value

    if meta and model_input_space_configured(effective):
        return effective, {
            "model_input_source": "checkpoint" if applied else "config",
            "checkpoint_model_input": meta,
            "applied_model_input": applied,
        }

    if not model_input_space_configured(effective):
        raise RuntimeError(
            "RATE_CONSTANT.model is set but model_input_space is not. "
            "Use a *_checkpoint.pt model so the training input metadata can be read, "
            "or add model_input_space: 'features' or 'cv' to RATE_CONSTANT. "
            "For model_input_space: 'cv', also provide cvs_to_use, periodic_cvs, "
            "and periodic_cv_units matching training."
        )

    return effective, {
        "model_input_source": "config",
        "checkpoint_model_input": meta,
        "applied_model_input": applied,
    }


def load_or_infer_q(
    config: dict[str, Any],
    pack,
    n_states: int,
    *,
    raw_n_frames: int | None = None,
    dataset_stride: int = 1,
) -> np.ndarray:
    q_npy = config.get("Q_npy", config.get("q_npy", None))
    if q_npy is not None:
        q = np.load(q_npy).astype(np.float32)
        if raw_n_frames is not None and int(dataset_stride) > 1 and q.shape[0] == int(raw_n_frames):
            q = q[:: int(dataset_stride)].copy()
    else:
        model_path = config.get("model", None)
        if model_path is None:
            raise KeyError("Provide RATE_CONSTANT.Q_npy or RATE_CONSTANT.model.")
        device = setup_device(config.get("device", "cuda:0"))
        model = load_committor_model(model_path, device)
        model_features, _ = select_model_inputs(pack, config)
        q = infer_probabilities(model, model_features.float(), device, batch_size=int(config.get("batch_size", 65536)))
    if q.ndim != 2 or q.shape[1] != n_states:
        raise RuntimeError(f"q shape {q.shape} does not match n_states={n_states}.")
    if q.shape[0] != pack.features.shape[0]:
        raise RuntimeError(
            f"q has {q.shape[0]} frames but the analysis dataset has {pack.features.shape[0]} frames. "
            "If using Q_npy with dataset_stride, provide Q for the same strided dataset or the full unstrided dataset."
        )
    return q


def validate_rate_inputs(
    q: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    n_states: int,
) -> None:
    q = np.asarray(q)
    weights = np.asarray(weights)
    state = np.asarray(state)
    idx0 = np.asarray(idx0)
    idx1 = np.asarray(idx1)
    if q.ndim != 2:
        raise ValueError(f"q must have shape (n_frames, n_states); got {q.shape}.")
    if q.shape[1] != int(n_states):
        raise ValueError(f"q shape {q.shape} does not match n_states={n_states}.")
    if weights.ndim != 1 or weights.shape[0] != q.shape[0]:
        raise ValueError(f"weights must have shape ({q.shape[0]},); got {weights.shape}.")
    if state.ndim != 1 or state.shape[0] != q.shape[0]:
        raise ValueError(f"state must have shape ({q.shape[0]},); got {state.shape}.")
    if idx0.ndim != 1 or idx1.ndim != 1 or idx0.shape != idx1.shape:
        raise ValueError(f"idx0 and idx1 must be matching 1D arrays; got {idx0.shape} and {idx1.shape}.")
    if idx0.size == 0:
        raise ValueError("At least one lagged pair is required.")
    if np.any(idx0 < 0) or np.any(idx1 < 0) or np.any(idx0 >= q.shape[0]) or np.any(idx1 >= q.shape[0]):
        raise ValueError("Lagged indices are out of bounds for q.")
    if not np.all(np.isfinite(q)):
        raise ValueError("q contains non-finite values.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contains non-finite values.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    valid_labels = (state >= -1) & (state < int(n_states))
    if not np.all(valid_labels):
        bad = np.unique(state[~valid_labels])[:8]
        raise ValueError(f"state labels must be -1 or in [0, {int(n_states) - 1}]; found {bad}.")


def positive_weight_masks(
    weights: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    weights = np.asarray(weights, dtype=np.float64)
    idx0 = np.asarray(idx0, dtype=np.int64)
    idx1 = np.asarray(idx1, dtype=np.int64)
    frame_mask = weights > 0.0
    if not np.any(frame_mask):
        raise ValueError("At least one frame must have weight > 0.")
    pair_mask = frame_mask[idx0] & frame_mask[idx1]
    if not np.any(pair_mask):
        raise ValueError("No lagged pairs remain after masking frames with weight == 0.")
    stats = {
        "n_zero_weight_frames": int(np.sum(~frame_mask)),
        "n_positive_weight_frames": int(np.sum(frame_mask)),
        "n_lagged_pairs_before_zero_weight_mask": int(idx0.shape[0]),
        "n_lagged_pairs_after_zero_weight_mask": int(np.sum(pair_mask)),
        "n_lagged_pairs_removed_by_zero_weight_mask": int(idx0.shape[0] - np.sum(pair_mask)),
    }
    return frame_mask, pair_mask, stats


def estimate_flux_profiles(
    q: np.ndarray,
    weights: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    pairs: list[tuple[int, int]],
    thresholds: np.ndarray,
    eps: float,
    tau: float,
    divide_by_tau: bool,
    surface: str,
    chunk_size: int,
    weighted: bool,
    return_current: bool = False,
    return_std: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[np.ndarray, ...]:
    if float(tau) <= 0 and divide_by_tau:
        raise ValueError("tau must be positive when divide_by_tau=True.")
    if float(eps) <= 0:
        raise ValueError("flux_eps must be positive.")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    flux_device = torch.device("cpu" if device is None else device)
    surface = str(surface).lower()
    if surface not in {"qi_decrease", "qj_increase"}:
        raise ValueError("surface must be either 'qi_decrease' or 'qj_increase'.")

    n_pairs = len(pairs)
    n_thr = len(thresholds)
    q_dev = torch.as_tensor(q, dtype=dtype, device=flux_device)
    weights_dev = torch.as_tensor(weights, dtype=dtype, device=flux_device)
    idx0_dev = torch.as_tensor(idx0, dtype=torch.long, device=flux_device)
    idx1_dev = torch.as_tensor(idx1, dtype=torch.long, device=flux_device)
    thresholds_dev = torch.as_tensor(thresholds, dtype=dtype, device=flux_device).view(1, 1, -1)
    pair_i = torch.as_tensor([p[0] for p in pairs], dtype=torch.long, device=flux_device)
    pair_j = torch.as_tensor([p[1] for p in pairs], dtype=torch.long, device=flux_device)

    numer = torch.zeros((n_pairs, n_thr), dtype=dtype, device=flux_device)
    c_numer = torch.zeros(n_pairs, dtype=dtype, device=flux_device)
    c_abs_numer = torch.zeros(n_pairs, dtype=dtype, device=flux_device)
    denom = torch.zeros((), dtype=dtype, device=flux_device)
    if return_std:
        numer_w2 = torch.zeros_like(numer)
        numer_x2_w2 = torch.zeros_like(numer)
        c_numer_w2 = torch.zeros_like(c_numer)
        c_x2_w2 = torch.zeros_like(c_numer)
        c_abs_numer_w2 = torch.zeros_like(c_abs_numer)
        c_abs_x2_w2 = torch.zeros_like(c_abs_numer)
        denom_w2 = torch.zeros((), dtype=dtype, device=flux_device)

    for start in range(0, len(idx0), int(chunk_size)):
        end = min(len(idx0), start + int(chunk_size))
        ids0 = idx0_dev[start:end]
        ids1 = idx1_dev[start:end]
        q_t = q_dev.index_select(0, ids0)
        q_tau = q_dev.index_select(0, ids1)

        C = q_tau[:, pair_j] * q_t[:, pair_i] - q_t[:, pair_j] * q_tau[:, pair_i]
        if surface == "qi_decrease":
            left = q_t[:, pair_i].unsqueeze(-1)
            right = q_tau[:, pair_i].unsqueeze(-1)
            chi = torch.sigmoid((left - thresholds_dev) / float(eps))
            chi = chi * torch.sigmoid((thresholds_dev - right) / float(eps))
        else:
            left = q_t[:, pair_j].unsqueeze(-1)
            right = q_tau[:, pair_j].unsqueeze(-1)
            chi = torch.sigmoid((thresholds_dev - left) / float(eps))
            chi = chi * torch.sigmoid((right - thresholds_dev) / float(eps))

        samples = chi * C.unsqueeze(-1)
        if weighted:
            w = weights_dev.index_select(0, ids0)
            numer += torch.sum(samples * w.view(-1, 1, 1), dim=0)
            c_numer += torch.sum(C * w.view(-1, 1), dim=0)
            c_abs_numer += torch.sum(torch.abs(C) * w.view(-1, 1), dim=0)
            denom += torch.sum(w)
            if return_std:
                w2 = w * w
                abs_C = torch.abs(C)
                numer_w2 += torch.sum(samples * w2.view(-1, 1, 1), dim=0)
                numer_x2_w2 += torch.sum((samples * samples) * w2.view(-1, 1, 1), dim=0)
                c_numer_w2 += torch.sum(C * w2.view(-1, 1), dim=0)
                c_x2_w2 += torch.sum((C * C) * w2.view(-1, 1), dim=0)
                c_abs_numer_w2 += torch.sum(abs_C * w2.view(-1, 1), dim=0)
                c_abs_x2_w2 += torch.sum((abs_C * abs_C) * w2.view(-1, 1), dim=0)
                denom_w2 += torch.sum(w2)
        else:
            numer += torch.sum(samples, dim=0)
            c_numer += torch.sum(C, dim=0)
            c_abs_numer += torch.sum(torch.abs(C), dim=0)
            denom += int(end - start)
            if return_std:
                abs_C = torch.abs(C)
                numer_w2 += torch.sum(samples, dim=0)
                numer_x2_w2 += torch.sum(samples * samples, dim=0)
                c_numer_w2 += torch.sum(C, dim=0)
                c_x2_w2 += torch.sum(C * C, dim=0)
                c_abs_numer_w2 += torch.sum(abs_C, dim=0)
                c_abs_x2_w2 += torch.sum(abs_C * abs_C, dim=0)
                denom_w2 += int(end - start)

    denom = torch.clamp(denom, min=torch.finfo(dtype).tiny)
    J_raw_t = numer / denom
    J_t = J_raw_t
    if divide_by_tau:
        J_t = J_t / float(tau)
    variance_t = torch.mean((J_t - J_t.mean(dim=1, keepdim=True)) ** 2, dim=1)
    J = J_t.detach().cpu().numpy().astype(np.float64)
    variance = variance_t.detach().cpu().numpy().astype(np.float64)
    outputs: list[np.ndarray] = [J, variance]
    C_mean_t = c_numer / denom
    C_abs_mean_t = c_abs_numer / denom
    if return_current:
        outputs.extend(
            [
                C_mean_t.detach().cpu().numpy().astype(np.float64),
                C_abs_mean_t.detach().cpu().numpy().astype(np.float64),
            ]
        )
    if return_std:
        denom_sq = torch.clamp(denom * denom, min=torch.finfo(dtype).tiny)
        J_var_t = torch.clamp(
            numer_x2_w2 - 2.0 * J_raw_t * numer_w2 + (J_raw_t * J_raw_t) * denom_w2,
            min=0.0,
        ) / denom_sq
        if divide_by_tau:
            J_var_t = J_var_t / (float(tau) * float(tau))
        C_var_t = torch.clamp(
            c_x2_w2 - 2.0 * C_mean_t * c_numer_w2 + (C_mean_t * C_mean_t) * denom_w2,
            min=0.0,
        ) / denom_sq
        C_abs_var_t = torch.clamp(
            c_abs_x2_w2 - 2.0 * C_abs_mean_t * c_abs_numer_w2 + (C_abs_mean_t * C_abs_mean_t) * denom_w2,
            min=0.0,
        ) / denom_sq
        outputs.extend(
            [
                torch.sqrt(J_var_t).detach().cpu().numpy().astype(np.float64),
                torch.sqrt(C_var_t).detach().cpu().numpy().astype(np.float64),
                torch.sqrt(C_abs_var_t).detach().cpu().numpy().astype(np.float64),
            ]
        )
    return tuple(outputs)


def estimate_pi(
    q: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    n_states: int,
    mode: str = "labels",
) -> np.ndarray:
    w = weights.astype(np.float64)
    w = w / (np.sum(w) + 1e-300)
    mode = str(mode).lower()
    if mode == "soft":
        pi = np.sum(q.astype(np.float64) * w[:, None], axis=0)
    elif mode == "labels":
        pi = np.asarray([np.sum(w[state == k]) for k in range(n_states)], dtype=np.float64)
        if np.sum(pi) <= 0:
            pi = np.sum(q.astype(np.float64) * w[:, None], axis=0)
    else:
        raise ValueError("pi_mode must be either 'labels' or 'soft'.")
    return pi / (np.sum(pi) + 1e-300)


def estimate_transition_hit_matrix(
    q: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    n_states: int,
    *,
    weighted: bool = True,
    labeled_exits_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    T_hit[i, j] = average q_j(z_exit) over lagged configurations that exit state i.

    z_exit is represented by the t+tau endpoint of pairs with state(t)=i and
    state(t+tau)!=i. Unlabeled endpoints (-1) are included by default because
    transition-region exits still carry a full next-hit committor.
    """
    q = np.asarray(q, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    state = np.asarray(state, dtype=np.int64)
    idx0 = np.asarray(idx0, dtype=np.int64)
    idx1 = np.asarray(idx1, dtype=np.int64)
    T_hit = np.full((int(n_states), int(n_states)), np.nan, dtype=np.float64)
    exit_counts = np.zeros(int(n_states), dtype=np.int64)
    exit_weight = np.zeros(int(n_states), dtype=np.float64)

    for i in range(int(n_states)):
        mask = (state[idx0] == i) & (state[idx1] != i)
        if labeled_exits_only:
            mask &= state[idx1] >= 0
        ids0 = idx0[mask]
        ids1 = idx1[mask]
        exit_counts[i] = int(ids1.size)
        if ids1.size == 0:
            continue
        w = weights[ids0] if weighted else np.ones(ids1.size, dtype=np.float64)
        denom = float(np.sum(w))
        exit_weight[i] = denom
        if denom > 0:
            T_hit[i] = np.sum(q[ids1] * w[:, None], axis=0) / denom
    return T_hit, exit_counts, exit_weight


def matrix_from_pair_values(
    n_states: int,
    pairs: list[tuple[int, int]],
    values: np.ndarray,
    *,
    fill: float = 0.0,
) -> np.ndarray:
    mat = np.full((int(n_states), int(n_states)), float(fill), dtype=np.float64)
    for pair_idx, (i, j) in enumerate(pairs):
        mat[int(i), int(j)] = float(values[pair_idx])
    return mat


def sanitize_rate_matrix(
    k_direct: np.ndarray,
    *,
    negative_policy: str = "clip",
    negative_tol: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Enforce generator-compatible off-diagonal rates.

    Flux estimates can have tiny negative values from finite sampling or
    threshold averaging. A continuous-time generator cannot use negative
    off-diagonal rates, so the default is to clip them before MFPT/P_jump
    calculations and before writing k_direct.
    """
    K = np.asarray(k_direct, dtype=np.float64).copy()
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"k_direct must be a square matrix; got {K.shape}.")
    if not np.all(np.isfinite(K)):
        raise ValueError("k_direct contains non-finite values.")
    negative_tol = float(negative_tol)
    if negative_tol < 0:
        raise ValueError("negative_rate_tolerance must be >= 0.")

    np.fill_diagonal(K, 0.0)
    offdiag = ~np.eye(K.shape[0], dtype=bool)
    negative = (K < 0.0) & offdiag
    significant = (K < -negative_tol) & offdiag
    policy = str(negative_policy).lower()
    if policy not in {"clip", "raise", "allow"}:
        raise ValueError("negative_rate_policy must be 'clip', 'raise', or 'allow'.")
    if policy == "raise" and np.any(significant):
        i, j = np.argwhere(significant)[0]
        raise ValueError(
            "k_direct contains negative off-diagonal rates; "
            f"first significant entry k[{int(i)},{int(j)}]={K[i, j]:.6g}. "
            "Set negative_rate_policy: 'clip' to truncate them to zero."
        )
    if policy == "clip":
        K[negative] = 0.0
    elif policy == "raise" and np.any(negative):
        K[negative] = 0.0

    stats = {
        "negative_rate_policy": policy,
        "negative_rate_tolerance": negative_tol,
        "n_negative_offdiag_rates": int(np.sum(negative)),
        "n_significant_negative_offdiag_rates": int(np.sum(significant)),
        "min_offdiag_rate_before_sanitize": float(np.min(np.asarray(k_direct, dtype=np.float64)[offdiag])),
        "total_negative_offdiag_rate_before_sanitize": (
            float(np.sum(np.asarray(k_direct, dtype=np.float64)[negative])) if np.any(negative) else 0.0
        ),
    }
    return K, stats


def assemble_generator(
    k_direct: np.ndarray,
    *,
    negative_policy: str = "clip",
    negative_tol: float = 0.0,
) -> np.ndarray:
    K, _stats = sanitize_rate_matrix(
        k_direct,
        negative_policy=negative_policy,
        negative_tol=negative_tol,
    )
    for i in range(K.shape[0]):
        K[i, i] = -float(np.sum(K[i]))
    return K


def compute_jump_probabilities(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix; got {K.shape}.")
    P = np.zeros_like(K)
    for i in range(K.shape[0]):
        off = K[i].copy()
        off[i] = 0.0
        denom = float(np.sum(off))
        if denom > 0:
            P[i] = off / denom
            P[i, i] = 0.0
    return P


def _reachability_from_rates(k_direct: np.ndarray) -> np.ndarray:
    rates = np.asarray(k_direct, dtype=np.float64)
    reach = (rates > 0.0) & np.isfinite(rates)
    np.fill_diagonal(reach, True)
    n_states = reach.shape[0]
    for via in range(n_states):
        reach |= reach[:, [via]] & reach[[via], :]
    return reach


def _resolve_kinetic_edge_filter(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("kinetic_edge_filter", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("kinetic_edge_filter must be a mapping.")
    enabled = bool(cfg.get("enabled", config.get("kinetic_edge_filter_enabled", False)))
    min_p = float(cfg.get("min_jump_probability", config.get("kinetic_min_jump_probability", 0.0)) or 0.0)
    if min_p < 0:
        raise ValueError("kinetic_edge_filter.min_jump_probability must be >= 0.")
    min_z_raw = cfg.get("min_rate_zscore", config.get("kinetic_min_rate_zscore", None))
    min_z = None if min_z_raw is None else float(min_z_raw)
    if min_z is not None and min_z < 0:
        raise ValueError("kinetic_edge_filter.min_rate_zscore must be >= 0.")
    preserve = bool(cfg.get("preserve_connectivity", config.get("kinetic_filter_preserve_connectivity", True)))
    return {
        "enabled": enabled,
        "min_jump_probability": min_p,
        "min_rate_zscore": min_z,
        "preserve_connectivity": preserve,
    }


def filter_kinetic_edges(
    k_direct: np.ndarray,
    P_jump: np.ndarray,
    *,
    k_direct_std: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = _resolve_kinetic_edge_filter({} if config is None else config)
    k = np.asarray(k_direct, dtype=np.float64).copy()
    P = np.asarray(P_jump, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError(f"k_direct must be a square matrix; got {k.shape}.")
    if P.shape != k.shape:
        raise ValueError(f"P_jump shape {P.shape} does not match k_direct shape {k.shape}.")
    n_states = k.shape[0]
    offdiag = ~np.eye(n_states, dtype=bool)
    removed = np.zeros_like(k, dtype=bool)
    stats: dict[str, Any] = {
        "enabled": bool(cfg["enabled"]),
        "min_jump_probability": float(cfg["min_jump_probability"]),
        "min_rate_zscore": cfg["min_rate_zscore"],
        "preserve_connectivity": bool(cfg["preserve_connectivity"]),
        "n_candidate_edges": 0,
        "n_removed_edges": 0,
        "n_preserved_for_connectivity": 0,
        "removed_edges": [],
    }
    active_p = float(cfg["min_jump_probability"]) > 0.0
    active_z = cfg["min_rate_zscore"] is not None and float(cfg["min_rate_zscore"]) > 0.0
    if not cfg["enabled"] or not (active_p or active_z):
        return k, removed, stats

    candidate = offdiag & (k > 0.0) & np.isfinite(k)
    if active_p:
        candidate &= np.isfinite(P) & (P < float(cfg["min_jump_probability"]))
    zscore = np.full_like(k, np.inf, dtype=np.float64)
    if active_z:
        if k_direct_std is None:
            raise ValueError("kinetic_edge_filter.min_rate_zscore requires k_direct_std.")
        std = np.asarray(k_direct_std, dtype=np.float64)
        if std.shape != k.shape:
            raise ValueError(f"k_direct_std shape {std.shape} does not match k_direct shape {k.shape}.")
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = np.abs(k) / std
        zscore[(std == 0.0) & (k > 0.0)] = np.inf
        candidate &= np.isfinite(zscore) & (zscore < float(cfg["min_rate_zscore"]))

    stats["n_candidate_edges"] = int(np.sum(candidate))
    if not np.any(candidate):
        return k, removed, stats

    original_reach = _reachability_from_rates(k)
    candidate_edges = [(float(P[i, j]) if np.isfinite(P[i, j]) else np.inf, int(i), int(j)) for i, j in np.argwhere(candidate)]
    candidate_edges.sort()
    for _pij, i, j in candidate_edges:
        old = float(k[i, j])
        k[i, j] = 0.0
        if cfg["preserve_connectivity"]:
            new_reach = _reachability_from_rates(k)
            if not np.all(new_reach[original_reach]):
                k[i, j] = old
                stats["n_preserved_for_connectivity"] = int(stats["n_preserved_for_connectivity"]) + 1
                continue
        removed[i, j] = True
        edge_stats = {
            "state_i": int(i),
            "state_j": int(j),
            "k_direct": old,
            "P_jump": float(P[i, j]) if np.isfinite(P[i, j]) else np.nan,
        }
        if active_z:
            edge_stats["rate_zscore"] = float(zscore[i, j]) if np.isfinite(zscore[i, j]) else np.nan
        stats["removed_edges"].append(edge_stats)

    stats["n_removed_edges"] = int(np.sum(removed))
    return k, removed, stats


def compute_mfpt_matrix(K: np.ndarray) -> np.ndarray:
    """
    Continuous-time MFPTs from the full generator.

    For each target j, remove target j from K and solve K_sub m = -1 with
    m_j=0. Indirect paths are included through the coupled linear system.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix; got {K.shape}.")
    n_states = K.shape[0]
    mfpt = np.zeros((n_states, n_states), dtype=np.float64)
    for target in range(n_states):
        keep = np.array([i for i in range(n_states) if i != target], dtype=np.int64)
        K_sub = K[np.ix_(keep, keep)]
        rhs = -np.ones(n_states - 1, dtype=np.float64)
        try:
            sol = np.linalg.solve(K_sub, rhs)
        except np.linalg.LinAlgError:
            sol = np.full(n_states - 1, np.nan, dtype=np.float64)
        mfpt[keep, target] = sol
        mfpt[target, target] = 0.0
    return mfpt


def compute_mfpt_rate_matrix(mfpt: np.ndarray) -> np.ndarray:
    """
    Effective state-to-state rates from the full-generator MFPT matrix.

    Off-diagonal entries are 1 / MFPT[i, j]. The diagonal is set to zero
    because self-passage is not a kinetic transition rate.
    """
    mfpt = np.asarray(mfpt, dtype=np.float64)
    if mfpt.ndim != 2 or mfpt.shape[0] != mfpt.shape[1]:
        raise ValueError(f"mfpt must be a square matrix; got {mfpt.shape}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = 1.0 / mfpt
    rate[~np.isfinite(rate)] = np.nan
    np.fill_diagonal(rate, 0.0)
    return rate


def _nanstd(values: list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < 2:
        return np.full(arr.shape[1:], np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    count = np.sum(finite, axis=0)
    value_scale = np.max(np.where(finite, np.abs(arr), 0.0), axis=0)
    nonzero = (count > 0) & np.isfinite(value_scale) & (value_scale > 0)
    scaled_arr = np.divide(arr, value_scale, out=np.zeros_like(arr), where=finite & nonzero[None, ...])
    scaled_mean = np.divide(
        np.sum(scaled_arr, axis=0),
        count,
        out=np.full(arr.shape[1:], np.nan),
        where=nonzero,
    )
    mean = value_scale * scaled_mean
    mean[(count > 0) & np.isfinite(value_scale) & (value_scale == 0)] = 0.0
    centered = np.where(finite, arr - mean, 0.0)
    scale = np.max(np.where(finite, np.abs(centered), 0.0), axis=0)
    valid = (count > 1) & np.isfinite(scale) & (scale > 0)
    scaled = np.divide(centered, scale, out=np.zeros_like(centered), where=valid[None, ...])
    scaled_var = np.divide(
        np.sum(scaled * scaled, axis=0),
        count - 1,
        out=np.full(arr.shape[1:], np.nan),
        where=valid,
    )
    out = scale * np.sqrt(np.maximum(scaled_var, 0.0))
    out[(count > 1) & np.isfinite(scale) & (scale == 0)] = 0.0
    out[count <= 1] = np.nan
    return out


def _nansem(values: list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < 2:
        return np.full(arr.shape[1:], np.nan, dtype=np.float64)
    std = _nanstd(values)
    count = np.sum(np.isfinite(arr), axis=0)
    return np.divide(std, np.sqrt(count), out=np.full_like(std, np.nan), where=count > 1)


def _nanjackknife_std(values: list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] < 2:
        return np.full(arr.shape[1:], np.nan, dtype=np.float64)
    std = _nanstd(values)
    count = np.sum(np.isfinite(arr), axis=0)
    factor = np.divide(count - 1, np.sqrt(count), out=np.zeros_like(std), where=count > 1)
    out = std * factor
    out[count <= 1] = np.nan
    return out


def _mean_std_from_component_stds(stds: np.ndarray, axis: int = 1) -> np.ndarray:
    stds = np.asarray(stds, dtype=np.float64)
    finite = np.isfinite(stds)
    count = np.sum(finite, axis=axis)
    sum_var = np.sum(np.where(finite, stds * stds, 0.0), axis=axis)
    return np.divide(np.sqrt(sum_var), count, out=np.full(count.shape, np.nan, dtype=np.float64), where=count > 0)


def _combine_std(*parts: np.ndarray | None) -> np.ndarray | None:
    arrays = [np.asarray(part, dtype=np.float64) for part in parts if part is not None]
    if not arrays:
        return None
    total = np.zeros_like(arrays[0], dtype=np.float64)
    any_finite = np.zeros_like(arrays[0], dtype=bool)
    for arr in arrays:
        finite = np.isfinite(arr)
        total += np.where(finite, arr * arr, 0.0)
        any_finite |= finite
    out = np.sqrt(total)
    out[~any_finite] = np.nan
    return out


def _prefer_std(primary: np.ndarray | None, fallback: np.ndarray | None) -> np.ndarray | None:
    if primary is None:
        return None if fallback is None else np.asarray(fallback, dtype=np.float64)
    primary_arr = np.asarray(primary, dtype=np.float64)
    if fallback is None:
        return primary_arr
    fallback_arr = np.asarray(fallback, dtype=np.float64)
    return np.where(np.isfinite(primary_arr), primary_arr, fallback_arr)


def _resolve_error_estimator(config: dict[str, Any]) -> str:
    raw = str(config.get("error_estimator", config.get("error_method", "block_jackknife"))).lower()
    aliases = {
        "jackknife": "block_jackknife",
        "delete_block_jackknife": "block_jackknife",
        "delete-one-block": "block_jackknife",
        "delete_one_block": "block_jackknife",
        "slices": "slice_sem",
        "slice": "slice_sem",
        "lagged_pair_slices": "slice_sem",
        "sem": "slice_sem",
        "weighted": "direct",
        "counting": "direct",
        "weighted_counting": "direct",
        "combined": "combined",
        "quadrature": "combined",
    }
    estimator = aliases.get(raw, raw)
    if estimator not in {"block_jackknife", "slice_sem", "direct", "combined"}:
        raise ValueError("error_estimator must be 'block_jackknife', 'slice_sem', 'direct', or 'combined'.")
    return estimator


def _pick_std(
    direct: np.ndarray | None,
    resampled: np.ndarray | None,
    estimator: str,
) -> np.ndarray | None:
    if estimator == "combined":
        return _combine_std(direct, resampled)
    if estimator == "direct":
        return None if direct is None else np.asarray(direct, dtype=np.float64)
    return _prefer_std(resampled, direct)


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray, mean: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0 or values.shape[0] == 0:
        return np.full(mean.shape, np.nan, dtype=np.float64)
    centered = values - mean
    var = np.sum((weights[:, None] * centered) ** 2, axis=0) / max(denom * denom, 1e-300)
    return np.sqrt(np.maximum(var, 0.0))


def estimate_pi_std(
    q: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    pi: np.ndarray,
    n_states: int,
    mode: str = "labels",
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w))
    if denom <= 0:
        return np.full(int(n_states), np.nan, dtype=np.float64)
    mode = str(mode).lower()
    if mode == "soft" or (mode == "labels" and not np.any(np.asarray(state) >= 0)):
        values = np.asarray(q, dtype=np.float64)
    elif mode == "labels":
        labels = np.asarray(state, dtype=np.int64)
        values = np.column_stack([(labels == k).astype(np.float64) for k in range(int(n_states))])
    else:
        raise ValueError("pi_mode must be either 'labels' or 'soft'.")
    return _weighted_mean_std(values, w, np.asarray(pi, dtype=np.float64))


def estimate_transition_hit_matrix_std(
    q: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    T_hit: np.ndarray,
    n_states: int,
    *,
    weighted: bool = True,
    labeled_exits_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    state = np.asarray(state, dtype=np.int64)
    T_hit = np.asarray(T_hit, dtype=np.float64)
    T_std = np.full_like(T_hit, np.nan, dtype=np.float64)
    count_std = np.zeros(int(n_states), dtype=np.float64)
    weight_std = np.zeros(int(n_states), dtype=np.float64)
    for i in range(int(n_states)):
        mask = (state[idx0] == i) & (state[idx1] != i)
        if labeled_exits_only:
            mask &= state[idx1] >= 0
        ids0 = idx0[mask]
        ids1 = idx1[mask]
        count_std[i] = np.sqrt(float(ids1.size))
        if ids1.size == 0:
            continue
        w = weights[ids0] if weighted else np.ones(ids1.size, dtype=np.float64)
        weight_std[i] = float(np.sqrt(np.sum(w * w)))
        if np.sum(w) > 0:
            T_std[i] = _weighted_mean_std(q[ids1], w, T_hit[i])
    return T_std, count_std, weight_std


def _slice_lagged_pairs(idx0: np.ndarray, n_slices: int, min_pairs: int) -> list[np.ndarray]:
    n_pairs = int(idx0.shape[0])
    if n_pairs <= 0:
        return []
    n_slices = max(1, min(int(n_slices), n_pairs))
    slices = [part.astype(np.int64) for part in np.array_split(np.arange(n_pairs, dtype=np.int64), n_slices)]
    return [part for part in slices if part.size >= int(min_pairs)]


def _rate_matrices_from_estimates(
    n_states: int,
    pairs: list[tuple[int, int]],
    J: np.ndarray,
    C_mean: np.ndarray,
    C_abs_mean: np.ndarray,
    pi: np.ndarray,
    time_unit: str,
    kinetic_edge_filter: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    _, J_matrix, k_direct = build_rate_table(pairs, J, np.zeros(len(pairs), dtype=np.float64), pi, time_unit)
    k_direct, _rate_sanitize = sanitize_rate_matrix(k_direct)
    C_matrix = matrix_from_pair_values(n_states, pairs, C_mean, fill=0.0)
    C_abs_matrix = matrix_from_pair_values(n_states, pairs, C_abs_mean, fill=0.0)
    K_unfiltered = assemble_generator(k_direct)
    P_jump_unfiltered = compute_jump_probabilities(K_unfiltered)
    slice_filter = dict(kinetic_edge_filter or {})
    slice_filter["min_rate_zscore"] = None
    k_direct, _removed, _filter_stats = filter_kinetic_edges(
        k_direct,
        P_jump_unfiltered,
        config={"kinetic_edge_filter": slice_filter},
    )
    K = assemble_generator(k_direct)
    P_jump = compute_jump_probabilities(K)
    mfpt = compute_mfpt_matrix(K)
    k_mfpt = compute_mfpt_rate_matrix(mfpt)
    return {
        "J_thresholds": np.asarray(J, dtype=np.float64),
        "J_matrix": J_matrix,
        "C_matrix": C_matrix,
        "C_abs_matrix": C_abs_matrix,
        "k_direct": k_direct,
        "k_matrix": k_direct,
        "K": K,
        "P_jump": P_jump,
        "MFPT": mfpt,
        "k_mfpt": k_mfpt,
    }


def propagate_generator_std(K: np.ndarray, K_offdiag_std: np.ndarray) -> dict[str, np.ndarray]:
    K = np.asarray(K, dtype=np.float64)
    K_offdiag_std = np.asarray(K_offdiag_std, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix; got {K.shape}.")
    if K_offdiag_std.shape != K.shape:
        raise ValueError(f"K_offdiag_std shape {K_offdiag_std.shape} does not match K shape {K.shape}.")
    n_states = K.shape[0]
    K_std = np.where(np.isfinite(K_offdiag_std), K_offdiag_std, np.nan)
    np.fill_diagonal(K_std, 0.0)
    for i in range(n_states):
        row = np.delete(K_std[i], i)
        K_std[i, i] = float(np.sqrt(np.nansum(row * row)))

    p_var = np.zeros_like(K, dtype=np.float64)
    mfpt_var = np.zeros_like(K, dtype=np.float64)
    k_mfpt_var = np.zeros_like(K, dtype=np.float64)
    for i in range(n_states):
        for j in range(n_states):
            step = float(K_offdiag_std[i, j])
            if i == j or not np.isfinite(step) or step <= 0:
                continue
            plus = K.copy()
            minus = K.copy()
            plus[i, j] = max(0.0, plus[i, j] + step)
            minus[i, j] = max(0.0, minus[i, j] - step)
            plus[i, i] = -float(np.sum(np.delete(plus[i], i)))
            minus[i, i] = -float(np.sum(np.delete(minus[i], i)))
            p_plus = compute_jump_probabilities(plus)
            p_minus = compute_jump_probabilities(minus)
            mfpt_plus = compute_mfpt_matrix(plus)
            mfpt_minus = compute_mfpt_matrix(minus)
            k_mfpt_plus = compute_mfpt_rate_matrix(mfpt_plus)
            k_mfpt_minus = compute_mfpt_rate_matrix(mfpt_minus)
            p_var += np.nan_to_num(0.5 * (p_plus - p_minus), nan=0.0) ** 2
            mfpt_var += np.nan_to_num(0.5 * (mfpt_plus - mfpt_minus), nan=0.0) ** 2
            k_mfpt_var += np.nan_to_num(0.5 * (k_mfpt_plus - k_mfpt_minus), nan=0.0) ** 2
    return {
        "K": K_std,
        "P_jump": np.sqrt(p_var),
        "MFPT": np.sqrt(mfpt_var),
        "k_mfpt": np.sqrt(k_mfpt_var),
    }


def propagate_rate_matrix_std(k_direct: np.ndarray, k_direct_std: np.ndarray) -> dict[str, np.ndarray]:
    return propagate_generator_std(assemble_generator(k_direct), k_direct_std)


def estimate_slice_rate_std(
    q: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    n_states: int,
    pairs: list[tuple[int, int]],
    thresholds: np.ndarray,
    *,
    tau: float,
    divide_by_tau: bool,
    eps: float,
    surface: str,
    chunk_size: int,
    weighted_flux: bool,
    weighted_hits: bool,
    labeled_exits_only: bool,
    pi_mode: str,
    time_unit: str,
    n_slices: int,
    min_pairs: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
    kinetic_edge_filter: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    collected: dict[str, list[np.ndarray]] = {
        "pi": [],
        "T_hit": [],
        "exit_counts": [],
        "exit_weight": [],
        "J_thresholds": [],
        "J_matrix": [],
        "C_matrix": [],
        "C_abs_matrix": [],
        "k_direct": [],
        "k_matrix": [],
        "K": [],
        "P_jump": [],
        "MFPT": [],
        "k_mfpt": [],
    }
    for pair_slice in _slice_lagged_pairs(idx0, n_slices, min_pairs):
        s_idx0 = idx0[pair_slice]
        s_idx1 = idx1[pair_slice]
        frame_ids = np.unique(s_idx0)
        if frame_ids.size == 0:
            continue
        s_pi = estimate_pi(q[frame_ids], weights[frame_ids], state[frame_ids], n_states, mode=pi_mode)
        s_flux = estimate_flux_profiles(
            q=q,
            weights=weights,
            idx0=s_idx0,
            idx1=s_idx1,
            pairs=pairs,
            thresholds=thresholds,
            eps=eps,
            tau=tau,
            divide_by_tau=divide_by_tau,
            surface=surface,
            chunk_size=chunk_size,
            weighted=weighted_flux,
            return_current=True,
            device=device,
            dtype=dtype,
        )
        s_J, _s_variance, s_C_mean, s_C_abs_mean = s_flux
        s_T_hit, s_exit_counts, s_exit_weight = estimate_transition_hit_matrix(
            q=q,
            weights=weights,
            state=state,
            idx0=s_idx0,
            idx1=s_idx1,
            n_states=n_states,
            weighted=weighted_hits,
            labeled_exits_only=labeled_exits_only,
        )
        collected["pi"].append(s_pi)
        collected["T_hit"].append(s_T_hit)
        collected["exit_counts"].append(s_exit_counts.astype(np.float64))
        collected["exit_weight"].append(s_exit_weight)
        for name, value in _rate_matrices_from_estimates(
            n_states, pairs, s_J, s_C_mean, s_C_abs_mean, s_pi, time_unit, kinetic_edge_filter
        ).items():
            collected[name].append(value)
    used = len(collected["pi"])
    if used < 2:
        return {}, used
    return {name: _nansem(values) for name, values in collected.items()}, used


def estimate_block_jackknife_rate_std(
    q: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    n_states: int,
    pairs: list[tuple[int, int]],
    thresholds: np.ndarray,
    *,
    tau: float,
    divide_by_tau: bool,
    eps: float,
    surface: str,
    chunk_size: int,
    weighted_flux: bool,
    weighted_hits: bool,
    labeled_exits_only: bool,
    pi_mode: str,
    time_unit: str,
    n_blocks: int,
    min_pairs: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
    kinetic_edge_filter: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Delete-one-contiguous-block jackknife standard errors.

    Each replicate drops one lagged-pair block and recomputes the full rate
    estimator on the remaining data. This is more stable for ratio estimates
    than computing rates on short slices, because pi_i is estimated from nearly
    the full trajectory in every replicate.
    """
    pair_blocks = _slice_lagged_pairs(idx0, n_blocks, min_pairs)
    if len(pair_blocks) < 2:
        return {}, len(pair_blocks)

    collected: dict[str, list[np.ndarray]] = {
        "pi": [],
        "T_hit": [],
        "exit_counts": [],
        "exit_weight": [],
        "J_thresholds": [],
        "J_matrix": [],
        "C_matrix": [],
        "C_abs_matrix": [],
        "k_direct": [],
        "k_matrix": [],
        "K": [],
        "P_jump": [],
        "MFPT": [],
        "k_mfpt": [],
    }
    all_pair_ids = np.arange(idx0.shape[0], dtype=np.int64)
    for omitted in pair_blocks:
        keep_mask = np.ones(idx0.shape[0], dtype=bool)
        keep_mask[omitted] = False
        keep_pair_ids = all_pair_ids[keep_mask]
        if keep_pair_ids.size < int(min_pairs):
            continue
        jk_idx0 = idx0[keep_pair_ids]
        jk_idx1 = idx1[keep_pair_ids]
        frame_ids = np.unique(jk_idx0)
        if frame_ids.size == 0:
            continue
        jk_pi = estimate_pi(q[frame_ids], weights[frame_ids], state[frame_ids], n_states, mode=pi_mode)
        jk_flux = estimate_flux_profiles(
            q=q,
            weights=weights,
            idx0=jk_idx0,
            idx1=jk_idx1,
            pairs=pairs,
            thresholds=thresholds,
            eps=eps,
            tau=tau,
            divide_by_tau=divide_by_tau,
            surface=surface,
            chunk_size=chunk_size,
            weighted=weighted_flux,
            return_current=True,
            device=device,
            dtype=dtype,
        )
        jk_J, _jk_variance, jk_C_mean, jk_C_abs_mean = jk_flux
        jk_T_hit, jk_exit_counts, jk_exit_weight = estimate_transition_hit_matrix(
            q=q,
            weights=weights,
            state=state,
            idx0=jk_idx0,
            idx1=jk_idx1,
            n_states=n_states,
            weighted=weighted_hits,
            labeled_exits_only=labeled_exits_only,
        )
        collected["pi"].append(jk_pi)
        collected["T_hit"].append(jk_T_hit)
        collected["exit_counts"].append(jk_exit_counts.astype(np.float64))
        collected["exit_weight"].append(jk_exit_weight)
        for name, value in _rate_matrices_from_estimates(
            n_states, pairs, jk_J, jk_C_mean, jk_C_abs_mean, jk_pi, time_unit, kinetic_edge_filter
        ).items():
            collected[name].append(value)

    used = len(collected["pi"])
    if used < 2:
        return {}, used
    return {name: _nanjackknife_std(values) for name, values in collected.items()}, used


def build_rate_std_table(
    pairs: list[tuple[int, int]],
    pi_std: np.ndarray,
    J_matrix_std: np.ndarray,
    k_direct_std: np.ndarray,
    time_unit: str,
    k_mfpt_std: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    k_rate_std = k_direct_std if k_mfpt_std is None else k_mfpt_std
    for i, j in pairs:
        row = {
            "state_i": int(i),
            "state_j": int(j),
            "pi_i_std": float(pi_std[i]) if np.isfinite(pi_std[i]) else np.nan,
            "J_ij_std": float(J_matrix_std[i, j]) if np.isfinite(J_matrix_std[i, j]) else np.nan,
            "k_direct_ij_std": float(k_direct_std[i, j]) if np.isfinite(k_direct_std[i, j]) else np.nan,
        }
        if k_mfpt_std is not None:
            row["k_mfpt_ij_std"] = float(k_mfpt_std[i, j]) if np.isfinite(k_mfpt_std[i, j]) else np.nan
        row["k_ij_std"] = float(k_rate_std[i, j]) if np.isfinite(k_rate_std[i, j]) else np.nan
        row["k_unit"] = f"1/{time_unit}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_flux_profiles_csv(
    path: str,
    pairs: list[tuple[int, int]],
    thresholds: np.ndarray,
    J: np.ndarray,
    variance: np.ndarray,
) -> None:
    rows = []
    for p_idx, (i, j) in enumerate(pairs):
        for t_idx, c in enumerate(thresholds):
            rows.append(
                {
                    "state_i": i,
                    "state_j": j,
                    "threshold": float(c),
                    "J_ij": float(J[p_idx, t_idx]),
                    "threshold_variance": float(variance[p_idx]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_flux_profiles_std_csv(
    path: str,
    pairs: list[tuple[int, int]],
    thresholds: np.ndarray,
    J_std: np.ndarray,
) -> None:
    rows = []
    for p_idx, (i, j) in enumerate(pairs):
        for t_idx, c in enumerate(thresholds):
            rows.append(
                {
                    "state_i": i,
                    "state_j": j,
                    "threshold": float(c),
                    "J_ij_std": float(J_std[p_idx, t_idx]) if np.isfinite(J_std[p_idx, t_idx]) else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_matrix_csv(path: str, matrix: np.ndarray, *, index_name: str = "state_i") -> None:
    pd.DataFrame(np.asarray(matrix, dtype=np.float64)).to_csv(path, index_label=index_name)


def write_population_csv(path: str, pi: np.ndarray) -> None:
    rows = [{"state": int(i), "pi": float(pi_i)} for i, pi_i in enumerate(np.asarray(pi, dtype=np.float64))]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_population_std_csv(path: str, pi_std: np.ndarray) -> None:
    rows = [
        {"state": int(i), "pi_std": float(std_i) if np.isfinite(std_i) else np.nan}
        for i, std_i in enumerate(np.asarray(pi_std, dtype=np.float64))
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_transition_hit_csv(
    path: str,
    T_hit: np.ndarray,
    exit_counts: np.ndarray,
    exit_weight: np.ndarray,
) -> None:
    rows = []
    for i in range(T_hit.shape[0]):
        for j in range(T_hit.shape[1]):
            rows.append(
                {
                    "state_i": int(i),
                    "state_j": int(j),
                    "T_hit_ij": float(T_hit[i, j]) if np.isfinite(T_hit[i, j]) else np.nan,
                    "exit_count_i": int(exit_counts[i]),
                    "exit_weight_i": float(exit_weight[i]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_transition_hit_std_csv(
    path: str,
    T_hit_std: np.ndarray,
    exit_counts_std: np.ndarray,
    exit_weight_std: np.ndarray,
) -> None:
    rows = []
    for i in range(T_hit_std.shape[0]):
        for j in range(T_hit_std.shape[1]):
            rows.append(
                {
                    "state_i": int(i),
                    "state_j": int(j),
                    "T_hit_ij_std": float(T_hit_std[i, j]) if np.isfinite(T_hit_std[i, j]) else np.nan,
                    "exit_count_i_std": float(exit_counts_std[i]) if np.isfinite(exit_counts_std[i]) else np.nan,
                    "exit_weight_i_std": float(exit_weight_std[i]) if np.isfinite(exit_weight_std[i]) else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_vector_bar(values: np.ndarray, out_path: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=160)
    x = np.arange(len(values))
    ax.bar(x, values)
    ax.set_xlabel("state")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_matrix_heatmap(matrix: np.ndarray, out_path: str, title: str, color_label: str) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=160)
    im = ax.imshow(matrix, interpolation="nearest", aspect="auto")
    ax.set_xlabel("state j")
    ax.set_ylabel("state i")
    ax.set_title(title)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_flux_threshold_profiles(
    pairs: list[tuple[int, int]],
    thresholds: np.ndarray,
    J: np.ndarray,
    out_dir: str,
    fmt: str = "png",
) -> list[str]:
    flux_dir = ensure_dir(os.path.join(out_dir, "flux_profiles"))
    paths = []
    for p_idx, (i, j) in enumerate(pairs):
        path = os.path.join(flux_dir, f"J_{i}_{j}.{fmt}")
        fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=160)
        ax.plot(thresholds, J[p_idx], marker="o", linewidth=1.2)
        ax.axhline(float(np.mean(J[p_idx])), color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("isocommittor threshold c")
        ax.set_ylabel(f"J_{i}_{j}(c)")
        ax.set_title(f"{i} -> {j}")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    if len(pairs) <= 20:
        path = os.path.join(out_dir, f"flux_profiles_all.{fmt}")
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
        for p_idx, (i, j) in enumerate(pairs):
            ax.plot(thresholds, J[p_idx], marker="o", linewidth=1.0, label=f"{i}->{j}")
        ax.set_xlabel("isocommittor threshold c")
        ax.set_ylabel("J_ij(c)")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_rate_diagnostics(
    out_dir: str,
    pi: np.ndarray,
    T_hit: np.ndarray,
    J_matrix: np.ndarray,
    k_direct: np.ndarray,
    K: np.ndarray,
    P_jump: np.ndarray,
    mfpt: np.ndarray,
    pairs: list[tuple[int, int]],
    thresholds: np.ndarray,
    J_thresholds: np.ndarray,
    fmt: str = "png",
) -> list[str]:
    plot_dir = ensure_dir(os.path.join(out_dir, "diagnostics"))
    specs = [
        ("T_hit", T_hit, "exit-hit committor averages"),
        ("J_matrix", J_matrix, "direct reactive flux"),
        ("k_direct", k_direct, "direct rate"),
        ("K", K, "generator"),
        ("P_jump", P_jump, "jump probability"),
        ("MFPT", mfpt, "MFPT"),
    ]
    paths = []
    pi_path = os.path.join(plot_dir, f"pi.{fmt}")
    plot_vector_bar(pi, pi_path, "population", "empirical state populations")
    paths.append(pi_path)
    for name, matrix, label in specs:
        path = os.path.join(plot_dir, f"{name}.{fmt}")
        plot_matrix_heatmap(matrix, path, name, label)
        paths.append(path)
    paths.extend(plot_flux_threshold_profiles(pairs, thresholds, J_thresholds, out_dir, fmt=fmt))
    return paths


def build_rate_table(
    pairs: list[tuple[int, int]],
    J: np.ndarray,
    variance: np.ndarray,
    pi: np.ndarray,
    time_unit: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n_states = int(pi.shape[0])
    Jbar = J.mean(axis=1)
    J_matrix = np.zeros((n_states, n_states), dtype=np.float64)
    k_direct = np.zeros((n_states, n_states), dtype=np.float64)
    rows = []
    for p_idx, (i, j) in enumerate(pairs):
        kij = Jbar[p_idx] / max(float(pi[i]), 1e-300)
        J_matrix[i, j] = Jbar[p_idx]
        k_direct[i, j] = kij
        rows.append(
            {
                "state_i": i,
                "state_j": j,
                "pi_i": float(pi[i]),
                "J_ij": float(Jbar[p_idx]),
                "J_threshold_variance": float(variance[p_idx]),
                "k_direct_ij": float(kij),
                "k_ij": float(kij),
                "k_unit": f"1/{time_unit}",
            }
        )
    return pd.DataFrame(rows), J_matrix, k_direct


def add_mfpt_rates_to_table(
    table: pd.DataFrame,
    pairs: list[tuple[int, int]],
    k_mfpt: np.ndarray,
) -> pd.DataFrame:
    table = table.copy()
    k_mfpt_ij = [float(k_mfpt[i, j]) if np.isfinite(k_mfpt[i, j]) else np.nan for i, j in pairs]
    table["k_mfpt_ij"] = k_mfpt_ij
    table["k_ij"] = k_mfpt_ij
    columns = [
        "state_i",
        "state_j",
        "pi_i",
        "J_ij",
        "J_threshold_variance",
        "k_direct_ij",
        "k_mfpt_ij",
        "k_ij",
        "k_unit",
    ]
    return table[[col for col in columns if col in table.columns]]


def run(config: dict[str, Any]) -> dict[str, Any]:
    config, model_input_summary = apply_checkpoint_model_input_config(config)
    out_dir = ensure_dir(config.get("out_dir", "./next_hit_rate"))
    dataset_path = config.get("dataset", config.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("RATE_CONSTANT config needs 'dataset' or 'dataset_path'.")

    dataset_stride = int(config.get("dataset_stride", 1))
    raw_pack = load_dataset(dataset_path)
    burn_in_mask_raw, burn_in_stats = trajectory_burn_in_mask(
        raw_pack.traj_id,
        int(raw_pack.features.shape[0]),
        int(config.get("discard_first_n_frames", 0)),
    )
    pack = apply_stride(raw_pack, dataset_stride)
    burn_in_mask = burn_in_mask_raw[::dataset_stride]
    if burn_in_mask.shape[0] != int(pack.features.shape[0]):
        raise RuntimeError("Internal burn-in mask length does not match the strided dataset.")
    burn_in_stats["dataset_stride"] = int(dataset_stride)
    burn_in_stats["n_frames_after_stride_before_discard"] = int(pack.features.shape[0])
    burn_in_stats["n_frames_after_stride_retained"] = int(np.count_nonzero(burn_in_mask))
    if burn_in_stats["n_frames_discarded"] > 0:
        print(
            f"[RATE] Discarded the first {burn_in_stats['discard_first_n_frames']} frame(s) "
            f"from each of {burn_in_stats['n_trajectory_blocks']} trajectory block(s): "
            f"{burn_in_stats['n_frames_discarded']} frames removed."
        )
    n_states = infer_n_states(pack, config.get("n_states", None))
    q = load_or_infer_q(
        config,
        pack,
        n_states,
        raw_n_frames=int(raw_pack.features.shape[0]),
        dataset_stride=dataset_stride,
    )
    checks = check_probability_rows(q, atol=float(config.get("normalization_atol", 1e-4)))

    timing = resolve_lag_timing(config, dataset_stride)
    lag_index_step = int(timing["lag_index_step"])
    tau = float(timing["tau"])
    time_unit = str(timing["time_unit"])
    flux_device = setup_device(config.get("flux_device", config.get("device", "cuda:0")))
    flux_dtype = resolve_flux_dtype(config.get("flux_dtype", None), flux_device)

    idx0_t, idx1_t = build_lagged_indices(
        q.shape[0],
        lag=lag_index_step,
        traj_id=pack.traj_id,
        allow_cross=bool(config.get("allow_cross_traj_pairs", False)),
    )
    idx0 = idx0_t.numpy()
    idx1 = idx1_t.numpy()
    n_pairs_before_burn_in = int(idx0.size)
    burn_in_pair_mask = burn_in_mask[idx0] & burn_in_mask[idx1]
    idx0 = idx0[burn_in_pair_mask]
    idx1 = idx1[burn_in_pair_mask]
    if idx0.size == 0:
        raise RuntimeError(
            "No lagged pairs remain after discard_first_n_frames filtering. "
            "Reduce discard_first_n_frames or lag."
        )
    burn_in_stats["n_lagged_pairs_before_discard"] = n_pairs_before_burn_in
    burn_in_stats["n_lagged_pairs_discarded"] = int(n_pairs_before_burn_in - idx0.size)
    burn_in_stats["n_lagged_pairs_retained"] = int(idx0.size)

    threshold_t = make_thresholds(
        config.get("thresholds", None),
        n_thresholds=int(config.get("n_thresholds", 9)),
        start=float(config.get("threshold_start", 0.1)),
        stop=float(config.get("threshold_stop", 0.9)),
    )
    thresholds = threshold_t.numpy()
    pairs = resolve_ordered_pairs(
        n_states,
        adjacency=config.get("adjacency", config.get("flux_pairs", None)),
        symmetric_adjacency=bool(config.get("symmetric_adjacency", True)),
    )

    weights = pack.weights.numpy().astype(np.float64)
    state = pack.state.numpy().astype(np.int64)
    validate_rate_inputs(q, weights, state, idx0, idx1, n_states)
    positive_frame_mask, positive_pair_mask, zero_weight_mask_stats = positive_weight_masks(weights, idx0, idx1)
    positive_frame_mask &= burn_in_mask
    idx0 = idx0[positive_pair_mask]
    idx1 = idx1[positive_pair_mask]
    q_weighted = q[positive_frame_mask]
    weights_weighted = weights[positive_frame_mask]
    state_weighted = state[positive_frame_mask]
    J, variance, C_mean, C_abs_mean, J_thresholds_direct_std, C_direct_std, C_abs_direct_std = estimate_flux_profiles(
        q=q,
        weights=weights,
        idx0=idx0,
        idx1=idx1,
        pairs=pairs,
        thresholds=thresholds,
        eps=float(config.get("flux_eps", 0.02)),
        tau=tau,
        divide_by_tau=bool(config.get("divide_by_tau", True)),
        surface=str(config.get("flux_surface", "qi_decrease")),
        chunk_size=int(config.get("chunk_size", 20000)),
        weighted=bool(config.get("weighted_flux", True)),
        return_current=True,
        return_std=True,
        device=flux_device,
        dtype=flux_dtype,
    )
    pi = estimate_pi(
        q=q_weighted,
        weights=weights_weighted,
        state=state_weighted,
        n_states=n_states,
        mode=str(config.get("pi_mode", "labels")),
    )
    T_hit, exit_counts, exit_weight = estimate_transition_hit_matrix(
        q=q,
        weights=weights,
        state=state,
        idx0=idx0,
        idx1=idx1,
        n_states=n_states,
        weighted=bool(config.get("weighted_hits", config.get("weighted_flux", True))),
        labeled_exits_only=bool(config.get("labeled_exits_only", False)),
    )
    table, J_matrix, k_direct = build_rate_table(pairs, J, variance, pi, time_unit)
    k_direct_raw = k_direct.copy()
    k_direct, rate_sanitize = sanitize_rate_matrix(
        k_direct,
        negative_policy=str(config.get("negative_rate_policy", "clip")),
        negative_tol=float(config.get("negative_rate_tolerance", 0.0)),
    )
    if rate_sanitize["n_negative_offdiag_rates"] > 0 and rate_sanitize["negative_rate_policy"] == "clip":
        print(
            "[RATE] Clipped "
            f"{rate_sanitize['n_negative_offdiag_rates']} negative off-diagonal k_direct entries "
            "to zero before generator/MFPT calculations."
        )
    for row_idx, (i, j) in enumerate(pairs):
        table.loc[row_idx, "k_direct_ij"] = float(k_direct[i, j])
        table.loc[row_idx, "k_ij"] = float(k_direct[i, j])
    C_matrix = matrix_from_pair_values(n_states, pairs, C_mean, fill=0.0)
    C_abs_matrix = matrix_from_pair_values(n_states, pairs, C_abs_mean, fill=0.0)
    k_direct_unfiltered = k_direct.copy()
    kinetic_edge_filter_cfg = _resolve_kinetic_edge_filter(config)

    pi_direct_std = estimate_pi_std(
        q_weighted, weights_weighted, state_weighted, pi, n_states, mode=str(config.get("pi_mode", "labels"))
    )
    T_hit_direct_std, exit_counts_direct_std, exit_weight_direct_std = estimate_transition_hit_matrix_std(
        q=q,
        weights=weights,
        state=state,
        idx0=idx0,
        idx1=idx1,
        T_hit=T_hit,
        n_states=n_states,
        weighted=bool(config.get("weighted_hits", config.get("weighted_flux", True))),
        labeled_exits_only=bool(config.get("labeled_exits_only", False)),
    )
    Jbar_direct_std = _mean_std_from_component_stds(J_thresholds_direct_std, axis=1)
    if bool(config.get("include_threshold_variance_in_error", False)):
        Jbar_direct_std = np.sqrt(Jbar_direct_std * Jbar_direct_std + np.maximum(variance, 0.0))
    J_matrix_direct_std = matrix_from_pair_values(n_states, pairs, Jbar_direct_std, fill=np.nan)
    C_matrix_direct_std = matrix_from_pair_values(n_states, pairs, C_direct_std, fill=np.nan)
    C_abs_matrix_direct_std = matrix_from_pair_values(n_states, pairs, C_abs_direct_std, fill=np.nan)
    k_direct_direct_std = np.zeros((n_states, n_states), dtype=np.float64)
    k_direct_direct_std[:] = np.nan
    for p_idx, (i, j) in enumerate(pairs):
        denom = max(float(pi[i]), 1e-300)
        kij = float(k_direct[i, j])
        k_direct_direct_std[i, j] = np.sqrt(
            (float(Jbar_direct_std[p_idx]) / denom) ** 2 + ((kij * float(pi_direct_std[i])) / denom) ** 2
        )

    error_enabled = bool(config.get("error_analysis", True))
    error_estimator = _resolve_error_estimator(config)
    error_n_slices = int(config.get("error_n_blocks", config.get("error_n_slices", config.get("n_error_slices", 10))))
    error_min_pairs = int(config.get("error_min_pairs_per_block", config.get("error_min_pairs_per_slice", 1)))
    resampled_std: dict[str, np.ndarray] = {}
    n_error_blocks_used = 0
    if error_enabled and error_estimator != "direct" and error_n_slices >= 2 and len(idx0) >= 2:
        common_error_kwargs = {
            "q": q,
            "weights": weights,
            "state": state,
            "idx0": idx0,
            "idx1": idx1,
            "n_states": n_states,
            "pairs": pairs,
            "thresholds": thresholds,
            "tau": tau,
            "divide_by_tau": bool(config.get("divide_by_tau", True)),
            "eps": float(config.get("flux_eps", 0.02)),
            "surface": str(config.get("flux_surface", "qi_decrease")),
            "chunk_size": int(config.get("chunk_size", 20000)),
            "weighted_flux": bool(config.get("weighted_flux", True)),
            "weighted_hits": bool(config.get("weighted_hits", config.get("weighted_flux", True))),
            "labeled_exits_only": bool(config.get("labeled_exits_only", False)),
            "pi_mode": str(config.get("pi_mode", "labels")),
            "time_unit": time_unit,
            "min_pairs": error_min_pairs,
            "device": flux_device,
            "dtype": flux_dtype,
            "kinetic_edge_filter": kinetic_edge_filter_cfg,
        }
        if error_estimator == "block_jackknife":
            resampled_std, n_error_blocks_used = estimate_block_jackknife_rate_std(
                **common_error_kwargs,
                n_blocks=error_n_slices,
            )
        else:
            resampled_std, n_error_blocks_used = estimate_slice_rate_std(
                **common_error_kwargs,
                n_slices=error_n_slices,
            )

    pi_std = _pick_std(pi_direct_std, resampled_std.get("pi"), error_estimator)
    T_hit_std = _pick_std(T_hit_direct_std, resampled_std.get("T_hit"), error_estimator)
    exit_counts_std = _pick_std(exit_counts_direct_std, resampled_std.get("exit_counts"), error_estimator)
    exit_weight_std = _pick_std(exit_weight_direct_std, resampled_std.get("exit_weight"), error_estimator)
    J_thresholds_std = _pick_std(J_thresholds_direct_std, resampled_std.get("J_thresholds"), error_estimator)
    J_matrix_std = _pick_std(J_matrix_direct_std, resampled_std.get("J_matrix"), error_estimator)
    C_matrix_std = _pick_std(C_matrix_direct_std, resampled_std.get("C_matrix"), error_estimator)
    C_abs_matrix_std = _pick_std(C_abs_matrix_direct_std, resampled_std.get("C_abs_matrix"), error_estimator)
    k_direct_std = _pick_std(k_direct_direct_std, resampled_std.get("k_direct"), error_estimator)
    assert pi_std is not None
    assert T_hit_std is not None
    assert exit_counts_std is not None
    assert exit_weight_std is not None
    assert J_thresholds_std is not None
    assert J_matrix_std is not None
    assert C_matrix_std is not None
    assert C_abs_matrix_std is not None
    assert k_direct_std is not None
    k_direct_std_unfiltered = k_direct_std.copy()

    K_unfiltered = assemble_generator(
        k_direct_unfiltered,
        negative_policy=str(config.get("negative_rate_policy", "clip")),
        negative_tol=float(config.get("negative_rate_tolerance", 0.0)),
    )
    P_jump_unfiltered = compute_jump_probabilities(K_unfiltered)
    k_direct, kinetic_edge_removed_mask, kinetic_edge_filter_stats = filter_kinetic_edges(
        k_direct_unfiltered,
        P_jump_unfiltered,
        k_direct_std=k_direct_std_unfiltered,
        config={"kinetic_edge_filter": kinetic_edge_filter_cfg},
    )
    if kinetic_edge_filter_stats["n_removed_edges"] > 0:
        print(
            "[RATE] Removed "
            f"{kinetic_edge_filter_stats['n_removed_edges']} low-probability kinetic edge(s) "
            "before generator/MFPT calculations."
        )
    k_direct_std = k_direct_std_unfiltered.copy()
    k_direct_std[kinetic_edge_removed_mask] = 0.0
    K = assemble_generator(
        k_direct,
        negative_policy=str(config.get("negative_rate_policy", "clip")),
        negative_tol=float(config.get("negative_rate_tolerance", 0.0)),
    )
    P_jump = compute_jump_probabilities(K)
    mfpt = compute_mfpt_matrix(K)
    k_mfpt = compute_mfpt_rate_matrix(mfpt)
    for row_idx, (i, j) in enumerate(pairs):
        table.loc[row_idx, "k_direct_ij"] = float(k_direct[i, j])
        table.loc[row_idx, "k_ij"] = float(k_direct[i, j])
    table = add_mfpt_rates_to_table(table, pairs, k_mfpt)

    propagated_std = propagate_rate_matrix_std(k_direct, k_direct_std)
    K_std = _pick_std(propagated_std["K"], resampled_std.get("K"), error_estimator)
    P_jump_std = _pick_std(propagated_std["P_jump"], resampled_std.get("P_jump"), error_estimator)
    MFPT_std = _pick_std(propagated_std["MFPT"], resampled_std.get("MFPT"), error_estimator)
    k_mfpt_std = _pick_std(propagated_std["k_mfpt"], resampled_std.get("k_mfpt"), error_estimator)
    assert K_std is not None
    assert P_jump_std is not None
    assert MFPT_std is not None
    assert k_mfpt_std is not None

    np.save(os.path.join(out_dir, "Q.npy"), q.astype(np.float32))
    np.save(os.path.join(out_dir, "burn_in_keep_mask.npy"), burn_in_mask)
    np.save(os.path.join(out_dir, "pi.npy"), pi)
    np.save(os.path.join(out_dir, "T_hit.npy"), T_hit)
    np.save(os.path.join(out_dir, "J_thresholds.npy"), J)
    np.save(os.path.join(out_dir, "J_matrix.npy"), J_matrix)
    np.save(os.path.join(out_dir, "C_matrix.npy"), C_matrix)
    np.save(os.path.join(out_dir, "C_abs_matrix.npy"), C_abs_matrix)
    np.save(os.path.join(out_dir, "k_direct_raw.npy"), k_direct_raw)
    np.save(os.path.join(out_dir, "k_direct_unfiltered.npy"), k_direct_unfiltered)
    np.save(os.path.join(out_dir, "k_direct.npy"), k_direct)
    np.save(os.path.join(out_dir, "k_matrix.npy"), k_direct)
    np.save(os.path.join(out_dir, "K_unfiltered.npy"), K_unfiltered)
    np.save(os.path.join(out_dir, "P_jump_unfiltered.npy"), P_jump_unfiltered)
    np.save(os.path.join(out_dir, "K.npy"), K)
    np.save(os.path.join(out_dir, "P_jump.npy"), P_jump)
    np.save(os.path.join(out_dir, "MFPT.npy"), mfpt)
    np.save(os.path.join(out_dir, "k_mfpt.npy"), k_mfpt)
    np.save(os.path.join(out_dir, "pi_std.npy"), pi_std)
    np.save(os.path.join(out_dir, "T_hit_std.npy"), T_hit_std)
    np.save(os.path.join(out_dir, "J_thresholds_std.npy"), J_thresholds_std)
    np.save(os.path.join(out_dir, "J_matrix_std.npy"), J_matrix_std)
    np.save(os.path.join(out_dir, "C_matrix_std.npy"), C_matrix_std)
    np.save(os.path.join(out_dir, "C_abs_matrix_std.npy"), C_abs_matrix_std)
    np.save(os.path.join(out_dir, "k_direct_std_unfiltered.npy"), k_direct_std_unfiltered)
    np.save(os.path.join(out_dir, "k_direct_std.npy"), k_direct_std)
    np.save(os.path.join(out_dir, "k_matrix_std.npy"), k_direct_std)
    np.save(os.path.join(out_dir, "kinetic_edge_removed_mask.npy"), kinetic_edge_removed_mask.astype(np.int8))
    np.save(os.path.join(out_dir, "K_std.npy"), K_std)
    np.save(os.path.join(out_dir, "P_jump_std.npy"), P_jump_std)
    np.save(os.path.join(out_dir, "MFPT_std.npy"), MFPT_std)
    np.save(os.path.join(out_dir, "k_mfpt_std.npy"), k_mfpt_std)
    write_flux_profiles_csv(os.path.join(out_dir, "flux_profiles.csv"), pairs, thresholds, J, variance)
    write_flux_profiles_std_csv(os.path.join(out_dir, "flux_profiles_std.csv"), pairs, thresholds, J_thresholds_std)
    write_population_csv(os.path.join(out_dir, "populations.csv"), pi)
    write_population_std_csv(os.path.join(out_dir, "populations_std.csv"), pi_std)
    write_population_std_csv(os.path.join(out_dir, "pi_std.csv"), pi_std)
    write_transition_hit_csv(os.path.join(out_dir, "transition_hit_matrix.csv"), T_hit, exit_counts, exit_weight)
    write_transition_hit_std_csv(
        os.path.join(out_dir, "transition_hit_matrix_std.csv"), T_hit_std, exit_counts_std, exit_weight_std
    )
    write_matrix_csv(os.path.join(out_dir, "T_hit_std.csv"), T_hit_std)
    table.to_csv(os.path.join(out_dir, "rate_constants.csv"), index=False)
    build_rate_std_table(pairs, pi_std, J_matrix_std, k_direct_std, time_unit, k_mfpt_std=k_mfpt_std).to_csv(
        os.path.join(out_dir, "rate_constants_std.csv"), index=False
    )
    write_matrix_csv(os.path.join(out_dir, "J_matrix.csv"), J_matrix)
    write_matrix_csv(os.path.join(out_dir, "J_matrix_std.csv"), J_matrix_std)
    write_matrix_csv(os.path.join(out_dir, "C_matrix.csv"), C_matrix)
    write_matrix_csv(os.path.join(out_dir, "C_matrix_std.csv"), C_matrix_std)
    write_matrix_csv(os.path.join(out_dir, "C_abs_matrix.csv"), C_abs_matrix)
    write_matrix_csv(os.path.join(out_dir, "C_abs_matrix_std.csv"), C_abs_matrix_std)
    write_matrix_csv(os.path.join(out_dir, "k_direct_raw.csv"), k_direct_raw)
    write_matrix_csv(os.path.join(out_dir, "k_direct_unfiltered.csv"), k_direct_unfiltered)
    write_matrix_csv(os.path.join(out_dir, "k_direct.csv"), k_direct)
    write_matrix_csv(os.path.join(out_dir, "k_direct_std_unfiltered.csv"), k_direct_std_unfiltered)
    write_matrix_csv(os.path.join(out_dir, "k_direct_std.csv"), k_direct_std)
    write_matrix_csv(os.path.join(out_dir, "k_matrix.csv"), k_direct)
    write_matrix_csv(os.path.join(out_dir, "k_matrix_std.csv"), k_direct_std)
    write_matrix_csv(os.path.join(out_dir, "kinetic_edge_removed_mask.csv"), kinetic_edge_removed_mask.astype(np.float64))
    write_matrix_csv(os.path.join(out_dir, "K_unfiltered.csv"), K_unfiltered)
    write_matrix_csv(os.path.join(out_dir, "K.csv"), K)
    write_matrix_csv(os.path.join(out_dir, "K_std.csv"), K_std)
    write_matrix_csv(os.path.join(out_dir, "P_jump_unfiltered.csv"), P_jump_unfiltered)
    write_matrix_csv(os.path.join(out_dir, "P_jump.csv"), P_jump)
    write_matrix_csv(os.path.join(out_dir, "P_jump_std.csv"), P_jump_std)
    write_matrix_csv(os.path.join(out_dir, "MFPT.csv"), mfpt)
    write_matrix_csv(os.path.join(out_dir, "MFPT_std.csv"), MFPT_std)
    write_matrix_csv(os.path.join(out_dir, "k_mfpt.csv"), k_mfpt)
    write_matrix_csv(os.path.join(out_dir, "k_mfpt_std.csv"), k_mfpt_std)
    plot_paths: list[str] = []
    if bool(config.get("make_plots", True)):
        plot_paths = plot_rate_diagnostics(
            out_dir=out_dir,
            pi=pi,
            T_hit=T_hit,
            J_matrix=J_matrix,
            k_direct=k_direct,
            K=K,
            P_jump=P_jump,
            mfpt=mfpt,
            pairs=pairs,
            thresholds=thresholds,
            J_thresholds=J,
            fmt=str(config.get("plot_format", "png")),
        )

    summary = {
        "dataset": os.path.abspath(str(dataset_path)),
        "out_dir": os.path.abspath(out_dir),
        "n_states": int(n_states),
        "lag": int(timing["lag"]),
        "lag_reference": str(timing["lag_reference"]),
        "lag_index_step_after_stride": int(timing["lag_index_step"]),
        "lag_original_frames": int(timing["lag_original_frames"]),
        "dataset_stride": int(timing["dataset_stride"]),
        "frame_time": timing["frame_time"],
        "tau": float(tau),
        "time_unit": time_unit,
        "flux_device": str(flux_device),
        "flux_dtype": str(flux_dtype).replace("torch.", ""),
        "divide_by_tau": bool(config.get("divide_by_tau", True)),
        "pi_mode": str(config.get("pi_mode", "labels")),
        "stationary_or_population_pi": [float(x) for x in pi],
        "exit_counts": [int(x) for x in exit_counts],
        "ordered_pairs": [[int(i), int(j)] for i, j in pairs],
        "thresholds": [float(x) for x in thresholds],
        "model_input": model_input_summary,
        "probability_checks": checks,
        "rate_sanitize": rate_sanitize,
        "kinetic_edge_filter": kinetic_edge_filter_stats,
        "n_lagged_pairs": int(len(idx0)),
        "trajectory_burn_in": burn_in_stats,
        "zero_weight_mask": zero_weight_mask_stats,
        "error_analysis": {
            "enabled": bool(error_enabled),
            "method": str(error_estimator),
            "include_threshold_variance_in_error": bool(config.get("include_threshold_variance_in_error", False)),
            "requested_blocks": int(error_n_slices),
            "used_blocks": int(n_error_blocks_used),
            "min_pairs_per_block": int(error_min_pairs),
        },
        "rate_outputs": {
            "burn_in_keep_mask": os.path.abspath(os.path.join(out_dir, "burn_in_keep_mask.npy")),
            "T_hit": os.path.abspath(os.path.join(out_dir, "T_hit.npy")),
            "k_direct_raw": os.path.abspath(os.path.join(out_dir, "k_direct_raw.npy")),
            "k_direct_unfiltered": os.path.abspath(os.path.join(out_dir, "k_direct_unfiltered.npy")),
            "k_direct": os.path.abspath(os.path.join(out_dir, "k_direct.npy")),
            "K_unfiltered": os.path.abspath(os.path.join(out_dir, "K_unfiltered.npy")),
            "P_jump_unfiltered": os.path.abspath(os.path.join(out_dir, "P_jump_unfiltered.npy")),
            "K": os.path.abspath(os.path.join(out_dir, "K.npy")),
            "P_jump": os.path.abspath(os.path.join(out_dir, "P_jump.npy")),
            "MFPT": os.path.abspath(os.path.join(out_dir, "MFPT.npy")),
            "k_mfpt": os.path.abspath(os.path.join(out_dir, "k_mfpt.npy")),
            "kinetic_edge_removed_mask_csv": os.path.abspath(os.path.join(out_dir, "kinetic_edge_removed_mask.csv")),
            "k_direct_std_csv": os.path.abspath(os.path.join(out_dir, "k_direct_std.csv")),
            "P_jump_std_csv": os.path.abspath(os.path.join(out_dir, "P_jump_std.csv")),
            "MFPT_std_csv": os.path.abspath(os.path.join(out_dir, "MFPT_std.csv")),
            "k_mfpt_std_csv": os.path.abspath(os.path.join(out_dir, "k_mfpt_std.csv")),
        },
        "diagnostic_plots": [os.path.abspath(path) for path in plot_paths],
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    print(f"[RATE] Saved flux profiles and k_ij table to {out_dir}")
    print(f"[CHECK] max |sum(q)-1| = {checks['max_sum_error']:.3e}, min(q) = {checks['min_q']:.3e}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate k_ij from native next-hit committor fluxes.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "NEXT_HIT_RATE", "RATE_CONSTANT")
    run(cfg)


if __name__ == "__main__":
    main()

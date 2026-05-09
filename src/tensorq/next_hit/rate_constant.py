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
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        else:
            numer += torch.sum(samples, dim=0)
            c_numer += torch.sum(C, dim=0)
            c_abs_numer += torch.sum(torch.abs(C), dim=0)
            denom += int(end - start)

    denom = torch.clamp(denom, min=torch.finfo(dtype).tiny)
    J_t = numer / denom
    if divide_by_tau:
        J_t = J_t / float(tau)
    variance_t = torch.mean((J_t - J_t.mean(dim=1, keepdim=True)) ** 2, dim=1)
    J = J_t.detach().cpu().numpy().astype(np.float64)
    variance = variance_t.detach().cpu().numpy().astype(np.float64)
    if return_current:
        C_mean = (c_numer / denom).detach().cpu().numpy().astype(np.float64)
        C_abs_mean = (c_abs_numer / denom).detach().cpu().numpy().astype(np.float64)
        return J, variance, C_mean, C_abs_mean
    return J, variance


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


def assemble_generator(k_direct: np.ndarray) -> np.ndarray:
    k_direct = np.asarray(k_direct, dtype=np.float64)
    if k_direct.ndim != 2 or k_direct.shape[0] != k_direct.shape[1]:
        raise ValueError(f"k_direct must be a square matrix; got {k_direct.shape}.")
    K = k_direct.copy()
    np.fill_diagonal(K, 0.0)
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


def write_matrix_csv(path: str, matrix: np.ndarray, *, index_name: str = "state_i") -> None:
    pd.DataFrame(np.asarray(matrix, dtype=np.float64)).to_csv(path, index_label=index_name)


def write_population_csv(path: str, pi: np.ndarray) -> None:
    rows = [{"state": int(i), "pi": float(pi_i)} for i, pi_i in enumerate(np.asarray(pi, dtype=np.float64))]
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


def run(config: dict[str, Any]) -> dict[str, Any]:
    config, model_input_summary = apply_checkpoint_model_input_config(config)
    out_dir = ensure_dir(config.get("out_dir", "./next_hit_rate"))
    dataset_path = config.get("dataset", config.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("RATE_CONSTANT config needs 'dataset' or 'dataset_path'.")

    dataset_stride = int(config.get("dataset_stride", 1))
    raw_pack = load_dataset(dataset_path)
    pack = apply_stride(raw_pack, dataset_stride)
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
    J, variance, C_mean, C_abs_mean = estimate_flux_profiles(
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
        device=flux_device,
        dtype=flux_dtype,
    )
    pi = estimate_pi(
        q=q,
        weights=weights,
        state=state,
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
    C_matrix = matrix_from_pair_values(n_states, pairs, C_mean, fill=0.0)
    C_abs_matrix = matrix_from_pair_values(n_states, pairs, C_abs_mean, fill=0.0)
    K = assemble_generator(k_direct)
    P_jump = compute_jump_probabilities(K)
    mfpt = compute_mfpt_matrix(K)
    k_mfpt = compute_mfpt_rate_matrix(mfpt)

    np.save(os.path.join(out_dir, "Q.npy"), q.astype(np.float32))
    np.save(os.path.join(out_dir, "pi.npy"), pi)
    np.save(os.path.join(out_dir, "T_hit.npy"), T_hit)
    np.save(os.path.join(out_dir, "J_thresholds.npy"), J)
    np.save(os.path.join(out_dir, "J_matrix.npy"), J_matrix)
    np.save(os.path.join(out_dir, "C_matrix.npy"), C_matrix)
    np.save(os.path.join(out_dir, "C_abs_matrix.npy"), C_abs_matrix)
    np.save(os.path.join(out_dir, "k_direct.npy"), k_direct)
    np.save(os.path.join(out_dir, "k_matrix.npy"), k_direct)
    np.save(os.path.join(out_dir, "K.npy"), K)
    np.save(os.path.join(out_dir, "P_jump.npy"), P_jump)
    np.save(os.path.join(out_dir, "MFPT.npy"), mfpt)
    np.save(os.path.join(out_dir, "k_mfpt.npy"), k_mfpt)
    write_flux_profiles_csv(os.path.join(out_dir, "flux_profiles.csv"), pairs, thresholds, J, variance)
    write_population_csv(os.path.join(out_dir, "populations.csv"), pi)
    write_transition_hit_csv(os.path.join(out_dir, "transition_hit_matrix.csv"), T_hit, exit_counts, exit_weight)
    table.to_csv(os.path.join(out_dir, "rate_constants.csv"), index=False)
    write_matrix_csv(os.path.join(out_dir, "J_matrix.csv"), J_matrix)
    write_matrix_csv(os.path.join(out_dir, "C_matrix.csv"), C_matrix)
    write_matrix_csv(os.path.join(out_dir, "C_abs_matrix.csv"), C_abs_matrix)
    write_matrix_csv(os.path.join(out_dir, "k_direct.csv"), k_direct)
    write_matrix_csv(os.path.join(out_dir, "k_matrix.csv"), k_direct)
    write_matrix_csv(os.path.join(out_dir, "K.csv"), K)
    write_matrix_csv(os.path.join(out_dir, "P_jump.csv"), P_jump)
    write_matrix_csv(os.path.join(out_dir, "MFPT.csv"), mfpt)
    write_matrix_csv(os.path.join(out_dir, "k_mfpt.csv"), k_mfpt)
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
        "n_lagged_pairs": int(len(idx0)),
        "rate_outputs": {
            "T_hit": os.path.abspath(os.path.join(out_dir, "T_hit.npy")),
            "k_direct": os.path.abspath(os.path.join(out_dir, "k_direct.npy")),
            "K": os.path.abspath(os.path.join(out_dir, "K.npy")),
            "P_jump": os.path.abspath(os.path.join(out_dir, "P_jump.npy")),
            "MFPT": os.path.abspath(os.path.join(out_dir, "MFPT.npy")),
            "k_mfpt": os.path.abspath(os.path.join(out_dir, "k_mfpt.npy")),
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

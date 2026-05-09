from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, build_lagged_indices, infer_n_states, load_dataset, select_model_inputs, unordered_pairs
from ..common.flux import make_thresholds, resolve_ordered_pairs
from .predict import apply_checkpoint_input_config, infer_pairwise, load_pairwise_committor_model, probability_checks, reconstruct_state_probabilities


def resolve_lag_timing(config: dict[str, Any], dataset_stride: int) -> dict[str, Any]:
    lag = int(config.get("lag", config.get("time_shift", 1)))
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    reference = str(config.get("lag_reference", "original_frames")).lower()
    if reference in {"original", "original_frame", "original_frames", "saved", "saved_frames"}:
        if lag % int(dataset_stride) != 0:
            raise ValueError(f"lag={lag} is not divisible by dataset_stride={dataset_stride}.")
        lag_index_step = lag // int(dataset_stride)
        lag_original_frames = lag
        lag_reference = "original_frames"
    elif reference in {"strided", "strided_frame", "strided_frames", "post_stride"}:
        lag_index_step = lag
        lag_original_frames = lag * int(dataset_stride)
        lag_reference = "strided_frames"
    else:
        raise ValueError("lag_reference must be 'original_frames' or 'strided_frames'.")
    frame_time = config.get("frame_time", None)
    tau = float(lag_original_frames) if frame_time is None else float(lag_original_frames) * float(frame_time)
    return {
        "lag": lag,
        "lag_reference": lag_reference,
        "lag_index_step": int(lag_index_step),
        "lag_original_frames": int(lag_original_frames),
        "dataset_stride": int(dataset_stride),
        "frame_time": None if frame_time is None else float(frame_time),
        "tau": float(tau),
        "time_unit": str(config.get("time_unit", "frame" if frame_time is None else "time")),
    }


def load_or_infer_probabilities(config: dict[str, Any], pack, n_states: int) -> tuple[np.ndarray, np.ndarray | None]:
    p_path = config.get("P_npy", config.get("p_npy", None))
    if p_path is not None:
        P = np.load(p_path).astype(np.float32)
        return P, None
    q_path = config.get("Q_npy", config.get("q_npy", None))
    if q_path is not None:
        Q = np.load(q_path).astype(np.float32)
    else:
        model_path = config.get("model", None)
        if model_path is None:
            raise KeyError("Provide RATE_CONSTANT.P_npy, RATE_CONSTANT.Q_npy, or RATE_CONSTANT.model.")
        device = setup_device(config.get("device", "cuda:0"))
        model = load_pairwise_committor_model(model_path, device)
        features, _ = select_model_inputs(pack, config)
        Q = infer_pairwise(model, features.float(), device, batch_size=int(config.get("batch_size", 65536)))
    P = reconstruct_state_probabilities(
        Q,
        n_states,
        anchor_state=int(config.get("anchor_state", 0)),
        eps=float(config.get("eps", 1e-4)),
        chunk_size=int(config.get("reconstruct_chunk", 20000)),
    )
    return P, Q


def validate_inputs(P: np.ndarray, weights: np.ndarray, state: np.ndarray, idx0: np.ndarray, idx1: np.ndarray, n_states: int) -> None:
    if P.ndim != 2 or P.shape[1] != int(n_states):
        raise ValueError(f"P must have shape (n_frames, {n_states}); got {P.shape}.")
    if weights.shape[0] != P.shape[0] or state.shape[0] != P.shape[0]:
        raise ValueError("weights, state, and P must have the same frame count.")
    if idx0.size == 0 or idx0.shape != idx1.shape:
        raise ValueError("Lagged index arrays must be non-empty and matching.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")


def estimate_flux_profiles(
    P: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_i = np.asarray([p[0] for p in pairs], dtype=np.int64)
    pair_j = np.asarray([p[1] for p in pairs], dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(1, 1, -1)
    numer = np.zeros((len(pairs), thresholds.shape[2]), dtype=np.float64)
    c_numer = np.zeros(len(pairs), dtype=np.float64)
    c_abs_numer = np.zeros(len(pairs), dtype=np.float64)
    denom = 0.0
    for start in range(0, idx0.shape[0], int(chunk_size)):
        end = min(idx0.shape[0], start + int(chunk_size))
        p0 = P[idx0[start:end]].astype(np.float64)
        pt = P[idx1[start:end]].astype(np.float64)
        C = pt[:, pair_j] * p0[:, pair_i] - p0[:, pair_j] * pt[:, pair_i]
        if str(surface).lower() == "qi_decrease":
            chi = 1.0 / (1.0 + np.exp(-((p0[:, pair_i, None] - thresholds) / float(eps))))
            chi *= 1.0 / (1.0 + np.exp(-((thresholds - pt[:, pair_i, None]) / float(eps))))
        elif str(surface).lower() == "qj_increase":
            chi = 1.0 / (1.0 + np.exp(-((thresholds - p0[:, pair_j, None]) / float(eps))))
            chi *= 1.0 / (1.0 + np.exp(-((pt[:, pair_j, None] - thresholds) / float(eps))))
        else:
            raise ValueError("flux_surface must be 'qi_decrease' or 'qj_increase'.")
        samples = chi * C[:, :, None]
        if weighted:
            w = weights[idx0[start:end]].astype(np.float64)
            numer += np.sum(samples * w[:, None, None], axis=0)
            c_numer += np.sum(C * w[:, None], axis=0)
            c_abs_numer += np.sum(np.abs(C) * w[:, None], axis=0)
            denom += float(np.sum(w))
        else:
            numer += np.sum(samples, axis=0)
            c_numer += np.sum(C, axis=0)
            c_abs_numer += np.sum(np.abs(C), axis=0)
            denom += float(end - start)
    J = numer / max(denom, 1e-300)
    if divide_by_tau:
        J = J / float(tau)
    variance = np.mean((J - J.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return J, variance, c_numer / max(denom, 1e-300), c_abs_numer / max(denom, 1e-300)


def estimate_pi(P: np.ndarray, weights: np.ndarray, state: np.ndarray, n_states: int, mode: str) -> np.ndarray:
    w = weights.astype(np.float64)
    w = w / (np.sum(w) + 1e-300)
    if str(mode).lower() == "soft":
        pi = np.sum(P.astype(np.float64) * w[:, None], axis=0)
    elif str(mode).lower() == "labels":
        pi = np.asarray([np.sum(w[state == k]) for k in range(int(n_states))], dtype=np.float64)
        if np.sum(pi) <= 0:
            pi = np.sum(P.astype(np.float64) * w[:, None], axis=0)
    else:
        raise ValueError("pi_mode must be 'labels' or 'soft'.")
    return pi / (np.sum(pi) + 1e-300)


def assemble_generator(k_direct: np.ndarray) -> np.ndarray:
    K = np.asarray(k_direct, dtype=np.float64).copy()
    np.fill_diagonal(K, 0.0)
    for i in range(K.shape[0]):
        K[i, i] = -float(np.sum(K[i]))
    return K


def compute_mfpt_matrix(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for target in range(n):
        keep = np.asarray([i for i in range(n) if i != target], dtype=np.int64)
        try:
            out[keep, target] = np.linalg.solve(K[np.ix_(keep, keep)], -np.ones(n - 1))
        except np.linalg.LinAlgError:
            out[keep, target] = np.nan
    return out


def write_matrix_csv(path: str, matrix: np.ndarray) -> None:
    pd.DataFrame(np.asarray(matrix, dtype=np.float64)).to_csv(path, index_label="state_i")


def plot_heatmap(matrix: np.ndarray, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=160)
    im = ax.imshow(matrix, interpolation="nearest", aspect="auto")
    ax.set_xlabel("state j")
    ax.set_ylabel("state i")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(config: dict[str, Any]) -> dict[str, Any]:
    config, input_source = apply_checkpoint_input_config(config)
    out_dir = ensure_dir(config.get("out_dir", "./pairwise_committor_rate"))
    dataset_path = config.get("dataset", config.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("RATE_CONSTANT config needs 'dataset' or 'dataset_path'.")
    dataset_stride = int(config.get("dataset_stride", 1))
    pack = apply_stride(load_dataset(dataset_path), dataset_stride)
    n_states = infer_n_states(pack, config.get("n_states", None))
    P, Q = load_or_infer_probabilities(config, pack, n_states)
    if P.shape[0] != pack.features.shape[0]:
        raise RuntimeError(f"P has {P.shape[0]} frames but dataset has {pack.features.shape[0]} frames.")

    timing = resolve_lag_timing(config, dataset_stride)
    idx0_t, idx1_t = build_lagged_indices(P.shape[0], int(timing["lag_index_step"]), pack.traj_id, bool(config.get("allow_cross_traj_pairs", False)))
    idx0, idx1 = idx0_t.numpy(), idx1_t.numpy()
    weights = pack.weights.numpy().astype(np.float64)
    state = pack.state.numpy().astype(np.int64)
    validate_inputs(P, weights, state, idx0, idx1, n_states)

    thresholds = make_thresholds(config.get("thresholds", None), int(config.get("n_thresholds", 9)), float(config.get("threshold_start", 0.1)), float(config.get("threshold_stop", 0.9))).numpy()
    pairs = resolve_ordered_pairs(n_states, config.get("adjacency", config.get("flux_pairs", None)), bool(config.get("symmetric_adjacency", True)))
    J, variance, C_mean, C_abs_mean = estimate_flux_profiles(
        P,
        weights,
        idx0,
        idx1,
        pairs,
        thresholds,
        eps=float(config.get("flux_eps", 0.02)),
        tau=float(timing["tau"]),
        divide_by_tau=bool(config.get("divide_by_tau", True)),
        surface=str(config.get("flux_surface", "qi_decrease")),
        chunk_size=int(config.get("chunk_size", 20000)),
        weighted=bool(config.get("weighted_flux", True)),
    )
    pi = estimate_pi(P, weights, state, n_states, str(config.get("pi_mode", "labels")))
    Jbar = J.mean(axis=1)
    J_matrix = np.zeros((n_states, n_states), dtype=np.float64)
    C_matrix = np.zeros((n_states, n_states), dtype=np.float64)
    C_abs_matrix = np.zeros((n_states, n_states), dtype=np.float64)
    k_direct = np.zeros((n_states, n_states), dtype=np.float64)
    rows = []
    for p_idx, (i, j) in enumerate(pairs):
        J_matrix[i, j] = Jbar[p_idx]
        C_matrix[i, j] = C_mean[p_idx]
        C_abs_matrix[i, j] = C_abs_mean[p_idx]
        k_direct[i, j] = Jbar[p_idx] / max(float(pi[i]), 1e-300)
        rows.append({"state_i": i, "state_j": j, "pi_i": float(pi[i]), "J_ij": float(Jbar[p_idx]), "J_threshold_variance": float(variance[p_idx]), "k_ij": float(k_direct[i, j]), "k_unit": f"1/{timing['time_unit']}"})
    K = assemble_generator(k_direct)
    mfpt = compute_mfpt_matrix(K)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_mfpt = 1.0 / mfpt
    np.fill_diagonal(k_mfpt, 0.0)

    np.save(os.path.join(out_dir, "P.npy"), P.astype(np.float32))
    if Q is not None:
        np.save(os.path.join(out_dir, "Q.npy"), Q.astype(np.float32))
    np.save(os.path.join(out_dir, "pi.npy"), pi)
    np.save(os.path.join(out_dir, "J_thresholds.npy"), J)
    np.save(os.path.join(out_dir, "J_matrix.npy"), J_matrix)
    np.save(os.path.join(out_dir, "C_matrix.npy"), C_matrix)
    np.save(os.path.join(out_dir, "C_abs_matrix.npy"), C_abs_matrix)
    np.save(os.path.join(out_dir, "k_direct.npy"), k_direct)
    np.save(os.path.join(out_dir, "K.npy"), K)
    np.save(os.path.join(out_dir, "MFPT.npy"), mfpt)
    np.save(os.path.join(out_dir, "k_mfpt.npy"), k_mfpt)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "rate_constants.csv"), index=False)
    pd.DataFrame([{"state_i": i, "state_j": j, "threshold": float(c), "J_ij": float(J[p_idx, t_idx]), "threshold_variance": float(variance[p_idx])} for p_idx, (i, j) in enumerate(pairs) for t_idx, c in enumerate(thresholds)]).to_csv(os.path.join(out_dir, "flux_profiles.csv"), index=False)
    for name, matrix in {"J_matrix": J_matrix, "C_matrix": C_matrix, "C_abs_matrix": C_abs_matrix, "k_direct": k_direct, "K": K, "MFPT": mfpt, "k_mfpt": k_mfpt}.items():
        write_matrix_csv(os.path.join(out_dir, f"{name}.csv"), matrix)
    plot_paths: list[str] = []
    if bool(config.get("make_plots", True)):
        plot_dir = ensure_dir(os.path.join(out_dir, "diagnostics"))
        for name, matrix in {"J_matrix": J_matrix, "k_direct": k_direct, "K": K, "MFPT": mfpt}.items():
            path = os.path.join(plot_dir, f"{name}.{config.get('plot_format', 'png')}")
            plot_heatmap(matrix, path, name)
            plot_paths.append(path)

    summary = {
        "dataset": os.path.abspath(str(dataset_path)),
        "out_dir": os.path.abspath(out_dir),
        "n_states": int(n_states),
        "pairs": [[int(i), int(j)] for i, j in unordered_pairs(n_states)],
        "ordered_flux_pairs": [[int(i), int(j)] for i, j in pairs],
        "lag": int(timing["lag"]),
        "lag_reference": timing["lag_reference"],
        "lag_index_step_after_stride": int(timing["lag_index_step"]),
        "tau": float(timing["tau"]),
        "time_unit": timing["time_unit"],
        "model_input": input_source,
        "probability_checks": probability_checks(P, float(config.get("normalization_atol", 1e-4))),
        "rate_constants_csv": os.path.abspath(os.path.join(out_dir, "rate_constants.csv")),
        "diagnostic_plots": [os.path.abspath(path) for path in plot_paths],
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    print(f"[RATE] Saved pair-wise committor rate constants to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate rate constants from pair-wise committors.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "PAIRWISE_RATE", "RATE_CONSTANT")
    run(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, build_lagged_indices, infer_n_states, load_dataset
from ..common.flux import make_thresholds, resolve_ordered_pairs
from .predict import check_probability_rows
from .rate_constant import (
    apply_checkpoint_model_input_config,
    build_rate_table,
    compute_jump_probabilities,
    compute_mfpt_matrix,
    compute_mfpt_rate_matrix,
    estimate_flux_profiles,
    estimate_pi,
    estimate_transition_hit_matrix,
    load_or_infer_q,
    matrix_from_pair_values,
    plot_matrix_heatmap,
    positive_weight_masks,
    resolve_flux_dtype,
    resolve_lag_timing,
    validate_rate_inputs,
    write_flux_profiles_csv,
    write_matrix_csv,
    write_population_csv,
    write_transition_hit_csv,
)


def offdiag_mask(n_states: int) -> np.ndarray:
    mask = np.ones((int(n_states), int(n_states)), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def assemble_generator_from_offdiag(rates: np.ndarray) -> np.ndarray:
    K = np.asarray(rates, dtype=np.float64).copy()
    K[~np.isfinite(K)] = 0.0
    K = np.maximum(K, 0.0)
    np.fill_diagonal(K, 0.0)
    for i in range(K.shape[0]):
        K[i, i] = -float(np.sum(K[i]))
    return K


def normalize_offdiag_rows(values: np.ndarray) -> np.ndarray:
    mat = np.asarray(values, dtype=np.float64).copy()
    mat[~np.isfinite(mat)] = 0.0
    mat = np.maximum(mat, 0.0)
    np.fill_diagonal(mat, 0.0)
    row_sum = np.sum(mat, axis=1)
    np.divide(mat, row_sum[:, None], out=mat, where=row_sum[:, None] > 0.0)
    return mat


def branch_matrix_from_t_hit(T_hit: np.ndarray, mode: str = "drop_self") -> np.ndarray:
    """
    Convert endpoint next-hit averages into an off-diagonal branching matrix.

    drop_self renormalizes over j != i. keep_self_mass first normalizes the full
    row, then removes the diagonal, so self-return probability reduces the
    committed off-diagonal mass.
    """
    mode = str(mode).lower()
    raw = np.asarray(T_hit, dtype=np.float64).copy()
    raw[~np.isfinite(raw)] = 0.0
    raw = np.maximum(raw, 0.0)

    if mode in {"drop_self", "renormalize_offdiag", "offdiag"}:
        return normalize_offdiag_rows(raw)

    if mode in {"keep_self_mass", "keep_self", "self_mass"}:
        row_sum = np.sum(raw, axis=1)
        np.divide(raw, row_sum[:, None], out=raw, where=row_sum[:, None] > 0.0)
        np.fill_diagonal(raw, 0.0)
        return raw

    if mode in {"raw_offdiag", "raw"}:
        np.fill_diagonal(raw, 0.0)
        return raw

    raise ValueError("fit_branch_mode must be 'drop_self', 'keep_self_mass', or 'raw_offdiag'.")


def stationary_distribution(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    A = K.T.copy()
    rhs = np.zeros(n, dtype=np.float64)
    A[-1] = 1.0
    rhs[-1] = 1.0
    try:
        pi = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eig(K.T)
        idx = int(np.argmin(np.abs(vals)))
        pi = np.real(vecs[:, idx])
    pi = np.where(np.isfinite(pi), pi, 0.0)
    if np.sum(pi) < 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    total = float(np.sum(pi))
    if total <= 0.0:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    return pi / total


def detailed_balance_projection(rates: np.ndarray, pi: np.ndarray) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    flux = pi[:, None] * np.maximum(np.where(np.isfinite(rates), rates, 0.0), 0.0)
    sym_flux = 0.5 * (flux + flux.T)
    projected = sym_flux / np.maximum(pi[:, None], 1e-300)
    np.fill_diagonal(projected, 0.0)
    return projected


def rmse(values: np.ndarray, mask: np.ndarray | None = None, scale: float | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not np.any(finite):
        return float("nan")
    diff = arr[finite]
    val = float(np.sqrt(np.mean(diff * diff)))
    if scale is None:
        return val
    return val / max(float(scale), 1e-300)


def pair_mask_matrix(n_states: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((int(n_states), int(n_states)), dtype=bool)
    for i, j in pairs:
        mask[int(i), int(j)] = True
    np.fill_diagonal(mask, False)
    return mask


def fit_generator_scf(
    k_flux: np.ndarray,
    J_obs: np.ndarray,
    pi_obs: np.ndarray,
    P_branch_obs: np.ndarray,
    pairs: list[tuple[int, int]],
    config: dict[str, Any],
) -> dict[str, Any]:
    n_states = int(k_flux.shape[0])
    offdiag = offdiag_mask(n_states)
    pair_mask = pair_mask_matrix(n_states, pairs)
    branch_row_has_info = np.sum(np.asarray(P_branch_obs, dtype=np.float64), axis=1) > 0.0
    branch_mask = branch_row_has_info[:, None] & offdiag

    flux_weight = float(config.get("fit_flux_weight", 1.0))
    branch_weight = float(config.get("fit_branch_weight", 1.0))
    db_blend = float(config.get("fit_detailed_balance_blend", 0.0))
    mixing = float(config.get("fit_mixing", 0.5))
    max_iter = int(config.get("fit_max_iter", 500))
    tol = float(config.get("fit_tol", 1e-10))
    min_rate = float(config.get("fit_min_rate", 0.0))

    K_flux = np.asarray(k_flux, dtype=np.float64).copy()
    K_flux[~np.isfinite(K_flux)] = 0.0
    K_flux = np.maximum(K_flux, 0.0)
    np.fill_diagonal(K_flux, 0.0)

    init = str(config.get("fit_init", "flux")).lower()
    if init in {"branch", "branching"}:
        row_rate = np.sum(K_flux, axis=1)
        K = row_rate[:, None] * np.asarray(P_branch_obs, dtype=np.float64)
    elif init in {"hybrid", "mixed"}:
        row_rate = np.sum(K_flux, axis=1)
        K = 0.5 * K_flux + 0.5 * row_rate[:, None] * np.asarray(P_branch_obs, dtype=np.float64)
    else:
        K = K_flux.copy()
    K[~np.isfinite(K)] = 0.0
    K = np.maximum(K, 0.0)
    np.fill_diagonal(K, 0.0)

    history: list[dict[str, float]] = []
    J_scale = float(np.nanmax(np.abs(J_obs[pair_mask]))) if np.any(pair_mask) else 1.0

    for iteration in range(max_iter + 1):
        row_rate = np.sum(K, axis=1)
        P_fit = normalize_offdiag_rows(K)
        J_fit = pi_obs[:, None] * K
        K_gen = assemble_generator_from_offdiag(K)
        pi_fit = stationary_distribution(K_gen)
        db_flux = pi_obs[:, None] * K - (pi_obs[:, None] * K).T

        metrics = {
            "iteration": float(iteration),
            "flux_rmse": rmse(J_fit - J_obs, pair_mask, J_scale),
            "branch_rmse": rmse(P_fit - P_branch_obs, branch_mask, 1.0),
            "pi_rmse": rmse(pi_fit - pi_obs, None, 1.0),
            "detailed_balance_rmse": rmse(db_flux, offdiag, J_scale),
            "max_exit_rate": float(np.max(row_rate)) if row_rate.size else 0.0,
        }
        history.append(metrics)
        if iteration >= max_iter:
            break

        K_branch = row_rate[:, None] * np.asarray(P_branch_obs, dtype=np.float64)
        target = np.zeros_like(K)
        denom = np.zeros_like(K)

        if flux_weight > 0.0:
            target[pair_mask] += flux_weight * K_flux[pair_mask]
            denom[pair_mask] += flux_weight
        if branch_weight > 0.0:
            target[branch_mask] += branch_weight * K_branch[branch_mask]
            denom[branch_mask] += branch_weight

        K_candidate = np.where(denom > 0.0, target / np.maximum(denom, 1e-300), K)
        K_candidate[~np.isfinite(K_candidate)] = 0.0
        K_candidate = np.maximum(K_candidate, 0.0)
        np.fill_diagonal(K_candidate, 0.0)
        if min_rate > 0.0:
            constrained = (pair_mask | branch_mask) & offdiag
            K_candidate[constrained] = np.maximum(K_candidate[constrained], min_rate)

        if db_blend > 0.0:
            K_db = detailed_balance_projection(K_candidate, pi_obs)
            K_candidate = (1.0 - db_blend) * K_candidate + db_blend * K_db
            np.fill_diagonal(K_candidate, 0.0)

        K_next = (1.0 - mixing) * K + mixing * K_candidate
        K_next[~np.isfinite(K_next)] = 0.0
        K_next = np.maximum(K_next, 0.0)
        np.fill_diagonal(K_next, 0.0)

        delta = float(np.linalg.norm(K_next - K) / max(np.linalg.norm(K), 1e-300))
        history[-1]["relative_update"] = delta
        K = K_next
        if delta < tol:
            break

    K_gen = assemble_generator_from_offdiag(K)
    P_fit = normalize_offdiag_rows(K)
    J_fit = pi_obs[:, None] * K
    pi_fit = stationary_distribution(K_gen)
    mfpt = compute_mfpt_matrix(K_gen)
    k_eff = compute_mfpt_rate_matrix(mfpt)
    return {
        "K_fit": K_gen,
        "rates_fit": K,
        "P_fit": P_fit,
        "J_fit": J_fit,
        "pi_fit": pi_fit,
        "MFPT_fit": mfpt,
        "k_eff_fit": k_eff,
        "J_residual": J_fit - J_obs,
        "P_residual": P_fit - P_branch_obs,
        "pi_residual": pi_fit - pi_obs,
        "exit_rate_fit": np.sum(K, axis=1),
        "history": history,
        "settings": {
            "fit_flux_weight": flux_weight,
            "fit_branch_weight": branch_weight,
            "fit_detailed_balance_blend": db_blend,
            "fit_mixing": mixing,
            "fit_max_iter": max_iter,
            "fit_tol": tol,
            "fit_min_rate": min_rate,
            "fit_init": init,
        },
    }


def write_vector_csv(path: str, name: str, values: np.ndarray) -> None:
    rows = [{"state": int(i), name: float(x) if np.isfinite(x) else np.nan} for i, x in enumerate(values)]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_fit_table(
    path: str,
    pairs: list[tuple[int, int]],
    pi_obs: np.ndarray,
    J_obs: np.ndarray,
    k_flux: np.ndarray,
    P_branch_obs: np.ndarray,
    rates_fit: np.ndarray,
    P_fit: np.ndarray,
    J_fit: np.ndarray,
    MFPT_fit: np.ndarray,
    k_eff_fit: np.ndarray,
    time_unit: str,
) -> None:
    rows = []
    for i, j in pairs:
        rows.append(
            {
                "state_i": int(i),
                "state_j": int(j),
                "pi_i": float(pi_obs[i]),
                "J_obs_ij": float(J_obs[i, j]) if np.isfinite(J_obs[i, j]) else np.nan,
                "k_flux_ij": float(k_flux[i, j]) if np.isfinite(k_flux[i, j]) else np.nan,
                "P_branch_obs_ij": float(P_branch_obs[i, j]) if np.isfinite(P_branch_obs[i, j]) else np.nan,
                "K_fit_ij": float(rates_fit[i, j]) if np.isfinite(rates_fit[i, j]) else np.nan,
                "P_fit_ij": float(P_fit[i, j]) if np.isfinite(P_fit[i, j]) else np.nan,
                "J_fit_ij": float(J_fit[i, j]) if np.isfinite(J_fit[i, j]) else np.nan,
                "MFPT_fit_ij": float(MFPT_fit[i, j]) if np.isfinite(MFPT_fit[i, j]) else np.nan,
                "k_eff_fit_ij": float(k_eff_fit[i, j]) if np.isfinite(k_eff_fit[i, j]) else np.nan,
                "k_unit": f"1/{time_unit}",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_vector_compare(obs: np.ndarray, fit: np.ndarray, out_path: str, title: str) -> None:
    x = np.arange(obs.shape[0])
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=160)
    ax.bar(x - width / 2, obs, width=width, label="observed")
    ax.bar(x + width / 2, fit, width=width, label="fit")
    ax.set_xlabel("state")
    ax.set_ylabel("population")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_convergence(history: list[dict[str, float]], out_path: str) -> None:
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=160)
    for name in ("flux_rmse", "branch_rmse", "pi_rmse", "detailed_balance_rmse"):
        if name in df:
            ax.plot(df["iteration"], df[name], label=name)
    ax.set_xlabel("SCF iteration")
    ax.set_ylabel("normalized RMSE")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scatter(obs: np.ndarray, fit: np.ndarray, mask: np.ndarray, out_path: str, title: str, xlabel: str, ylabel: str) -> None:
    use = np.asarray(mask, dtype=bool) & np.isfinite(obs) & np.isfinite(fit)
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=160)
    ax.scatter(obs[use], fit[use], s=20, alpha=0.8)
    if np.any(use):
        lo = float(min(np.min(obs[use]), np.min(fit[use])))
        hi = float(max(np.max(obs[use]), np.max(fit[use])))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_matrix_outputs(out_dir: str, arrays: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), value)
        if np.asarray(value).ndim == 2 and name not in {"Q"}:
            write_matrix_csv(os.path.join(out_dir, f"{name}.csv"), value)


def run(config: dict[str, Any]) -> dict[str, Any]:
    config, model_input_summary = apply_checkpoint_model_input_config(config)
    out_dir = ensure_dir(config.get("out_dir", "./next_hit_fit_rate"))

    dataset_path = config.get("dataset", config.get("dataset_path", None))
    if dataset_path is None:
        raise KeyError("FIT_RATE config needs 'dataset' or 'dataset_path'.")
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
    tau = float(timing["tau"])
    time_unit = str(timing["time_unit"])
    idx0_t, idx1_t = build_lagged_indices(
        q.shape[0],
        int(timing["lag_index_step"]),
        pack.traj_id,
        bool(config.get("allow_cross_traj_pairs", False)),
    )
    idx0 = idx0_t.numpy()
    idx1 = idx1_t.numpy()

    thresholds = make_thresholds(
        config.get("thresholds", None),
        n_thresholds=int(config.get("n_thresholds", 9)),
        start=float(config.get("threshold_start", 0.1)),
        stop=float(config.get("threshold_stop", 0.9)),
    ).numpy()
    pairs = resolve_ordered_pairs(
        n_states,
        adjacency=config.get("adjacency", config.get("flux_pairs", None)),
        symmetric_adjacency=bool(config.get("symmetric_adjacency", True)),
    )

    weights = pack.weights.numpy().astype(np.float64)
    state = pack.state.numpy().astype(np.int64)
    validate_rate_inputs(q, weights, state, idx0, idx1, n_states)
    positive_frame_mask, positive_pair_mask, zero_weight_mask_stats = positive_weight_masks(weights, idx0, idx1)
    idx0 = idx0[positive_pair_mask]
    idx1 = idx1[positive_pair_mask]
    q_weighted = q[positive_frame_mask]
    weights_weighted = weights[positive_frame_mask]
    state_weighted = state[positive_frame_mask]

    flux_device = setup_device(config.get("flux_device", config.get("device", "cuda:0")))
    flux_dtype = resolve_flux_dtype(config.get("flux_dtype", None), flux_device)
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
    pi_obs = estimate_pi(
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

    _table, J_matrix, k_flux = build_rate_table(pairs, J, variance, pi_obs, time_unit)
    C_matrix = matrix_from_pair_values(n_states, pairs, C_mean, fill=0.0)
    C_abs_matrix = matrix_from_pair_values(n_states, pairs, C_abs_mean, fill=0.0)
    branch_source = "T_hit"
    branch_path = config.get("P_matrix_npy", config.get("branch_matrix_npy", None))
    if branch_path is not None:
        raw_branch = np.load(branch_path).astype(np.float64)
        if raw_branch.shape != (n_states, n_states):
            raise ValueError(f"P_matrix_npy has shape {raw_branch.shape}; expected {(n_states, n_states)}.")
        branch_source = os.path.abspath(str(branch_path))
    else:
        raw_branch = T_hit
    P_branch_obs = branch_matrix_from_t_hit(raw_branch, mode=str(config.get("fit_branch_mode", "drop_self")))

    K_direct = assemble_generator_from_offdiag(k_flux)
    P_direct = compute_jump_probabilities(K_direct)
    MFPT_direct = compute_mfpt_matrix(K_direct)
    k_mfpt_direct = compute_mfpt_rate_matrix(MFPT_direct)

    fit = fit_generator_scf(k_flux, J_matrix, pi_obs, P_branch_obs, pairs, config)
    history = fit.pop("history")
    settings = fit.pop("settings")

    arrays = {
        "Q": q.astype(np.float32),
        "pi_obs": pi_obs,
        "T_hit": T_hit,
        "P_branch_raw": raw_branch,
        "P_branch_obs": P_branch_obs,
        "J_thresholds": J,
        "J_matrix": J_matrix,
        "C_matrix": C_matrix,
        "C_abs_matrix": C_abs_matrix,
        "k_flux": k_flux,
        "k_direct": k_flux,
        "K_direct": K_direct,
        "P_direct": P_direct,
        "MFPT_direct": MFPT_direct,
        "k_mfpt_direct": k_mfpt_direct,
        **{name: value for name, value in fit.items() if isinstance(value, np.ndarray)},
        "k_mfpt_fit": fit["k_eff_fit"],
    }
    save_matrix_outputs(out_dir, arrays)
    write_flux_profiles_csv(os.path.join(out_dir, "flux_profiles.csv"), pairs, thresholds, J, variance)
    write_population_csv(os.path.join(out_dir, "populations.csv"), pi_obs)
    write_vector_csv(os.path.join(out_dir, "pi_fit.csv"), "pi_fit", fit["pi_fit"])
    write_vector_csv(os.path.join(out_dir, "pi_residual.csv"), "pi_fit_minus_obs", fit["pi_residual"])
    write_vector_csv(os.path.join(out_dir, "exit_rate_fit.csv"), "exit_rate_fit", fit["exit_rate_fit"])
    write_transition_hit_csv(os.path.join(out_dir, "transition_hit_matrix.csv"), T_hit, exit_counts, exit_weight)
    write_fit_table(
        os.path.join(out_dir, "rate_constants_fit.csv"),
        pairs,
        pi_obs,
        J_matrix,
        k_flux,
        P_branch_obs,
        fit["rates_fit"],
        fit["P_fit"],
        fit["J_fit"],
        fit["MFPT_fit"],
        fit["k_eff_fit"],
        time_unit,
    )
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "fit_history.csv"), index=False)

    plot_paths: list[str] = []
    if bool(config.get("make_plots", True)):
        plot_dir = ensure_dir(os.path.join(out_dir, "diagnostics"))
        plot_specs = {
            "J_matrix": (J_matrix, "observed committor-current flux"),
            "T_hit": (T_hit, "raw exit endpoint next-hit averages"),
            "P_branch_obs": (P_branch_obs, "branching matrix used by fit"),
            "k_flux": (k_flux, "flux-normalized rate estimate"),
            "K_direct": (K_direct, "direct generator"),
            "P_direct": (P_direct, "direct jump probabilities"),
            "k_mfpt_direct": (k_mfpt_direct, "direct MFPT effective rate"),
            "rates_fit": (fit["rates_fit"], "fitted off-diagonal rates"),
            "K_fit": (fit["K_fit"], "fitted generator"),
            "P_fit": (fit["P_fit"], "fitted jump probabilities"),
            "J_fit": (fit["J_fit"], "fitted flux"),
            "J_residual": (fit["J_residual"], "fitted minus observed flux"),
            "P_residual": (fit["P_residual"], "fitted minus observed branching"),
            "MFPT_fit": (fit["MFPT_fit"], "fitted MFPT"),
            "k_eff_fit": (fit["k_eff_fit"], "fitted MFPT effective rate"),
        }
        for name, (matrix, label) in plot_specs.items():
            path = os.path.join(plot_dir, f"{name}.{config.get('plot_format', 'png')}")
            plot_matrix_heatmap(matrix, path, name, label)
            plot_paths.append(path)

        path = os.path.join(plot_dir, f"pi_obs_vs_fit.{config.get('plot_format', 'png')}")
        plot_vector_compare(pi_obs, fit["pi_fit"], path, "population consistency")
        plot_paths.append(path)

        path = os.path.join(plot_dir, f"fit_convergence.{config.get('plot_format', 'png')}")
        plot_convergence(history, path)
        plot_paths.append(path)

        pair_mask = pair_mask_matrix(n_states, pairs)
        path = os.path.join(plot_dir, f"flux_obs_vs_fit.{config.get('plot_format', 'png')}")
        plot_scatter(J_matrix, fit["J_fit"], pair_mask, path, "flux fit", "J_obs", "J_fit")
        plot_paths.append(path)

        path = os.path.join(plot_dir, f"branch_obs_vs_fit.{config.get('plot_format', 'png')}")
        plot_scatter(P_branch_obs, fit["P_fit"], pair_mask, path, "branch fit", "P_obs", "P_fit")
        plot_paths.append(path)

    final_history = history[-1] if history else {}
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
        "pi_mode": str(config.get("pi_mode", "labels")),
        "fit_branch_mode": str(config.get("fit_branch_mode", "drop_self")),
        "branch_source": branch_source,
        "fit_settings": settings,
        "fit_final_metrics": {k: float(v) for k, v in final_history.items()},
        "model_input": model_input_summary,
        "probability_checks": checks,
        "ordered_pairs": [[int(i), int(j)] for i, j in pairs],
        "thresholds": [float(x) for x in thresholds],
        "n_lagged_pairs": int(len(idx0)),
        "zero_weight_mask": zero_weight_mask_stats,
        "outputs": {
            "K_fit": os.path.abspath(os.path.join(out_dir, "K_fit.npy")),
            "P_fit": os.path.abspath(os.path.join(out_dir, "P_fit.npy")),
            "MFPT_fit": os.path.abspath(os.path.join(out_dir, "MFPT_fit.npy")),
            "k_eff_fit": os.path.abspath(os.path.join(out_dir, "k_eff_fit.npy")),
            "k_mfpt_fit": os.path.abspath(os.path.join(out_dir, "k_mfpt_fit.npy")),
            "rate_table": os.path.abspath(os.path.join(out_dir, "rate_constants_fit.csv")),
            "fit_history": os.path.abspath(os.path.join(out_dir, "fit_history.csv")),
        },
        "diagnostic_plots": [os.path.abspath(path) for path in plot_paths],
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    print(f"[FIT_RATE] Saved fitted kinetic model and diagnostics to {out_dir}")
    print(
        "[FIT_RATE] final normalized RMSE: "
        f"flux={final_history.get('flux_rmse', np.nan):.3e}, "
        f"branch={final_history.get('branch_rmse', np.nan):.3e}, "
        f"pi={final_history.get('pi_rmse', np.nan):.3e}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a next-hit kinetic model from flux and branching observables.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "NEXT_HIT_FIT_RATE", "FIT_RATE", "NEXT_HIT_RATE", "RATE_CONSTANT")
    run(cfg)


if __name__ == "__main__":
    main()

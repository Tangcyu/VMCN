from __future__ import annotations

from itertools import combinations
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import ensure_dir, stage_path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def cv_pairs_from_config(cfg: Dict) -> List[List[str]]:
    plotting = cfg.get("plotting", {})
    if "cv_pairs" in plotting and plotting["cv_pairs"]:
        return [list(pair) for pair in plotting["cv_pairs"] if len(pair) == 2]
    if "cv_pair" in plotting and plotting["cv_pair"]:
        return [list(plotting["cv_pair"])]
    return [list(pair) for pair in combinations(cfg["data"]["cvs"], 2)]


def plot_implied_timescales(cfg: Dict, msms: Dict[int, Dict[str, np.ndarray]]) -> str:
    out_dir = ensure_dir(stage_path(cfg, "05_plots"))
    path = f"{out_dir}/implied_timescales.png"
    lags = sorted(msms)
    max_len = max(len(msms[lag]["implied_timescales"]) for lag in lags)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i in range(max_len):
        y = [msms[lag]["implied_timescales"][i] if i < len(msms[lag]["implied_timescales"]) else np.nan for lag in lags]
        ax.plot(lags, y, marker="o", linewidth=1.2, label=f"t{i + 1}")
    ax.set_xlabel("Lag time (frames)")
    ax.set_ylabel("Implied timescale (frames)")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _nontrivial_eigenvalues(T: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvals(np.asarray(T, dtype=np.float64))
    vals = np.sort(np.abs(vals))[::-1]
    return vals[vals < 1.0 - 1e-10]


def _timescales_from_eigenvalues(vals: np.ndarray, lag: int) -> np.ndarray:
    vals = np.clip(np.asarray(vals, dtype=np.float64), 1e-15, 1.0 - 1e-12)
    return -float(lag) / np.log(vals)


def plot_spectral_analysis(cfg: Dict, msms: Dict[int, Dict[str, np.ndarray]]) -> list[str]:
    spec_cfg = cfg.get("spectral_analysis", {})
    if not bool(spec_cfg.get("enabled", True)):
        return []

    out_dir = ensure_dir(stage_path(cfg, "05_plots", "spectrum"))
    n_eigs = int(spec_cfg.get("n_eigenvalues", cfg["msm"].get("n_timescales", 12)))
    max_processes = int(spec_cfg.get("max_processes", max(2, n_eigs - 1)))
    selected_lag = spec_cfg.get("selected_lag", cfg.get("pcca", {}).get("selected_lag", None))
    selected_lag = None if selected_lag is None else int(selected_lag)

    paths: list[str] = []
    rows = []
    suggestion_rows = []
    best_by_lag = {}

    for lag in sorted(msms):
        result = msms[lag]
        T = np.asarray(result.get("active_transition_matrix", result["transition_matrix"]), dtype=np.float64)
        vals = _nontrivial_eigenvalues(T)
        keep = min(n_eigs, vals.size)
        vals_keep = vals[:keep]
        its = _timescales_from_eigenvalues(vals_keep, int(lag))

        for idx, (eigval, ts) in enumerate(zip(vals_keep, its), start=1):
            rows.append(
                {
                    "lag": int(lag),
                    "process_index": idx,
                    "candidate_m_for_this_process": idx + 1,
                    "eigenvalue_abs": float(eigval),
                    "timescale_frames": float(ts),
                }
            )

        n_gap = min(max_processes, max(0, vals.size - 1))
        gap_rows = []
        for process_idx in range(1, n_gap + 1):
            lam_k = float(vals[process_idx - 1])
            lam_next = float(vals[process_idx])
            eigengap = lam_k - lam_next
            t_k = float(_timescales_from_eigenvalues(np.asarray([lam_k]), int(lag))[0])
            t_next = float(_timescales_from_eigenvalues(np.asarray([lam_next]), int(lag))[0])
            ratio = t_k / max(t_next, 1e-300)
            row = {
                "lag": int(lag),
                "slow_processes": process_idx,
                "candidate_m": process_idx + 1,
                "lambda_k": lam_k,
                "lambda_next": lam_next,
                "eigengap": eigengap,
                "timescale_k_frames": t_k,
                "timescale_next_frames": t_next,
                "timescale_ratio": ratio,
            }
            rows.append(row)
            suggestion_rows.append(row)
            gap_rows.append(row)

        if gap_rows:
            # Rank by eigengap first. The timescale ratio is saved as a sanity check.
            best = max(gap_rows, key=lambda item: item["eigengap"])
            best_by_lag[int(lag)] = best["candidate_m"]

    spectrum_csv = f"{out_dir}/spectral_analysis.csv"
    pd.DataFrame(rows).to_csv(spectrum_csv, index=False)
    paths.append(spectrum_csv)

    suggestions = pd.DataFrame(suggestion_rows)
    if not suggestions.empty:
        suggestions_csv = f"{out_dir}/candidate_m_by_lag.csv"
        suggestions.to_csv(suggestions_csv, index=False)
        paths.append(suggestions_csv)

    lags = sorted(msms)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for idx in range(1, n_eigs + 1):
        y = []
        for lag in lags:
            vals = _nontrivial_eigenvalues(np.asarray(msms[lag].get("active_transition_matrix", msms[lag]["transition_matrix"])))
            y.append(vals[idx - 1] if vals.size >= idx else np.nan)
        ax.plot(lags, y, marker="o", linewidth=1.2, label=f"lambda {idx}")
    ax.set_xlabel("Lag time (frames)")
    ax.set_ylabel("Nontrivial |eigenvalue|")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    path = f"{out_dir}/eigenvalue_spectrum_vs_lag.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for process_idx in range(1, max_processes + 1):
        y = []
        for lag in lags:
            vals = _nontrivial_eigenvalues(np.asarray(msms[lag].get("active_transition_matrix", msms[lag]["transition_matrix"])))
            if vals.size > process_idx:
                y.append(vals[process_idx - 1] - vals[process_idx])
            else:
                y.append(np.nan)
        ax.plot(lags, y, marker="o", linewidth=1.2, label=f"m={process_idx + 1}")
    ax.set_xlabel("Lag time (frames)")
    ax.set_ylabel("Eigengap after slow processes")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    path = f"{out_dir}/eigengap_candidate_m_vs_lag.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    if selected_lag in msms:
        vals = _nontrivial_eigenvalues(np.asarray(msms[selected_lag].get("active_transition_matrix", msms[selected_lag]["transition_matrix"])))
        vals = vals[:n_eigs]
        its = _timescales_from_eigenvalues(vals, selected_lag)
        x = np.arange(1, len(vals) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
        axes[0].plot(x, vals, marker="o")
        axes[0].set_xlabel("Process index")
        axes[0].set_ylabel("Nontrivial |eigenvalue|")
        axes[0].set_title(f"Eigenvalues, lag={selected_lag}")

        axes[1].plot(x, its, marker="o")
        axes[1].set_xlabel("Process index")
        axes[1].set_ylabel("Timescale (frames)")
        axes[1].set_yscale("log")
        axes[1].set_title("Timescale spectrum")

        if vals.size >= 2:
            gap_x = np.arange(1, min(max_processes, vals.size - 1) + 1)
            gaps = vals[gap_x - 1] - vals[gap_x]
            axes[2].bar(gap_x + 1, gaps)
            axes[2].set_xlabel("Candidate m")
            axes[2].set_ylabel("Eigengap")
            axes[2].set_title("Largest gap suggests m")
        fig.tight_layout()
        path = f"{out_dir}/selected_lag_{selected_lag}_spectrum.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    if best_by_lag:
        counts = pd.Series(list(best_by_lag.values())).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Candidate m")
        ax.set_ylabel("Number of lags selecting m")
        ax.set_title("Spectral-gap vote across lags")
        fig.tight_layout()
        path = f"{out_dir}/candidate_m_votes.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    return paths


def plot_macrostates(cfg: Dict, table: pd.DataFrame, micro: Dict[str, np.ndarray], pcca: Dict[int, Dict[int, Dict[str, np.ndarray]]]) -> list[str]:
    out_dir = ensure_dir(stage_path(cfg, "05_plots", "macrostates"))
    cv_pairs = cv_pairs_from_config(cfg)
    max_points = int(cfg["plotting"].get("max_points", 200000))
    frame_idx = np.arange(len(table))
    if len(frame_idx) > max_points:
        rng = np.random.default_rng(int(cfg["project"].get("seed", 2026)))
        frame_idx = np.sort(rng.choice(frame_idx, size=max_points, replace=False))

    paths: list[str] = []
    micro_labels = np.asarray(micro["labels"], dtype=np.int64)
    weight_column = cfg["data"].get("weight_column", "weight")
    if weight_column in table.columns:
        plot_origin_valid = pd.to_numeric(table[weight_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) > 0.0
    else:
        plot_origin_valid = np.ones(len(table), dtype=bool)
    for lag, by_m in pcca.items():
        for m, result in by_m.items():
            macro = np.asarray(result["macro_by_micro"], dtype=np.int64)[micro_labels]
            macro_plot = macro.copy()
            macro_plot[~plot_origin_valid] = -1
            for cvs in cv_pairs:
                if cvs[0] not in table.columns or cvs[1] not in table.columns:
                    print(f"[warn] skip macrostate plot for missing CV pair: {cvs}")
                    continue
                pair_name = f"{_safe_name(cvs[0])}_vs_{_safe_name(cvs[1])}"
                path = f"{out_dir}/lag_{lag}_m_{m}_{pair_name}.png"
                fig, ax = plt.subplots(figsize=(6.0, 5.2))
                masked = macro_plot[frame_idx] < 0
                if np.any(masked):
                    ax.scatter(
                        table.iloc[frame_idx[masked]][cvs[0]],
                        table.iloc[frame_idx[masked]][cvs[1]],
                        c="lightgray",
                        s=4,
                        alpha=0.18,
                        linewidths=0,
                        rasterized=True,
                        label="weight=0",
                    )
                active = ~masked
                if np.any(active):
                    sc = ax.scatter(
                        table.iloc[frame_idx[active]][cvs[0]],
                        table.iloc[frame_idx[active]][cvs[1]],
                        c=macro_plot[frame_idx[active]],
                        s=5,
                        cmap="tab10",
                        alpha=0.55,
                        linewidths=0,
                        rasterized=True,
                    )
                    fig.colorbar(sc, ax=ax, label="macrostate")
                ax.set_xlabel(cvs[0])
                ax.set_ylabel(cvs[1])
                ax.set_title(f"Macrostates: lag={lag}, m={m}")
                fig.tight_layout()
                fig.savefig(path, dpi=220)
                plt.close(fig)
                paths.append(path)
    return paths


def plot_ck(cfg: Dict, pcca: Dict[int, Dict[int, Dict[str, np.ndarray]]]) -> list[str]:
    out_dir = ensure_dir(stage_path(cfg, "05_plots", "ck"))
    paths: list[str] = []
    for lag, by_m in pcca.items():
        for m, result in by_m.items():
            multiples = np.asarray(result["ck_multiples"], dtype=np.int64)
            x = multiples * int(lag)
            direct = np.asarray(result["ck_direct"], dtype=np.float64)
            predicted = np.asarray(result["ck_predicted"], dtype=np.float64)

            path = f"{out_dir}/lag_{lag}_m_{m}_ck_rmsd.png"
            fig, ax = plt.subplots(figsize=(5.8, 4.0))
            ax.plot(x, result["ck_rmsd"], marker="o")
            ax.set_xlabel("Lag multiple (frames)")
            ax.set_ylabel("CK RMSD")
            ax.set_title(f"CK validation: m={m}")
            fig.tight_layout()
            fig.savefig(path, dpi=220)
            plt.close(fig)
            paths.append(path)

            path = f"{out_dir}/lag_{lag}_m_{m}_ck_self_transitions.png"
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(m, 2)))
            for state in range(m):
                observed = direct[:, state, state]
                expected = predicted[:, state, state]
                color = colors[state % len(colors)]
                ax.plot(x, observed, marker="o", color=color, linewidth=1.5, label=f"state {state} observed")
                ax.plot(x, expected, marker="s", color=color, linewidth=1.2, linestyle="--", label=f"state {state} predicted")
            ax.set_xlabel("Lag time (frames)")
            ax.set_ylabel("Self-transition probability")
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"CK self-transitions: lag={lag}, m={m}")
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(path, dpi=220)
            plt.close(fig)
            paths.append(path)

            path = f"{out_dir}/lag_{lag}_m_{m}_ck_observed_vs_predicted.png"
            obs = direct.reshape(direct.shape[0], -1)
            pred = predicted.reshape(predicted.shape[0], -1)
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            for idx, lag_time in enumerate(x):
                ax.scatter(pred[idx], obs[idx], s=14, alpha=0.65, label=f"{lag_time}")
            ax.plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle=":")
            ax.set_xlabel("Predicted transition probability")
            ax.set_ylabel("Observed transition probability")
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"CK observed vs predicted: m={m}")
            ax.legend(title="frames", frameon=False, fontsize=7)
            fig.tight_layout()
            fig.savefig(path, dpi=220)
            plt.close(fig)
            paths.append(path)

            rows = []
            for idx, mult in enumerate(multiples):
                for i in range(m):
                    for j in range(m):
                        rows.append(
                            {
                                "lag": lag,
                                "multiple": int(mult),
                                "lag_time_frames": int(mult) * int(lag),
                                "from_macrostate": i,
                                "to_macrostate": j,
                                "observed": float(direct[idx, i, j]),
                                "predicted": float(predicted[idx, i, j]),
                                "difference": float(direct[idx, i, j] - predicted[idx, i, j]),
                            }
                        )
            csv_path = f"{out_dir}/lag_{lag}_m_{m}_ck_observed_vs_predicted.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            paths.append(csv_path)
    return paths

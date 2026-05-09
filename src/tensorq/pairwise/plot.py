from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, cv_headers_for_pack, infer_n_states, load_dataset, select_model_inputs, unordered_pairs
from .predict import apply_checkpoint_input_config, infer_pairwise, load_pairwise_committor_model, reconstruct_state_probabilities


def weighted_mean_2d(x, y, v, w, xedges, yedges):
    denom, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w)
    numer, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w * v)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = numer / denom
    avg[denom <= 0] = np.nan
    return avg


def load_or_infer_Q(config: dict[str, Any], pack, n_pairs: int) -> np.ndarray:
    q_path = config.get("Q_npy", config.get("q_npy", None))
    if q_path is not None:
        Q = np.load(q_path).astype(np.float32)
    else:
        model_path = config.get("model", None)
        if model_path is None:
            raise KeyError("Plot config needs Q_npy or model.")
        device = setup_device(config.get("device", config.get("prediction_device", "cuda:0")))
        model = load_pairwise_committor_model(model_path, device)
        features, _ = select_model_inputs(pack, config)
        Q = infer_pairwise(model, features.float(), device, batch_size=int(config.get("batch_size", 65536)))
    if Q.ndim != 2 or Q.shape[1] != int(n_pairs):
        raise RuntimeError(f"Q shape {Q.shape} does not match n_pairs={n_pairs}.")
    return Q


def plot_distributions(Q: np.ndarray, P: np.ndarray, weights: np.ndarray, pairs: list[tuple[int, int]], out_dir: str, fmt: str, bins: int) -> list[str]:
    paths = []
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for col, (i, j) in enumerate(pairs):
        ax.hist(Q[:, col], bins=bins, range=(0.0, 1.0), weights=weights, histtype="step", linewidth=1.2, label=f"Q_{i}_{j}")
    ax.set_xlabel("pair-wise committor")
    ax.set_ylabel("weighted density")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, f"Q_distributions.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for j in range(P.shape[1]):
        ax.hist(P[:, j], bins=bins, range=(0.0, 1.0), weights=weights, histtype="step", linewidth=1.3, label=f"P_{j}")
    ax.set_xlabel("reconstructed state probability")
    ax.set_ylabel("weighted density")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, f"P_distributions.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


def normalize_planes(value: Any) -> list[list[str]]:
    if not value:
        return []
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
        value = [value]
    out = []
    for plane in value:
        names = [str(name) for name in plane]
        if len(names) != 2:
            raise ValueError("Minimal pair-wise committor plotting supports 2D CV planes only.")
        out.append(names)
    return out


def plot_cv_fields(config: dict[str, Any], pack, Q: np.ndarray, P: np.ndarray, pairs: list[tuple[int, int]], weights: np.ndarray, out_dir: str) -> list[str]:
    planes = normalize_planes(config.get("planes", []))
    if not planes:
        return []
    if pack.cv is None:
        raise RuntimeError("CV projection requested, but dataset has no cv block.")
    cv = pack.cv.numpy()
    headers = cv_headers_for_pack(pack)
    fmt = str(config.get("format", "png"))
    bins = int(config.get("bins", 60))
    paths = []
    for cvx, cvy in planes:
        if cvx not in headers or cvy not in headers:
            raise ValueError(f"CV plane {[cvx, cvy]} contains columns not present in dataset.")
        x = cv[:, headers.index(cvx)]
        y = cv[:, headers.index(cvy)]
        xlim = config.get("xlim", [float(np.nanmin(x)), float(np.nanmax(x))])
        ylim = config.get("ylim", [float(np.nanmin(y)), float(np.nanmax(y))])
        xedges = np.linspace(float(xlim[0]), float(xlim[1]), bins + 1)
        yedges = np.linspace(float(ylim[0]), float(ylim[1]), bins + 1)
        subdir = ensure_dir(os.path.join(out_dir, f"{cvx}__{cvy}"))
        for col, (i, j) in enumerate(pairs):
            field = weighted_mean_2d(x, y, Q[:, col], weights, xedges, yedges)
            path = os.path.join(subdir, f"Q_{i}_{j}__{cvx}__{cvy}.{fmt}")
            fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=160)
            pcm = ax.pcolormesh(xedges, yedges, field.T, cmap=str(config.get("Q_cmap", "RdBu_r")), shading="auto", vmin=0.0, vmax=1.0)
            ax.set_xlabel(cvx)
            ax.set_ylabel(cvy)
            ax.set_title(f"Q_{i}_{j}")
            fig.colorbar(pcm, ax=ax)
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
        for j in range(P.shape[1]):
            field = weighted_mean_2d(x, y, P[:, j], weights, xedges, yedges)
            path = os.path.join(subdir, f"P_{j}__{cvx}__{cvy}.{fmt}")
            fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=160)
            pcm = ax.pcolormesh(xedges, yedges, field.T, cmap=str(config.get("P_cmap", "viridis")), shading="auto", vmin=0.0, vmax=1.0)
            ax.set_xlabel(cvx)
            ax.set_ylabel(cvy)
            ax.set_title(f"P_{j}")
            fig.colorbar(pcm, ax=ax)
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def run(config: dict[str, Any]) -> dict[str, Any]:
    config, input_source = apply_checkpoint_input_config(config)
    out_dir = ensure_dir(config.get("out_dir", "./pairwise_committor_plots"))
    dataset_path = config.get("dataset", config.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("Plot config needs 'dataset' or 'dataset_path'.")
    pack = apply_stride(load_dataset(dataset_path), int(config.get("dataset_stride", 1)))
    n_states = infer_n_states(pack, config.get("n_states", None))
    pairs = unordered_pairs(n_states)
    Q = load_or_infer_Q(config, pack, len(pairs))
    P = reconstruct_state_probabilities(
        Q,
        n_states,
        anchor_state=int(config.get("anchor_state", 0)),
        eps=float(config.get("eps", 1e-4)),
        chunk_size=int(config.get("reconstruct_chunk", 20000)),
    )
    weights = pack.weights.numpy().astype(np.float64)
    weights = weights / (np.sum(weights) + 1e-300)
    fmt = str(config.get("format", "png"))
    paths = plot_distributions(Q, P, weights, pairs, out_dir, fmt, int(config.get("distribution_bins", 60)))
    paths.extend(plot_cv_fields(config, pack, Q, P, pairs, weights, out_dir))
    summary = {
        "dataset": os.path.abspath(str(dataset_path)),
        "out_dir": os.path.abspath(out_dir),
        "n_states": int(n_states),
        "pairs": [[int(i), int(j)] for i, j in pairs],
        "model_input": input_source,
        "plots": [os.path.abspath(path) for path in paths],
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    print(f"[PLOT] Saved pair-wise committor distributions and CV fields to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pair-wise committors and reconstructed probabilities.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "PAIRWISE_PLOT", "PLOT")
    run(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, infer_n_states, load_dataset, select_model_inputs, unordered_pairs
from .predict import (
    apply_checkpoint_input_config,
    infer_pairwise,
    load_pairwise_committor_model,
    probability_checks,
    reconstruct_state_probabilities,
)


def save_assignments_csv(path: str, Q: np.ndarray, P: np.ndarray, pack, pairs: list[tuple[int, int]], include_cv: bool) -> None:
    data: dict[str, Any] = {"frame": np.arange(Q.shape[0], dtype=np.int64)}
    if pack.traj_id is not None:
        data["traj_id"] = pack.traj_id.numpy().astype(np.int64)
    data["weight"] = pack.weights.numpy().astype(np.float64)
    data["meta_state"] = pack.state.numpy().astype(np.int64)
    data["destination_argmax"] = np.argmax(P, axis=1).astype(np.int64)
    for col, (i, j) in enumerate(pairs):
        data[f"Q_{i}_{j}"] = Q[:, col].astype(np.float32)
    for j in range(P.shape[1]):
        data[f"P_{j}"] = P[:, j].astype(np.float32)
    df = pd.DataFrame(data)
    if include_cv and pack.cv is not None:
        headers = pack.meta.get("cv_headers", None)
        cv = pack.cv.numpy()
        if headers is None or len(headers) != cv.shape[1]:
            headers = [f"cv_{idx}" for idx in range(cv.shape[1])]
        for idx, name in enumerate(headers):
            df[str(name)] = cv[:, idx]
    df.to_csv(path, index=False)


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    config, input_source = apply_checkpoint_input_config(config)
    out_dir = ensure_dir(config.get("out_dir", "./pairwise_committor_infer"))
    dataset_path = config.get("dataset", config.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("Inference config needs 'dataset' or 'dataset_path'.")
    model_path = config["model"]
    device = setup_device(config.get("device", "cuda:0"))

    pack = apply_stride(load_dataset(dataset_path), int(config.get("dataset_stride", 1)))
    n_states = infer_n_states(pack, config.get("n_states", None))
    pairs = unordered_pairs(n_states)
    model_features, input_meta = select_model_inputs(pack, config)
    model = load_pairwise_committor_model(model_path, device)
    Q = infer_pairwise(model, model_features.float(), device, batch_size=int(config.get("batch_size", 65536)))
    if Q.shape != (model_features.shape[0], len(pairs)):
        raise RuntimeError(f"Model returned Q shape {Q.shape}, expected ({model_features.shape[0]}, {len(pairs)}).")
    P = reconstruct_state_probabilities(
        Q,
        n_states,
        anchor_state=int(config.get("anchor_state", 0)),
        eps=float(config.get("eps", 1e-4)),
        chunk_size=int(config.get("reconstruct_chunk", 20000)),
    )

    q_path = os.path.join(out_dir, "Q.npy")
    p_path = os.path.join(out_dir, "P.npy")
    dest_path = os.path.join(out_dir, "destination_argmax.npy")
    np.save(q_path, Q)
    np.save(p_path, P)
    np.save(dest_path, np.argmax(P, axis=1).astype(np.int64))
    csv_path = None
    if bool(config.get("save_csv", False)):
        csv_path = os.path.join(out_dir, "pairwise_committor_assignments.csv")
        save_assignments_csv(csv_path, Q, P, pack, pairs, include_cv=bool(config.get("include_cv_in_csv", True)))

    summary = {
        "dataset": os.path.abspath(str(dataset_path)),
        "model": os.path.abspath(str(model_path)),
        "out_dir": os.path.abspath(out_dir),
        "n_frames": int(Q.shape[0]),
        "n_states": int(n_states),
        "pairs": [[int(i), int(j)] for i, j in pairs],
        "model_input": input_meta,
        "checkpoint_input": input_source,
        "Q_npy": os.path.abspath(q_path),
        "P_npy": os.path.abspath(p_path),
        "destination_argmax_npy": os.path.abspath(dest_path),
        "assignments_csv": os.path.abspath(csv_path) if csv_path else None,
        "probability_checks": probability_checks(P, atol=float(config.get("normalization_atol", 1e-4))),
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    print(f"[INFER] Saved Q.npy, P.npy, and destination_argmax.npy to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer pair-wise committors and reconstructed state probabilities.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "PAIRWISE_INFER", "INFER")
    run_inference(cfg)


if __name__ == "__main__":
    main()

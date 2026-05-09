from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import apply_stride, infer_n_states, load_dataset, select_model_inputs
from ..common.flux import unordered_pairs
from .predict import check_probability_rows, infer_probabilities, load_committor_model


def boundary_summary(q: np.ndarray, state: np.ndarray, n_states: int) -> dict[str, Any]:
    valid = state >= 0
    out: dict[str, Any] = {"n_labeled": int(np.sum(valid))}
    if not np.any(valid):
        out["boundary_accuracy"] = None
        out["mean_q_on_own_state"] = None
        return out
    pred = np.argmax(q, axis=1)
    out["boundary_accuracy"] = float(np.mean(pred[valid] == state[valid]))
    own = []
    for k in range(n_states):
        mask = state == k
        own.append(float(np.mean(q[mask, k])) if np.any(mask) else None)
    out["mean_q_on_own_state"] = own
    return out


def save_assignments_csv(
    path: str,
    q: np.ndarray,
    destination: np.ndarray,
    pack,
    include_cv: bool = True,
) -> None:
    data: dict[str, Any] = {"frame": np.arange(q.shape[0], dtype=np.int64)}
    if pack.traj_id is not None:
        data["traj_id"] = pack.traj_id.numpy().astype(np.int64)
    data["weight"] = pack.weights.numpy().astype(np.float64)
    data["meta_state"] = pack.state.numpy().astype(np.int64)
    data["destination_argmax"] = destination.astype(np.int64)
    for j in range(q.shape[1]):
        data[f"q_{j}"] = q[:, j].astype(np.float32)
    df = pd.DataFrame(data)

    if include_cv and pack.cv is not None:
        cv_headers = pack.meta.get("cv_headers", None)
        cv = pack.cv.numpy()
        if cv_headers is None or len(cv_headers) != cv.shape[1]:
            cv_headers = [f"cv_{i}" for i in range(cv.shape[1])]
        for idx, name in enumerate(cv_headers):
            df[str(name)] = cv[:, idx]
    df.to_csv(path, index=False)


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(config.get("out_dir", "./next_hit_infer"))
    dataset_path = config.get("dataset", config.get("dataset_path"))
    if dataset_path is None:
        raise KeyError("Inference config needs 'dataset' or 'dataset_path'.")
    model_path = config["model"]
    device = setup_device(config.get("device", "cuda:0"))

    pack = apply_stride(load_dataset(dataset_path), int(config.get("dataset_stride", 1)))
    n_states = infer_n_states(pack, config.get("n_states", None))
    model_features, input_meta = select_model_inputs(pack, config)
    model = load_committor_model(model_path, device)
    q = infer_probabilities(model, model_features.float(), device, batch_size=int(config.get("batch_size", 65536)))
    if q.ndim != 2 or q.shape[1] != n_states:
        raise RuntimeError(f"Model returned q shape {q.shape}, expected (_, {n_states}).")

    destination = np.argmax(q, axis=1).astype(np.int64)
    np.save(os.path.join(out_dir, "Q.npy"), q)
    np.save(os.path.join(out_dir, "destination_argmax.npy"), destination)

    if bool(config.get("save_csv", False)):
        save_assignments_csv(
            os.path.join(out_dir, "committor_assignments.csv"),
            q=q,
            destination=destination,
            pack=pack,
            include_cv=bool(config.get("include_cv_in_csv", True)),
        )

    reactive_weight_path = None
    reactive_columns_path = None
    if bool(config.get("save_reactive_weights", True)):
        pairs = unordered_pairs(n_states)
        rw = np.stack([q[:, i] * q[:, j] for i, j in pairs], axis=1).astype(np.float32)
        reactive_weight_path = os.path.join(out_dir, "reactive_weights.npy")
        reactive_columns_path = os.path.join(out_dir, "reactive_weight_columns.csv")
        np.save(reactive_weight_path, rw)
        pd.DataFrame(
            [{"column": k, "state_i": i, "state_j": j} for k, (i, j) in enumerate(pairs)]
        ).to_csv(reactive_columns_path, index=False)

    checks = check_probability_rows(q, atol=float(config.get("normalization_atol", 1e-4)))
    bsum = boundary_summary(q, pack.state.numpy().astype(np.int64), n_states)
    entropy = -np.sum(np.clip(q, 1e-12, 1.0) * np.log(np.clip(q, 1e-12, 1.0)), axis=1)

    summary = {
        "dataset": os.path.abspath(str(dataset_path)),
        "model": os.path.abspath(str(model_path)),
        "out_dir": os.path.abspath(out_dir),
        "n_frames": int(q.shape[0]),
        "model_input": input_meta,
        "n_states": int(n_states),
        "Q_npy": os.path.abspath(os.path.join(out_dir, "Q.npy")),
        "destination_argmax_npy": os.path.abspath(os.path.join(out_dir, "destination_argmax.npy")),
        "reactive_weights_npy": os.path.abspath(reactive_weight_path) if reactive_weight_path else None,
        "reactive_weight_columns_csv": os.path.abspath(reactive_columns_path) if reactive_columns_path else None,
        "probability_checks": checks,
        "boundary_checks": bsum,
        "mean_entropy": float(np.mean(entropy)),
    }
    write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
    print(f"[INFER] Saved Q.npy and destination_argmax.npy to {out_dir}")
    print(f"[CHECK] max |sum(q)-1| = {checks['max_sum_error']:.3e}, min(q) = {checks['min_q']:.3e}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer q_j(z) for all frames with a next-hit committor model.")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "NEXT_HIT_INFER", "INFER")
    run_inference(cfg)


if __name__ == "__main__":
    main()

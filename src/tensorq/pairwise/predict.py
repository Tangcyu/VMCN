from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ..common.config import torch_load
from ..common.data import unordered_pairs
from .model import PairwiseCommittorNet


def load_pairwise_committor_model(path: str | os.PathLike[str], device: torch.device) -> torch.nn.Module:
    path = str(path)
    try:
        model = torch.jit.load(path, map_location=device)
        model.eval()
        return model
    except Exception:
        checkpoint = torch_load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Could not load {path} as TorchScript or a pair-wise committor checkpoint.")
    kwargs: dict[str, Any] | None = checkpoint.get("model_kwargs")
    if kwargs is None:
        raise RuntimeError("Checkpoint is missing 'model_kwargs'; cannot rebuild PairwiseCommittorNet.")
    model = PairwiseCommittorNet(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model



def checkpoint_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        checkpoint = torch_load(path, map_location="cpu")
    except Exception:
        return {}
    return checkpoint if isinstance(checkpoint, dict) else {}


def apply_checkpoint_input_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    effective = dict(config)
    model_path = effective.get("model", None)
    if model_path is None:
        return effective, {"model_input_source": "not_used_no_model"}
    meta = checkpoint_metadata(model_path).get("model_input", {})
    if not isinstance(meta, dict) or not meta:
        return effective, {"model_input_source": "config", "checkpoint_model_input": {}}
    mapping = {
        "model_input_space": meta.get("model_input_space", None),
        "cvs_to_use": meta.get("model_cvs_to_use", None),
        "periodic_cvs": meta.get("model_periodic_cvs", None),
        "periodic_cv_units": meta.get("model_periodic_cv_units", None),
    }
    applied: dict[str, Any] = {}
    for key, value in mapping.items():
        if value is not None and effective.get(key, None) is None:
            effective[key] = value
            applied[key] = value
    return effective, {
        "model_input_source": "checkpoint" if applied else "config",
        "checkpoint_model_input": meta,
        "applied_model_input": applied,
    }


def infer_pairwise(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, int(features.shape[0]), int(batch_size)), desc="infer Q"):
            end = min(int(features.shape[0]), start + int(batch_size))
            q = model(features[start:end].to(device, non_blocking=True))
            out.append(torch.clamp(q, 0.0, 1.0).detach().float().cpu().numpy())
    return np.vstack(out).astype(np.float32)


def infer_n_states_from_pair_dim(n_pairs: int, requested: int | None = None) -> int:
    if requested is not None:
        n_states = int(requested)
    else:
        n_states = int((1 + np.sqrt(1 + 8 * int(n_pairs))) / 2)
    if n_states * (n_states - 1) // 2 != int(n_pairs):
        raise RuntimeError(f"Cannot match {n_pairs} pair columns to C(n_states, 2).")
    return n_states


def reconstruct_state_probabilities(
    Q: np.ndarray,
    n_states: int,
    *,
    anchor_state: int = 0,
    eps: float = 1e-4,
    chunk_size: int = 20000,
) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.float64)
    pairs = unordered_pairs(n_states)
    if Q.ndim != 2 or Q.shape[1] != len(pairs):
        raise ValueError(f"Q must have shape (n_frames, {len(pairs)}); got {Q.shape}.")
    if not (0 <= int(anchor_state) < int(n_states)):
        raise ValueError("anchor_state is outside n_states.")

    var_index: dict[int, int] = {}
    col = 0
    for state in range(int(n_states)):
        if state == int(anchor_state):
            continue
        var_index[state] = col
        col += 1

    A = np.zeros((len(pairs), int(n_states) - 1), dtype=np.float64)
    for row, (i, j) in enumerate(pairs):
        if j != int(anchor_state):
            A[row, var_index[j]] += 1.0
        if i != int(anchor_state):
            A[row, var_index[i]] -= 1.0
    pinv = np.linalg.pinv(A)

    P = np.empty((Q.shape[0], int(n_states)), dtype=np.float32)
    for start in range(0, Q.shape[0], int(chunk_size)):
        end = min(Q.shape[0], start + int(chunk_size))
        q = np.clip(Q[start:end], float(eps), 1.0 - float(eps))
        logits = np.log(q) - np.log1p(-q)
        s_free = logits @ pinv.T
        s = np.zeros((end - start, int(n_states)), dtype=np.float64)
        for state, idx in var_index.items():
            s[:, state] = s_free[:, idx]
        s -= np.max(s, axis=1, keepdims=True)
        exp_s = np.exp(s)
        P[start:end] = (exp_s / (np.sum(exp_s, axis=1, keepdims=True) + 1e-300)).astype(np.float32)
    return P


def probability_checks(P: np.ndarray, atol: float = 1e-4) -> dict[str, float | bool]:
    row_sum = np.sum(P, axis=1)
    return {
        "max_sum_error": float(np.max(np.abs(row_sum - 1.0))) if P.size else 0.0,
        "min_probability": float(np.min(P)) if P.size else 0.0,
        "normalization_ok": bool((np.max(np.abs(row_sum - 1.0)) if P.size else 0.0) <= float(atol)),
        "nonnegative_ok": bool((np.min(P) if P.size else 0.0) >= -float(atol)),
    }

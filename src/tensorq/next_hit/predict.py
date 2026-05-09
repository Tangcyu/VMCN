from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ..common.config import torch_load
from .model import NextHitCommittorNet


def load_committor_model(path: str | os.PathLike[str], device: torch.device) -> torch.nn.Module:
    path = str(path)
    try:
        model = torch.jit.load(path, map_location=device)
        model.eval()
        return model
    except Exception:
        checkpoint = torch_load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Could not load {path} as TorchScript or a next_hit checkpoint.")
    kwargs: dict[str, Any] | None = checkpoint.get("model_kwargs")
    if kwargs is None:
        raise RuntimeError("Checkpoint is missing 'model_kwargs'; cannot rebuild NextHitCommittorNet.")
    model = NextHitCommittorNet(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def infer_probabilities(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    n_frames = int(features.shape[0])
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n_frames, int(batch_size)), desc="infer q"):
            end = min(n_frames, start + int(batch_size))
            q = model(features[start:end].to(device, non_blocking=True))
            out.append(q.detach().float().cpu().numpy())
    return np.vstack(out).astype(np.float32)


def check_probability_rows(q: np.ndarray, atol: float = 1e-4) -> dict[str, float | bool]:
    row_sum = np.sum(q, axis=1)
    max_sum_error = float(np.max(np.abs(row_sum - 1.0))) if q.size else 0.0
    min_q = float(np.min(q)) if q.size else 0.0
    return {
        "max_sum_error": max_sum_error,
        "min_q": min_q,
        "normalization_ok": bool(max_sum_error <= float(atol)),
        "nonnegative_ok": bool(min_q >= -float(atol)),
    }

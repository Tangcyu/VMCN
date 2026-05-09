from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import torch_load


@dataclass
class CommittorDatasetPack:
    features: torch.Tensor
    weights: torch.Tensor
    state: torch.Tensor
    traj_id: torch.Tensor | None
    cv: torch.Tensor | None
    meta: dict[str, Any]
    source_path: str


def _list_or_empty(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype)
    return torch.as_tensor(value, dtype=dtype)


def load_dataset(path: str | os.PathLike[str]) -> CommittorDatasetPack:
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pt", ".pth"}:
        pack = torch_load(path, map_location="cpu")
        if not isinstance(pack, dict):
            raise RuntimeError(f"Expected a dict-like torch dataset in {path}")
        features = _as_tensor(pack.get("features"), torch.float32)
        if features is None:
            raise RuntimeError("Dataset must contain 'features'.")
        weights = _as_tensor(pack.get("weights"), torch.float32)
        if weights is None:
            weights = torch.ones(features.shape[0], dtype=torch.float32) / max(1, features.shape[0])
        state = _as_tensor(pack.get("meta_state"), torch.long)
        if state is None:
            state = torch.full((features.shape[0],), -1, dtype=torch.long)
        traj_id = _as_tensor(pack.get("traj_id"), torch.long)
        meta = pack.get("meta", {}) or {}
        if traj_id is None and isinstance(meta, dict) and "traj_id" in meta:
            traj_id = _as_tensor(meta["traj_id"], torch.long)
        cv = _as_tensor(pack.get("cv"), torch.float32)
        if cv is None and isinstance(meta, dict) and meta.get("stored_feature_space") == "cv":
            cv = features
        return CommittorDatasetPack(features, weights, state, traj_id, cv, dict(meta), path)

    if ext == ".npz":
        pack = np.load(path, allow_pickle=True)
        if "features" not in pack:
            raise RuntimeError("NPZ dataset must contain 'features'.")
        features = torch.as_tensor(pack["features"], dtype=torch.float32)
        weights = torch.as_tensor(
            pack["weights"] if "weights" in pack else np.ones(features.shape[0]) / max(1, features.shape[0]),
            dtype=torch.float32,
        )
        state = torch.as_tensor(
            pack["meta_state"] if "meta_state" in pack else np.full(features.shape[0], -1),
            dtype=torch.long,
        )
        traj_id = torch.as_tensor(pack["traj_id"], dtype=torch.long) if "traj_id" in pack else None
        cv = torch.as_tensor(pack["cv"], dtype=torch.float32) if "cv" in pack else None
        meta: dict[str, Any] = {}
        if "meta_yaml" in pack and len(pack["meta_yaml"]) > 0:
            import yaml

            meta = yaml.safe_load(str(pack["meta_yaml"][0])) or {}
        return CommittorDatasetPack(features, weights, state, traj_id, cv, meta, path)

    raise ValueError(f"Unsupported dataset format: {path}")


def apply_stride(pack: CommittorDatasetPack, stride: int) -> CommittorDatasetPack:
    stride = int(stride)
    if stride < 1:
        raise ValueError("dataset_stride must be >= 1.")
    if stride == 1:
        return pack
    return CommittorDatasetPack(
        features=pack.features[::stride].contiguous(),
        weights=pack.weights[::stride].contiguous(),
        state=pack.state[::stride].contiguous(),
        traj_id=(pack.traj_id[::stride].contiguous() if pack.traj_id is not None else None),
        cv=(pack.cv[::stride].contiguous() if pack.cv is not None else None),
        meta=dict(pack.meta),
        source_path=pack.source_path,
    )


def cv_headers_for_pack(pack: CommittorDatasetPack) -> list[str]:
    if pack.cv is None:
        return []
    headers = pack.meta.get("cv_headers", None) if isinstance(pack.meta, dict) else None
    if headers is None or len(headers) != pack.cv.shape[1]:
        return [f"cv_{idx}" for idx in range(pack.cv.shape[1])]
    return [str(name) for name in headers]


def featurize_cv_inputs(
    cv: torch.Tensor,
    cv_headers: list[str],
    *,
    cvs_to_use: Any = None,
    periodic_cvs: Any = None,
    periodic_units: str = "degrees",
) -> tuple[torch.Tensor, list[str]]:
    if cv.ndim != 2:
        raise ValueError("cv must have shape (n_frames, n_cv).")
    if len(cv_headers) != cv.shape[1]:
        raise ValueError("cv_headers length must match cv.shape[1].")

    selected = _list_or_empty(cvs_to_use)
    if selected:
        missing = [name for name in selected if name not in cv_headers]
        if missing:
            raise ValueError(f"Requested CV columns are missing from dataset: {missing}")
    else:
        selected = list(cv_headers)

    if periodic_cvs is True:
        periodic = set(selected)
    else:
        periodic = set(_list_or_empty(periodic_cvs))
        missing_periodic = [name for name in periodic if name not in selected]
        if missing_periodic:
            raise ValueError(f"periodic_cvs must be selected CV columns: {missing_periodic}")

    units = str(periodic_units).lower()
    if units not in {"degrees", "degree", "deg", "radians", "radian", "rad"}:
        raise ValueError("periodic_cv_units must be 'degrees' or 'radians'.")

    columns: list[torch.Tensor] = []
    feature_names: list[str] = []
    for name in selected:
        column = cv[:, cv_headers.index(name)].float()
        if name in periodic:
            angle = column if units in {"radians", "radian", "rad"} else column * (math.pi / 180.0)
            columns.extend([torch.sin(angle), torch.cos(angle)])
            feature_names.extend([f"sin{name}", f"cos{name}"])
        else:
            columns.append(column)
            feature_names.append(name)

    if not columns:
        raise ValueError("No CV input columns were selected.")
    return torch.stack(columns, dim=1).contiguous(), feature_names


def select_model_inputs(pack: CommittorDatasetPack, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    input_space = str(
        config.get("model_input_space", config.get("input_space", config.get("feature_space", "features")))
    ).lower()
    if input_space in {"features", "feature", "z"}:
        return pack.features, {
            "model_input_space": "features",
            "model_feature_names": pack.meta.get("feature_names", None) if isinstance(pack.meta, dict) else None,
        }
    if input_space not in {"cv", "cvs", "colvars"}:
        raise ValueError("model_input_space must be 'features' or 'cv'.")
    if pack.cv is None:
        raise RuntimeError("model_input_space='cv' requires a dataset with saved CV data.")

    cv_headers = cv_headers_for_pack(pack)
    selected_cvs = _list_or_empty(config.get("cvs_to_use", config.get("model_cvs", []))) or cv_headers
    periodic_cfg = config.get("periodic_cvs", config.get("periodic", False))
    features, feature_names = featurize_cv_inputs(
        pack.cv,
        cv_headers,
        cvs_to_use=selected_cvs,
        periodic_cvs=periodic_cfg,
        periodic_units=str(config.get("periodic_cv_units", config.get("cv_units", "degrees"))),
    )
    return features, {
        "model_input_space": "cv",
        "model_cv_headers": cv_headers,
        "model_feature_names": feature_names,
        "model_cvs_to_use": selected_cvs,
        "model_periodic_cvs": selected_cvs if periodic_cfg is True else _list_or_empty(periodic_cfg),
        "model_periodic_cv_units": str(config.get("periodic_cv_units", config.get("cv_units", "degrees"))),
    }


def infer_n_states(pack: CommittorDatasetPack, requested: int | None = None) -> int:
    if requested is not None:
        return int(requested)
    if "k_selected" in pack.meta and pack.meta["k_selected"] is not None:
        return int(pack.meta["k_selected"])
    labeled = pack.state[pack.state >= 0]
    if labeled.numel() == 0:
        raise RuntimeError("Cannot infer n_states: provide it in the config or include labeled meta_state values.")
    return int(labeled.max().item()) + 1


def build_lagged_indices(
    n_frames: int,
    lag: int,
    traj_id: torch.Tensor | None = None,
    allow_cross: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    lag = int(lag)
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    if n_frames <= lag:
        raise RuntimeError(f"Need more than lag={lag} frames; got n_frames={n_frames}.")

    idx0 = torch.arange(0, n_frames - lag, dtype=torch.long)
    idx1 = idx0 + lag
    if traj_id is not None and not allow_cross:
        keep = traj_id[idx0] == traj_id[idx1]
        idx0 = idx0[keep]
        idx1 = idx1[keep]
    if idx0.numel() == 0:
        raise RuntimeError("No lagged pairs remain. Reduce lag or allow cross-trajectory pairs.")
    return idx0, idx1


class LaggedCommittorDataset(Dataset):
    """
    Lagged samples for q-vector training.

    Each item provides z_t, z_tau, weight_t, state_t, and state_tau. State labels
    are integers 0..n_states-1 inside metastable states and -1 outside them.
    """

    def __init__(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        state: torch.Tensor,
        lag: int,
        traj_id: torch.Tensor | None = None,
        allow_cross_traj_pairs: bool = False,
        require_labeled: str = "none",
    ):
        if features.ndim != 2:
            raise ValueError("features must have shape (n_frames, feature_dim).")
        if weights.ndim != 1 or weights.shape[0] != features.shape[0]:
            raise ValueError("weights must have shape (n_frames,).")
        if state.ndim != 1 or state.shape[0] != features.shape[0]:
            raise ValueError("state labels must have shape (n_frames,).")
        self.features = features.float()
        self.weights = weights.float()
        self.state = state.long()

        idx0, idx1 = build_lagged_indices(
            features.shape[0],
            lag=lag,
            traj_id=traj_id,
            allow_cross=allow_cross_traj_pairs,
        )

        mode = str(require_labeled).lower()
        if mode not in {"none", "any", "both"}:
            raise ValueError("require_labeled must be one of: none, any, both.")
        if mode != "none":
            labeled0 = self.state[idx0] >= 0
            labeled1 = self.state[idx1] >= 0
            keep = (labeled0 | labeled1) if mode == "any" else (labeled0 & labeled1)
            idx0 = idx0[keep]
            idx1 = idx1[keep]
        if idx0.numel() == 0:
            raise RuntimeError("No lagged training pairs remain after filtering.")

        self.idx_t = idx0
        self.idx_tau = idx1

    def __len__(self) -> int:
        return int(self.idx_t.numel())

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        i = self.idx_t[item]
        j = self.idx_tau[item]
        return {
            "z_t": self.features[i],
            "z_tau": self.features[j],
            "weight": self.weights[i],
            "state_t": self.state[i],
            "state_tau": self.state[j],
        }


def split_train_val(n: int, val_ratio: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise RuntimeError("Need at least two lagged pairs to split train/validation sets.")
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = int(math.ceil(float(val_ratio) * n))
    n_val = min(max(n_val, 1), n - 1)
    return idx[n_val:], idx[:n_val]


class IndexSubset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int):
        return self.base[int(self.indices[item])]


def unordered_pairs(n_states: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(int(n_states)) for j in range(i + 1, int(n_states))]


def pair_labels_from_state(state: torch.Tensor, n_states: int) -> torch.Tensor:
    """
    Build pairwise endpoint labels from label.py meta_state values.

    For each unordered pair (i, j), frames in state i get target 0, frames in
    state j get target 1, and all other frames get -1.
    """
    state = state.long()
    labels = torch.full((state.shape[0], len(unordered_pairs(n_states))), -1, dtype=torch.int8)
    for col, (i, j) in enumerate(unordered_pairs(n_states)):
        labels[state == i, col] = 0
        labels[state == j, col] = 1
    return labels


class PairwiseLaggedDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
        pair_labels: torch.Tensor,
        lag: int,
        traj_id: torch.Tensor | None = None,
        allow_cross_traj_pairs: bool = False,
        require_labeled: str = "both",
    ):
        if features.ndim != 2:
            raise ValueError("features must have shape (n_frames, feature_dim).")
        if weights.ndim != 1 or weights.shape[0] != features.shape[0]:
            raise ValueError("weights must have shape (n_frames,).")
        if pair_labels.ndim != 2 or pair_labels.shape[0] != features.shape[0]:
            raise ValueError("pair_labels must have shape (n_frames, n_pairs).")
        self.features = features.float()
        self.weights = weights.float()
        self.pair_labels = pair_labels.to(torch.int8)

        idx0, idx1 = build_lagged_indices(
            features.shape[0],
            lag=lag,
            traj_id=traj_id,
            allow_cross=allow_cross_traj_pairs,
        )

        mode = str(require_labeled).lower()
        if mode not in {"none", "any", "both"}:
            raise ValueError("require_labeled must be one of: none, any, both.")
        if mode != "none":
            labeled0 = (self.pair_labels[idx0] >= 0).any(dim=1)
            labeled1 = (self.pair_labels[idx1] >= 0).any(dim=1)
            keep = (labeled0 | labeled1) if mode == "any" else (labeled0 & labeled1)
            idx0 = idx0[keep]
            idx1 = idx1[keep]
        if idx0.numel() == 0:
            raise RuntimeError("No lagged VCN training pairs remain after filtering.")

        self.idx_t = idx0
        self.idx_tau = idx1

    def __len__(self) -> int:
        return int(self.idx_t.numel())

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        i = self.idx_t[item]
        j = self.idx_tau[item]
        return {
            "z_t": self.features[i],
            "z_tau": self.features[j],
            "weight": self.weights[i],
            "pair_label_t": self.pair_labels[i],
            "pair_label_tau": self.pair_labels[j],
        }

from __future__ import annotations

import os

import numpy as np

from ..common.config import ensure_dir


def _entropy_confidence(q_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    q = np.clip(q_values, eps, 1.0)
    q_max = np.max(q, axis=1)
    q_argmax = np.argmax(q, axis=1).astype(np.int64)
    entropy = -np.sum(q * np.log(q), axis=1)
    entropy_norm = entropy / np.log(q.shape[1])
    return q_max, q_argmax, entropy, entropy_norm


def _compact_nonnegative_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    compact = labels.copy()
    valid_labels = sorted(int(label) for label in np.unique(labels[labels >= 0]))
    mapping = {old: new for new, old in enumerate(valid_labels)}
    for old, new in mapping.items():
        compact[labels == old] = new
    rows = [{"old_label": int(old), "new_label": int(new)} for old, new in mapping.items()]
    return compact, rows


def _save_dataset_like_input(dataset_path, output_path, pack, new_state, config, stride):
    import torch
    import yaml

    dataset_path = str(dataset_path)
    output_path = str(output_path)
    ext = os.path.splitext(dataset_path)[1].lower()
    out_ext = os.path.splitext(output_path)[1].lower() or ext
    ensure_dir(os.path.dirname(output_path) or ".")

    meta = dict(pack.meta)
    section_cfg = config.get("relabel", {})
    compact_labels = bool(section_cfg.get("compact_labels", config.get("compact_labels", True)))
    save_state = np.asarray(new_state, dtype=np.int64)
    label_mapping = []
    if compact_labels:
        save_state, label_mapping = _compact_nonnegative_labels(save_state)
    valid_labeled = save_state[save_state >= 0]
    relabeled_n_states = int(np.max(valid_labeled) + 1) if valid_labeled.size else 0
    if relabeled_n_states > 0:
        meta["k_selected"] = relabeled_n_states
        meta["n_states"] = relabeled_n_states
    meta["relabel"] = {
        "source_dataset": os.path.abspath(dataset_path),
        "dataset_stride": int(stride),
        "config": section_cfg,
        "compact_labels": compact_labels,
        "label_mapping": label_mapping,
        "n_states": relabeled_n_states,
    }

    if out_ext in {".pt", ".pth"}:
        out = {
            "features": pack.features.detach().cpu().float(),
            "weights": pack.weights.detach().cpu().float(),
            "meta_state": torch.as_tensor(save_state, dtype=torch.long),
            "meta": meta,
        }
        if pack.cv is not None:
            out["cv"] = pack.cv.detach().cpu().float()
        if pack.traj_id is not None:
            out["traj_id"] = pack.traj_id.detach().cpu().long()
        torch.save(out, output_path)
        return output_path

    if out_ext == ".npz":
        out = {
            "features": pack.features.detach().cpu().numpy().astype(np.float32),
            "weights": pack.weights.detach().cpu().numpy().astype(np.float32),
            "meta_state": save_state.astype(np.int64, copy=False),
            "meta_yaml": np.array([yaml.safe_dump(meta, sort_keys=False)], dtype=object),
        }
        if pack.cv is not None:
            out["cv"] = pack.cv.detach().cpu().numpy().astype(np.float32)
        if pack.traj_id is not None:
            out["traj_id"] = pack.traj_id.detach().cpu().numpy().astype(np.int64)
        np.savez_compressed(output_path, **out)
        return output_path

    raise ValueError("Relabeled dataset output must end in .pt, .pth, or .npz.")

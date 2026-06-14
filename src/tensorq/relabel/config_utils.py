from __future__ import annotations

import numpy as np

def _relabel_cfg(config):
    if "relabel" in config and isinstance(config["relabel"], dict):
        return config["relabel"]
    return {}

def _standardize_features(x):
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std

def _sample_indices(indices, scores, max_count, seed):
    indices = np.asarray(indices, dtype=np.int64)
    max_count = int(max_count)
    if max_count <= 0 or indices.size <= max_count:
        return indices

    scores = np.asarray(scores, dtype=np.float64)
    scores = np.clip(scores, 0.0, np.inf)
    if np.sum(scores) <= 0.0:
        prob = None
    else:
        prob = scores / np.sum(scores)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, size=max_count, replace=False, p=prob))

def _select_graph_features(pack, model_features, config):
    relabel_cfg = _relabel_cfg(config)
    space = str(relabel_cfg.get("graph_space", "auto")).lower()
    if space == "auto":
        model_space = str(config.get("model_input_space", "")).lower()
        space = "features" if model_space in {"features", "feature", "model_features", "model"} else "cv"
    if space in {"cv", "cvs", "colvars"}:
        if pack.cv is None:
            raise RuntimeError("relabel.graph_space='cv' requires saved CV data.")
        return pack.cv.detach().cpu().numpy(), "cv"
    if space in {"features", "feature", "model_features", "model"}:
        return model_features.detach().cpu().numpy(), "model_features"
    raise ValueError("relabel.graph_space must be 'cv' or 'model_features'.")

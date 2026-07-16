from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .checkpoint import exists, save_npz
from .config import ensure_dir, stage_path

# Module-level cache for large feature matrices to avoid redundant disk loads.
# Keyed by (path, key) tuples.  See _load_external_feature_matrix().
_feature_cache: Dict[Tuple[str, str], Tuple[np.ndarray, Dict[str, Any]]] = {}


def read_colvars_traj(path: str) -> pd.DataFrame:
    colnames = None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                tokens = stripped.lstrip("#").strip().split()
                if len(tokens) >= 2 and all("=" not in token for token in tokens):
                    colnames = tokens
            else:
                break
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if colnames is not None and len(colnames) == df.shape[1]:
        df.columns = colnames
    else:
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    return df


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def load_torch_dataset(path: str) -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"Reading TensorQ .pt dataset '{path}' requires torch.") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _metadata_feature_headers(meta: Any, n_features: int) -> List[str] | None:
    if not isinstance(meta, dict):
        return None
    names = meta.get("feature_headers", meta.get("feature_names", meta.get("model_feature_names", None)))
    if names is None and str(meta.get("stored_feature_space", "")).lower() == "cv":
        names = meta.get("cv_headers", None)
    if names is None:
        return None
    names = [str(name) for name in list(names)]
    return names if len(names) == n_features else None


def _tensorq_table_matrix(pack: Dict[str, Any], data_cfg: Dict[str, Any], path: str) -> tuple[np.ndarray, str]:
    key = data_cfg.get("tensor_key", data_cfg.get("pt_table_key", None))
    if key is not None:
        key = str(key)
        if key not in pack:
            raise SystemExit(f"TensorQ .pt dataset {path} does not contain configured table key '{key}'.")
        return _tensor_to_numpy(pack[key]).astype(np.float64), key

    if "cv" in pack:
        return _tensor_to_numpy(pack["cv"]).astype(np.float64), "cv"

    meta = pack.get("meta", {})
    if "features" in pack and isinstance(meta, dict) and str(meta.get("stored_feature_space", "")).lower() == "cv":
        return _tensor_to_numpy(pack["features"]).astype(np.float64), "features"

    raise SystemExit(
        f"TensorQ .pt dataset is missing 'cv': {path}. Recreate it with label.py save_cv: true, "
        "set data.pt_table_key/data.tensor_key to a saved 2D tensor, or use a dataset whose "
        "meta.stored_feature_space is 'cv'."
    )


def _tensorq_table_headers(
    meta: Any,
    data_cfg: Dict[str, Any],
    n_columns: int,
    tensor_key: str,
) -> List[str]:
    headers = list(data_cfg.get("cv_columns", []))
    if not headers and isinstance(meta, dict):
        headers = [str(name) for name in list(meta.get("cv_headers", []))]
    if not headers and tensor_key == "features":
        inferred = _metadata_feature_headers(meta, n_columns)
        if inferred is not None:
            headers = inferred
    if not headers:
        headers = [str(name) for name in list(data_cfg.get("cvs", []))]
    if len(headers) != n_columns:
        raise SystemExit(
            f"TensorQ .pt dataset tensor '{tensor_key}' has {n_columns} columns, but {len(headers)} CV names "
            "were found. Set data.cv_columns to the saved table columns, or set data.pt_table_key to "
            "the tensor you want to expose as the frame table."
        )
    return [str(name) for name in headers]


def read_tensorq_dataset_table(path: str, data_cfg: Dict[str, Any]) -> pd.DataFrame:
    pack = load_torch_dataset(path)
    if not isinstance(pack, dict):
        raise SystemExit(f"TensorQ .pt dataset must contain a dict, got {type(pack).__name__}: {path}")
    meta = pack.get("meta", {})
    cv, tensor_key = _tensorq_table_matrix(pack, data_cfg, path)
    if cv.ndim == 1:
        cv = cv.reshape(-1, 1)
    if cv.ndim != 2:
        raise SystemExit(f"TensorQ .pt dataset tensor '{tensor_key}' must be 2D, got shape {cv.shape}: {path}")
    cv_headers = _tensorq_table_headers(meta, data_cfg, cv.shape[1], tensor_key)

    df = pd.DataFrame(cv, columns=cv_headers)
    weight_column = data_cfg.get("weight_column", "weight")
    if "weights" in pack:
        weights = _tensor_to_numpy(pack["weights"]).reshape(-1).astype(np.float64)
        if weights.shape[0] != cv.shape[0]:
            raise SystemExit(f"TensorQ .pt weights rows ({weights.shape[0]}) != cv rows ({cv.shape[0]}): {path}")
        df[weight_column] = weights
    else:
        df[weight_column] = 1.0

    if "traj_id" in pack:
        traj_id = _tensor_to_numpy(pack["traj_id"]).reshape(-1)
        if traj_id.shape[0] != cv.shape[0]:
            raise SystemExit(f"TensorQ .pt traj_id rows ({traj_id.shape[0]}) != cv rows ({cv.shape[0]}): {path}")
        df.insert(0, "traj_id", traj_id.astype(str))
    if "meta_state" in pack:
        meta_state = _tensor_to_numpy(pack["meta_state"]).reshape(-1)
        if meta_state.shape[0] == cv.shape[0]:
            df["meta_state"] = meta_state.astype(np.int64)
    return df


def _iter_source_tables(data_cfg: Dict[str, Any]) -> Iterable[Tuple[str, pd.DataFrame, str]]:
    for item in data_cfg.get("tables", []):
        if isinstance(item, str):
            path = item
            traj_id = os.path.splitext(os.path.basename(path))[0]
        else:
            path = item["path"]
            traj_id = str(item.get("traj_id", os.path.splitext(os.path.basename(path))[0]))
        if os.path.splitext(path)[1].lower() in {".pt", ".pth"}:
            yield path, read_tensorq_dataset_table(path, data_cfg), traj_id
        else:
            yield path, pd.read_csv(path), traj_id

    for item in data_cfg.get("colvars", []):
        if isinstance(item, str):
            path = item
            traj_id = os.path.splitext(os.path.basename(path))[0]
        else:
            path = item["path"]
            traj_id = str(item.get("traj_id", os.path.splitext(os.path.basename(path))[0]))
        yield path, read_colvars_traj(path), traj_id

    for root in data_cfg.get("folders", []):
        pattern = data_cfg.get("match_colvars", "*.colvars.traj")
        tag_re = re.compile(data_cfg.get("tag_regex", r"(.+)"))
        for path in sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True)):
            rel = os.path.relpath(path, root)
            stem = os.path.splitext(os.path.basename(path))[0]
            match = tag_re.search(stem)
            tag = match.group(1) if match and match.lastindex else stem
            yield path, read_colvars_traj(path), os.path.join(os.path.dirname(rel), tag)


def _assign_traj_metadata(df: pd.DataFrame, traj_id: str, source: str, frame_offset: int) -> pd.DataFrame:
    out = df.copy()
    if "global_frame" in out.columns:
        out = out.drop(columns=["global_frame"])
    if "traj_id" not in out.columns:
        out.insert(0, "traj_id", traj_id)
    if "frame_in_traj" not in out.columns:
        out.insert(1, "frame_in_traj", np.arange(len(out), dtype=np.int64))
    if "source_file" not in out.columns:
        out.insert(2, "source_file", source)
    out.insert(0, "global_frame", np.arange(frame_offset, frame_offset + len(out), dtype=np.int64))
    return out


def _infer_reset_blocks(df: pd.DataFrame, step_column: str) -> np.ndarray | None:
    if step_column not in df.columns:
        return None
    steps = pd.to_numeric(df[step_column], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(steps)
    if not np.all(finite):
        return None
    reset = np.r_[False, np.diff(steps) < 0.0]
    if not np.any(reset):
        return None
    return np.cumsum(reset).astype(np.int64)


def _fixed_length_blocks(n_rows: int, frames_per_traj: int) -> np.ndarray:
    idx = np.arange(n_rows, dtype=np.int64)
    return idx // int(frames_per_traj)


def _feature_names(n_features: int, configured: Any = None) -> List[str]:
    if configured is None:
        return [f"feature_{idx}" for idx in range(n_features)]
    names = [str(name) for name in list(configured)]
    if len(names) != n_features:
        raise SystemExit(
            f"data.model.feature_columns has {len(names)} names, but the feature matrix has {n_features} columns."
        )
    if len(set(names)) != len(names):
        raise SystemExit("data.model.feature_columns must be unique.")
    return names


def _load_external_feature_matrix(model_cfg: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    path = model_cfg.get("path", model_cfg.get("features_path", None))
    if path is None:
        raise SystemExit("data.model.source='features' requires data.model.path.")
    path = str(path)
    if not os.path.exists(path):
        raise SystemExit(f"Feature matrix file does not exist: {path}")

    suffix = os.path.splitext(path)[1].lower()
    default_key = "features" if suffix in {".pt", ".pth"} else "X"
    cache_key = (path, str(model_cfg.get("key", default_key)))
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]

    meta: Dict[str, Any] = {"path": path}
    if suffix in {".pt", ".pth"}:
        key = str(model_cfg.get("key", "features"))
        pack = load_torch_dataset(path)
        if not isinstance(pack, dict):
            raise SystemExit(f"TensorQ .pt dataset must contain a dict, got {type(pack).__name__}: {path}")
        if key not in pack:
            raise SystemExit(f"TensorQ .pt dataset {path} does not contain key '{key}'.")
        X = _tensor_to_numpy(pack[key]).astype(np.float32)
        meta_obj = pack.get("meta", {})
        if isinstance(meta_obj, dict):
            meta["tensorq_meta"] = meta_obj
            inferred = _metadata_feature_headers(meta_obj, X.shape[1] if X.ndim == 2 else 1)
            if inferred is not None and "feature_columns" not in model_cfg and "columns" not in model_cfg:
                meta["inferred_feature_columns"] = inferred
    elif suffix == ".npz":
        key = str(model_cfg.get("key", "X"))
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise SystemExit(f"Feature matrix file {path} does not contain key '{key}'.")
            X = np.asarray(data[key], dtype=np.float32)
            if "meta" in data:
                try:
                    meta["riteweight_meta"] = json.loads(str(data["meta"]))
                except Exception:
                    meta["riteweight_meta"] = str(data["meta"])
    elif suffix == ".npy":
        # Uncompressed .npy – use memory-mapping for instant load without RAM allocation.
        X = np.load(path, mmap_mode="r")
        if X.dtype != np.float32:
            X = np.asarray(X, dtype=np.float32)
        # Look for a sidecar JSON file with metadata (e.g. from npz→npy conversion).
        meta_path = path.replace(".npy", "_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as mh:
                    meta["riteweight_meta"] = mh.read()
            except Exception:
                pass
    else:
        first = ""
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            first = handle.readline().strip()
        skiprows = 0
        if first.startswith("META="):
            skiprows = 1
            try:
                meta["riteweight_meta"] = json.loads(first[len("META="):])
            except Exception:
                meta["riteweight_meta"] = first[len("META="):]
        X = np.loadtxt(path, delimiter=",", comments="#", skiprows=skiprows, dtype=np.float32)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise SystemExit(f"Feature matrix must be 2D, got shape {X.shape} from {path}.")
    if not bool(model_cfg.get("skip_finite_check", False)):
        if not np.all(np.isfinite(X)):
            raise SystemExit(f"Feature matrix contains NaN or infinite values: {path}")
    meta["n_features_raw"] = int(X.shape[1])
    _feature_cache[cache_key] = (X, meta)
    return X, meta


def _select_feature_dimensions(
    X: np.ndarray,
    names: List[str],
    dimensions: Any,
) -> tuple[np.ndarray, List[str], List[int]]:
    if dimensions is None or dimensions == "all":
        idx = np.arange(X.shape[1], dtype=np.int64)
    else:
        if isinstance(dimensions, (str, int)):
            dimensions = [dimensions]
        selected: List[int] = []
        by_name = {name: i for i, name in enumerate(names)}
        for dim in list(dimensions):
            if isinstance(dim, str) and dim in by_name:
                selected.append(by_name[dim])
            else:
                try:
                    selected.append(int(dim))
                except Exception as exc:
                    raise SystemExit(f"Feature dimension '{dim}' is neither a feature name nor an integer index.") from exc
        idx = np.asarray(selected, dtype=np.int64)
        if idx.size == 0:
            raise SystemExit("data.model.dimensions selected no feature dimensions.")
        if np.any(idx < 0) or np.any(idx >= X.shape[1]):
            raise SystemExit(f"data.model.dimensions contains indices outside [0, {X.shape[1] - 1}].")
    idx_list = [int(i) for i in idx.tolist()]
    return X[:, idx].astype(np.float32, copy=False), [names[i] for i in idx_list], idx_list


def external_feature_coordinate_matrix(
    model_cfg: Dict[str, Any],
    table: pd.DataFrame,
    *,
    context: str,
) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    X_all, meta = _load_external_feature_matrix(model_cfg)
    if X_all.shape[0] != len(table):
        raise SystemExit(
            f"{context} feature matrix rows ({X_all.shape[0]}) != frame table rows ({len(table)}). "
            "Use a feature cache aligned to the same concatenated, strided frame table."
        )
    configured_names = model_cfg.get("columns", model_cfg.get("feature_columns", meta.get("inferred_feature_columns", None)))
    names = _feature_names(X_all.shape[1], configured_names)
    X, selected_names, selected_idx = _select_feature_dimensions(
        X_all,
        names,
        model_cfg.get("dimensions", model_cfg.get("feature_dimensions", "all")),
    )
    feature_meta = dict(meta)
    feature_meta.update(
        {
            "source": "features",
            "raw_columns": names,
            "selected_dimensions": selected_idx,
        }
    )
    return X, selected_names, feature_meta


def modeling_coordinate_matrix(cfg: Dict[str, Any], table: pd.DataFrame) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    data_cfg = cfg["data"]
    model_cfg = data_cfg.get("model", {})
    if model_cfg is None:
        model_cfg = {}
    source = str(model_cfg.get("source", data_cfg.get("model_source", "cvs"))).lower()
    dimensions = model_cfg.get("dimensions", model_cfg.get("feature_dimensions", "all"))

    if source in ("cv", "cvs", "table"):
        names = [str(name) for name in model_cfg.get("columns", data_cfg["cvs"])]
        missing = [name for name in names if name not in table.columns]
        if missing:
            raise SystemExit(f"Model CV columns are missing from the input data: {missing}")
        X_all = table[names].to_numpy(dtype=np.float64)
        X, selected_names, selected_idx = _select_feature_dimensions(X_all, names, dimensions)
        return X, selected_names, {
            "source": "cvs",
            "raw_columns": names,
            "selected_dimensions": selected_idx,
        }

    if source in ("feature", "features", "zmatrix", "internal_zmat"):
        return external_feature_coordinate_matrix(model_cfg, table, context="data.model")

    raise SystemExit("data.model.source must be 'cvs' or 'features'.")


def core_coordinate_matrix(cfg: Dict[str, Any], table: pd.DataFrame) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    label_cfg = cfg.get("core_labeling", {})
    source = str(label_cfg.get("feature_source", "model")).lower()
    if source == "model":
        return modeling_coordinate_matrix(cfg, table)
    if source in ("cv", "cvs"):
        names = list(label_cfg.get("feature_cvs", cfg["data"]["cvs"]))
        missing = [name for name in names if name not in table.columns]
        if missing:
            raise SystemExit(f"core_labeling.feature_cvs contains missing columns: {missing}")
        return table[names].to_numpy(dtype=np.float64), names, {"source": "cvs", "raw_columns": names}
    if source in ("feature", "features", "zmatrix", "internal_zmat"):
        model_cfg = dict(cfg["data"].get("model", {}))
        model_cfg.update(label_cfg.get("features", {}))
        return external_feature_coordinate_matrix(model_cfg, table, context="core_labeling.features")
    raise SystemExit("core_labeling.feature_source must be 'model', 'cvs', or 'features'.")


def output_feature_coordinate_matrix(
    cfg: Dict[str, Any],
    table: pd.DataFrame,
    label_matrix: np.ndarray,
    label_headers: List[str],
    label_meta: Dict[str, Any],
) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    label_cfg = cfg.get("core_labeling", {})
    output_cfg = label_cfg.get("output_features", label_cfg.get("dataset_features", None))
    if output_cfg is None:
        return label_matrix, label_headers, dict(label_meta)
    if isinstance(output_cfg, str):
        output_cfg = {"source": output_cfg}
    if not isinstance(output_cfg, dict):
        raise SystemExit("core_labeling.output_features must be a mapping or one of: label, model, cvs, features.")

    source = str(output_cfg.get("source", output_cfg.get("feature_source", "label"))).lower()
    if source in ("label", "core", "core_labeling"):
        return label_matrix, label_headers, dict(label_meta)
    if source == "model":
        return modeling_coordinate_matrix(cfg, table)
    if source in ("cv", "cvs", "table"):
        names = [str(name) for name in output_cfg.get("columns", output_cfg.get("feature_cvs", cfg["data"]["cvs"]))]
        missing = [name for name in names if name not in table.columns]
        if missing:
            raise SystemExit(f"core_labeling.output_features feature_cvs contains missing columns: {missing}")
        return table[names].to_numpy(dtype=np.float64), names, {"source": "cvs", "raw_columns": names}
    if source in ("feature", "features", "zmatrix", "internal_zmat"):
        model_cfg = dict(cfg["data"].get("model", {}))
        model_cfg.update({key: value for key, value in output_cfg.items() if key != "features"})
        model_cfg.update(output_cfg.get("features", {}))
        model_cfg["source"] = "features"
        return external_feature_coordinate_matrix(model_cfg, table, context="core_labeling.output_features")
    raise SystemExit("core_labeling.output_features.source must be 'label', 'model', 'cvs', or 'features'.")


def source_trajectory_blocks(
    df: pd.DataFrame,
    *,
    source: str,
    default_traj_id: str,
    data_cfg: Dict[str, Any],
) -> Iterable[Tuple[str, pd.DataFrame]]:
    split_cfg = data_cfg.get("infer_trajectories", {})
    if split_cfg is None:
        split_cfg = {}
    if isinstance(split_cfg, bool):
        split_cfg = {"enabled": split_cfg}
    enabled = bool(split_cfg.get("enabled", True))
    step_column = split_cfg.get("step_column", data_cfg.get("step_column", "step"))

    if "traj_id" in df.columns:
        for traj_id, block in df.groupby("traj_id", sort=False):
            yield str(traj_id), block.reset_index(drop=True)
        return

    block_ids = None
    method = "single_table"
    if enabled:
        frames_per_traj = split_cfg.get("frames_per_traj", data_cfg.get("frames_per_traj", None))
        if frames_per_traj is not None:
            frames_per_traj = int(frames_per_traj)
            if frames_per_traj <= 0:
                raise SystemExit("infer_trajectories.frames_per_traj must be positive.")
            block_ids = _fixed_length_blocks(len(df), frames_per_traj)
            method = f"fixed_length_{frames_per_traj}"
        else:
            block_ids = _infer_reset_blocks(df, step_column)
            method = f"{step_column}_reset"

    if block_ids is None:
        print(
            f"[warn] Treating {source} as one trajectory. If this is a concatenated table, "
            "set data.infer_trajectories.enabled=true with a resetting step column or frames_per_traj."
        )
        yield default_traj_id, df.reset_index(drop=True)
        return

    n_blocks = int(np.max(block_ids)) + 1
    print(f"[info] Split {source} into {n_blocks} trajectories by {method}.")
    for block_id in range(n_blocks):
        block = df[block_ids == block_id].reset_index(drop=True)
        if len(block) == 0:
            continue
        yield f"{default_traj_id}:{block_id:06d}", block


def load_frame_table(cfg: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg["data"]
    stride = int(data_cfg.get("stride", 1))
    discard = data_cfg.get("discard_before_step", None)
    step_column = data_cfg.get("step_column", "step")

    frames: List[pd.DataFrame] = []
    offset = 0
    n_trajectories = 0
    for source, df, traj_id in _iter_source_tables(data_cfg):
        if discard is not None and step_column in df.columns:
            df = df[pd.to_numeric(df[step_column], errors="coerce") >= float(discard)].reset_index(drop=True)
        for block_traj_id, block in source_trajectory_blocks(
            df,
            source=source,
            default_traj_id=traj_id,
            data_cfg=data_cfg,
        ):
            if stride > 1:
                block = block.iloc[::stride].reset_index(drop=True)
            if len(block) == 0:
                continue
            out = _assign_traj_metadata(block, traj_id=block_traj_id, source=source, frame_offset=offset)
            frames.append(out)
            offset += len(out)
            n_trajectories += 1

    if not frames:
        raise SystemExit("No input frames found. Configure data.tables, data.colvars, or data.folders.")

    table = pd.concat(frames, ignore_index=True)
    for cv in data_cfg["cvs"]:
        if cv not in table.columns:
            raise SystemExit(f"Required CV column '{cv}' is missing from the input data.")

    weight_column = data_cfg.get("weight_column", "weight")
    if weight_column not in table.columns:
        table[weight_column] = 1.0
    table[weight_column] = pd.to_numeric(table[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(table[weight_column].sum()) <= 0.0:
        table[weight_column] = 1.0
    print(f"[info] Prepared {len(table)} frames from {n_trajectories} trajectories.")
    return table


def prepare_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    force = bool(cfg["project"].get("force", False))
    out_csv = stage_path(cfg, "01_data", "frame_table.csv.gz")
    if exists(out_csv, force=force):
        print(f"[reuse] data: {out_csv}")
        return pd.read_csv(out_csv)

    out_dir = ensure_dir(stage_path(cfg, "01_data"))
    table = load_frame_table(cfg)
    cvs = cfg["data"]["cvs"]
    # Load features once (cached for later stages) and validate row count.
    X, feature_names, feature_meta = modeling_coordinate_matrix(cfg, table)
    weight_column = cfg["data"].get("weight_column", "weight")
    weights = table[weight_column].to_numpy(dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-300)

    table.to_csv(out_csv, index=False)
    # Save only lightweight arrays – the features matrix stays at its source path
    # and is kept in the module-level _feature_cache for downstream stages.
    save_npz(
        out_csv.replace(".csv.gz", ".npz").replace(".csv", ".npz"),
        weights=weights,
        global_frame=table["global_frame"].to_numpy(dtype=np.int64),
        manifest={
            "stage": "data",
            "n_frames": len(table),
            "n_trajectories": int(table["traj_id"].nunique()),
            "cvs": cvs,
            "model_feature_names": feature_names,
            "model_feature_source": feature_meta.get("source", "cvs"),
            "model_feature_meta": feature_meta,
            "out_dir": out_dir,
        },
    )
    print(f"[ok] data: {out_csv}")
    return table

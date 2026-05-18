from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .core import as_points


@dataclass(frozen=True)
class SampleData:
    points: np.ndarray
    weights: np.ndarray | None = None
    traj_id: np.ndarray | None = None


def _load_plain_array(path: str, *, key: str | None = None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        pack = np.load(path, allow_pickle=True)
        if key is not None:
            if key not in pack:
                raise KeyError(f"Key {key!r} not found in {path}.")
            return pack[key]
        for candidate in ("points", "samples", "centers", "images", "cv", "features", "arr_0"):
            if candidate in pack:
                return pack[candidate]
        raise KeyError(f"No default array key found in {path}. Provide a key.")
    delimiter = "," if ext == ".csv" else None
    return np.loadtxt(path, delimiter=delimiter)


def _load_optional_array(path: str | None, *, key: str | None = None) -> np.ndarray | None:
    if path is None:
        return None
    return np.asarray(_load_plain_array(str(path), key=key))


def load_samples(
    path: str,
    config: dict[str, Any],
    *,
    key: str | None = None,
    stride: int = 1,
    weights_path: str | None = None,
    weights_key: str | None = None,
    traj_id_path: str | None = None,
    traj_id_key: str | None = None,
) -> SampleData:
    ext = os.path.splitext(str(path))[1].lower()
    if ext in {".pt", ".pth"}:
        from ..common.data import apply_stride, load_dataset, select_model_inputs

        pack = apply_stride(load_dataset(path), int(stride))
        features, _meta = select_model_inputs(pack, config)
        points = features.detach().cpu().double().numpy()
        weights = pack.weights.detach().cpu().double().numpy()
        traj_id = None if pack.traj_id is None else pack.traj_id.detach().cpu().numpy()
        return SampleData(points=as_points(points, name="samples"), weights=weights, traj_id=traj_id)

    if ext == ".npz" and key is None:
        try:
            from ..common.data import apply_stride, load_dataset, select_model_inputs

            pack = apply_stride(load_dataset(path), int(stride))
        except Exception:
            pack = None
        if pack is not None:
            features, _meta = select_model_inputs(pack, config)
            points = features.detach().cpu().double().numpy()
            weights = pack.weights.detach().cpu().double().numpy()
            traj_id = None if pack.traj_id is None else pack.traj_id.detach().cpu().numpy()
            return SampleData(points=as_points(points, name="samples"), weights=weights, traj_id=traj_id)

    points = as_points(_load_plain_array(str(path), key=key), name="samples")
    if int(stride) > 1:
        points = points[:: int(stride)]
    weights = _load_optional_array(weights_path, key=weights_key)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if int(stride) > 1:
            weights = weights[:: int(stride)]
    traj_id = _load_optional_array(traj_id_path, key=traj_id_key)
    if traj_id is not None:
        traj_id = np.asarray(traj_id).reshape(-1)
        if int(stride) > 1:
            traj_id = traj_id[:: int(stride)]
    return SampleData(points=points, weights=weights, traj_id=traj_id)


def _iter_image_files(path: str, kind: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    patterns: list[str]
    kind_l = str(kind).lower()
    if kind_l == "center":
        patterns = ["cluster_*_center_path.txt", "*_center_path.txt"]
    elif kind_l == "medoid":
        patterns = ["cluster_*_medoid_path.txt", "*_medoid_path.txt"]
    else:
        patterns = ["*.txt", "*.dat", "*.csv", "*.npy", "*.npz"]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(path, pattern)))
    return sorted(dict.fromkeys(files))


def _even_indices(n_items: int, n_keep: int) -> np.ndarray:
    if n_items <= 0:
        raise ValueError("n_items must be positive.")
    keep = max(1, min(int(n_keep), int(n_items)))
    if keep == n_items:
        return np.arange(n_items, dtype=np.int64)
    return np.unique(np.linspace(0, n_items - 1, keep).round().astype(np.int64))


def coarsen_path_images(
    path: np.ndarray,
    *,
    image_stride: int = 1,
    max_images_per_path: int | None = None,
    num_images_per_path: int | None = None,
) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("path images must have shape (n_images, n_dim).")
    stride = max(1, int(image_stride))
    if stride > 1:
        idx = np.arange(0, arr.shape[0], stride, dtype=np.int64)
        if idx[-1] != arr.shape[0] - 1:
            idx = np.append(idx, arr.shape[0] - 1)
        arr = arr[idx]
    if num_images_per_path is not None:
        arr = arr[_even_indices(arr.shape[0], int(num_images_per_path))]
    elif max_images_per_path is not None and arr.shape[0] > int(max_images_per_path):
        arr = arr[_even_indices(arr.shape[0], int(max_images_per_path))]
    return arr


def load_images(
    paths: str | Iterable[str],
    *,
    kind: str = "center",
    key: str | None = None,
    flatten: bool = True,
    image_stride: int = 1,
    max_images_per_path: int | None = None,
    num_images_per_path: int | None = None,
) -> np.ndarray | list[np.ndarray]:
    raw_paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    files: list[str] = []
    for item in raw_paths:
        files.extend(_iter_image_files(str(item), kind))
    if not files:
        raise RuntimeError(f"No Voronoi image files found for {raw_paths!r}.")

    all_paths: list[np.ndarray] = []
    for file_path in files:
        arr = np.asarray(_load_plain_array(file_path, key=key), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2:
            arr = coarsen_path_images(
                arr,
                image_stride=image_stride,
                max_images_per_path=max_images_per_path,
                num_images_per_path=num_images_per_path,
            )
            all_paths.append(arr)
        elif arr.ndim == 3:
            coarse = [
                coarsen_path_images(
                    item,
                    image_stride=image_stride,
                    max_images_per_path=max_images_per_path,
                    num_images_per_path=num_images_per_path,
                )
                for item in arr
            ]
            all_paths.extend(coarse)
        else:
            raise ValueError(f"Unsupported image shape {arr.shape} in {file_path}.")
    if not all_paths:
        raise RuntimeError(f"No valid pathway images loaded from {raw_paths!r}.")
    if flatten:
        return as_points(np.vstack(all_paths), name="images")
    ndim = all_paths[0].shape[-1]
    for p in all_paths:
        if p.ndim != 2:
            raise ValueError(f"Each pathway image must be 2D (n_images, n_dim), got {p.shape}.")
        if p.shape[-1] != ndim:
            raise ValueError(f"All pathway images must have the same dimension, got {p.shape[-1]} and {ndim}.")
    return all_paths

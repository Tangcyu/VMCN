from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..common.config import ensure_dir, load_yaml, select_section, write_yaml
from ..common.data import load_dataset


def require_mdanalysis():
    try:
        from MDAnalysis import Universe
        from MDAnalysis.coordinates.DCD import DCDWriter
        from MDAnalysis.lib.distances import calc_angles, calc_bonds, calc_dihedrals
    except Exception as exc:
        raise RuntimeError(
            "label.py needs MDAnalysis for DCD feature extraction. Install MDAnalysis or use "
            "feature_space='cv' with existing colvars-only features."
        ) from exc
    return Universe, DCDWriter, calc_bonds, calc_angles, calc_dihedrals


def require_sklearn():
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("label.py needs scikit-learn for KMeans/PCA/standardization.") from exc
    return KMeans, PCA, StandardScaler


def require_scipy_eig():
    try:
        from scipy.linalg import eig
    except Exception as exc:
        raise RuntimeError("RiteWeight needs scipy.linalg.eig.") from exc
    return eig


def build_min_zmatrix_indices(n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_atoms < 4:
        raise ValueError("Need at least 4 atoms for a minimal internal-coordinate feature set.")
    bonds = np.array([[i, i - 1] for i in range(1, n_atoms)], dtype=np.int32)
    angles = np.array([[i, i - 1, i - 2] for i in range(2, n_atoms)], dtype=np.int32)
    dihedrals = np.array([[i, i - 1, i - 2, i - 3] for i in range(3, n_atoms)], dtype=np.int32)
    return bonds, angles, dihedrals


def internal_coords_min_zmatrix(positions: np.ndarray, bonds, angles, dihedrals) -> np.ndarray:
    _, _, calc_bonds, calc_angles, calc_dihedrals = require_mdanalysis()
    b = calc_bonds(positions[bonds[:, 0]], positions[bonds[:, 1]]).astype(np.float32)
    a = calc_angles(positions[angles[:, 0]], positions[angles[:, 1]], positions[angles[:, 2]]).astype(np.float32)
    d = calc_dihedrals(
        positions[dihedrals[:, 0]],
        positions[dihedrals[:, 1]],
        positions[dihedrals[:, 2]],
        positions[dihedrals[:, 3]],
    ).astype(np.float32)
    return np.concatenate(
        [
            b,
            np.sin(a).astype(np.float32),
            np.cos(a).astype(np.float32),
            np.sin(d).astype(np.float32),
            np.cos(d).astype(np.float32),
        ],
        axis=0,
    )


def extract_min_zmatrix_features(
    trajectory,
    atomgroup,
    every: int,
    chunk_size: int = 250,
    distance_backend: str = "serial",
    show_progress: bool = True,
) -> np.ndarray:
    _, _, calc_bonds, calc_angles, calc_dihedrals = require_mdanalysis()
    every = int(every)
    chunk_size = int(chunk_size)
    if every < 1:
        raise ValueError("every must be >= 1.")
    if chunk_size < 1:
        raise ValueError("feature_chunk_size must be >= 1.")

    bonds, angles, dihedrals = build_min_zmatrix_indices(atomgroup.n_atoms)
    n_bonds = bonds.shape[0]
    n_angles = angles.shape[0]
    n_dihedrals = dihedrals.shape[0]
    feat_dim = n_bonds + 2 * n_angles + 2 * n_dihedrals
    n_total_frames = len(trajectory)
    n_frames = len(range(0, n_total_frames, every))
    if n_frames == 0:
        return np.zeros((0, feat_dim), dtype=np.float32)

    feats = np.empty((n_frames, feat_dim), dtype=np.float32)
    chunk_starts = range(0, n_total_frames, every * chunk_size)
    total_chunks = len(range(0, n_total_frames, every * chunk_size))
    iterator = tqdm(chunk_starts, total=total_chunks, desc="Extracting features") if show_progress else chunk_starts

    try:
        out_start = 0
        for raw_start in iterator:
            raw_stop = min(n_total_frames, raw_start + every * chunk_size)
            coords = trajectory.timeseries(asel=atomgroup, start=raw_start, stop=raw_stop, step=every, order="fac")
            n_chunk = coords.shape[0]

            bonds_chunk = calc_bonds(
                coords[:, bonds[:, 0], :].reshape(-1, 3),
                coords[:, bonds[:, 1], :].reshape(-1, 3),
                backend=distance_backend,
            ).reshape(n_chunk, n_bonds).astype(np.float32, copy=False)

            angles_chunk = calc_angles(
                coords[:, angles[:, 0], :].reshape(-1, 3),
                coords[:, angles[:, 1], :].reshape(-1, 3),
                coords[:, angles[:, 2], :].reshape(-1, 3),
                backend=distance_backend,
            ).reshape(n_chunk, n_angles).astype(np.float32, copy=False)

            dihedral_chunk = calc_dihedrals(
                coords[:, dihedrals[:, 0], :].reshape(-1, 3),
                coords[:, dihedrals[:, 1], :].reshape(-1, 3),
                coords[:, dihedrals[:, 2], :].reshape(-1, 3),
                coords[:, dihedrals[:, 3], :].reshape(-1, 3),
                backend=distance_backend,
            ).reshape(n_chunk, n_dihedrals).astype(np.float32, copy=False)

            out_stop = out_start + n_chunk
            feats[out_start:out_stop, :n_bonds] = bonds_chunk
            feats[out_start:out_stop, n_bonds:n_bonds + n_angles] = np.sin(angles_chunk)
            feats[out_start:out_stop, n_bonds + n_angles:n_bonds + 2 * n_angles] = np.cos(angles_chunk)
            offset = n_bonds + 2 * n_angles
            feats[out_start:out_stop, offset:offset + n_dihedrals] = np.sin(dihedral_chunk)
            feats[out_start:out_stop, offset + n_dihedrals:] = np.cos(dihedral_chunk)
            out_start = out_stop
        return feats
    except (AttributeError, NotImplementedError, TypeError):
        pass

    iterator = tqdm(trajectory[::every], desc="Extracting features") if show_progress else trajectory[::every]
    for frame_idx, _ in enumerate(iterator):
        feats[frame_idx] = internal_coords_min_zmatrix(atomgroup.positions, bonds, angles, dihedrals)
    return feats


def read_colvars(colvars_path: str, index_mismatch: bool = True, skip_rows: int = 1) -> tuple[list[str], np.ndarray]:
    with open(colvars_path, "r", encoding="utf-8") as f:
        headers = None
        for line in f:
            if line.startswith("#"):
                headers = line[1:].strip().split()
                break
    if headers is None:
        raise ValueError(f"Cannot find a '#'-prefixed header in {colvars_path}")

    data = np.loadtxt(colvars_path, comments=["#", "@"], skiprows=skip_rows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if index_mismatch:
        data = data[1:]

    seen: set[str] = set()
    keep: list[int] = []
    clean_headers: list[str] = []
    for idx, name in enumerate(headers):
        if name not in seen:
            seen.add(name)
            keep.append(idx)
            clean_headers.append(name)
    return clean_headers, data[:, keep].astype(np.float32)


def prepare_cv_matrix(
    colvars_all: np.ndarray,
    headers: list[str],
    cvs_to_use: list[str] | None = None,
    periodic: bool = False,
) -> tuple[np.ndarray, list[str]]:
    if cvs_to_use:
        missing = [name for name in cvs_to_use if name not in headers]
        if missing:
            raise ValueError(f"Requested CV columns are missing: {missing}")
        indices = [headers.index(name) for name in cvs_to_use]
        cv_data = colvars_all[:, indices].astype(np.float32)
        cv_headers = list(cvs_to_use)
    else:
        cv_data = colvars_all.astype(np.float32)
        cv_headers = list(headers)

    if periodic and cv_headers:
        sincos = []
        sincos_headers = []
        for idx, name in enumerate(cv_headers):
            angle = cv_data[:, idx] * np.pi / 180.0
            sincos.append(np.sin(angle).astype(np.float32))
            sincos.append(np.cos(angle).astype(np.float32))
            sincos_headers.extend([f"s{name}", f"c{name}"])
        cv_data = np.concatenate([cv_data, np.stack(sincos, axis=1)], axis=1).astype(np.float32)
        cv_headers = cv_headers + sincos_headers
    return cv_data, cv_headers


@dataclass
class TrajectoryBlock:
    traj_idx: int
    dcd_path: str
    features: np.ndarray | None
    colvars: np.ndarray
    headers: list[str]
    n_frames: int


def load_trajectory_block(
    traj_idx: int,
    dcd_path: str,
    topology_file: str,
    selection: str,
    every: int,
    index_mismatch: bool,
    use_internal_features: bool,
    feature_chunk_size: int,
    distance_backend: str,
) -> TrajectoryBlock | None:
    Universe, _, _, _, _ = require_mdanalysis()
    colvars_path = os.path.splitext(dcd_path)[0] + ".colvars.traj"
    if not os.path.exists(colvars_path):
        print(f"[WARN] Missing colvars for {dcd_path}; skipping.")
        return None

    u = None
    try:
        u = Universe(topology_file, dcd_path)
        if use_internal_features:
            atomgroup = u.select_atoms(selection)
            features = extract_min_zmatrix_features(
                u.trajectory,
                atomgroup,
                every=every,
                chunk_size=feature_chunk_size,
                distance_backend=distance_backend,
                show_progress=False,
            )
            n_traj_frames = int(features.shape[0])
        else:
            features = None
            n_traj_frames = len(u.trajectory[::every])

        headers, colvars = read_colvars(colvars_path, index_mismatch=index_mismatch)
        colvars = colvars[::every].astype(np.float32)
        if colvars.shape[0] != n_traj_frames:
            raise ValueError(
                f"Frame mismatch for {dcd_path}: colvars has {colvars.shape[0]} frames, "
                f"DCD path has {n_traj_frames} frames."
            )
        return TrajectoryBlock(traj_idx, dcd_path, features, colvars, headers, n_traj_frames)
    finally:
        if u is not None and hasattr(u.trajectory, "close"):
            u.trajectory.close()


def find_dcd_files(folder: str, match_prefix: str) -> list[str]:
    files = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(folder)
        for name in names
        if name.startswith(match_prefix) and name.endswith(".dcd")
    )
    if not files:
        raise RuntimeError(f"No DCD files starting with {match_prefix!r} found under {folder}")
    return files


def load_trajectory_blocks(config: dict[str, Any], use_internal_features: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int], list[str]]:
    topology_file = config["topology_file"]
    dcd_folder = config["dcd_folder"]
    match_prefix = config["match"]
    selection = config.get("sel_weights", "protein and not name H*")
    every = int(config.get("every", 1))
    index_mismatch = bool(config.get("colvars_mismatch", True))
    feature_chunk_size = int(config.get("feature_chunk_size", 250))
    distance_backend = str(config.get("distance_backend", "serial"))
    workers = int(config.get("trajectory_workers", config.get("num_workers", 1)))
    if workers < 0:
        raise ValueError("trajectory_workers must be >= 0.")

    dcd_files = find_dcd_files(dcd_folder, match_prefix)
    jobs = [
        (
            traj_idx,
            dcd_path,
            topology_file,
            selection,
            every,
            index_mismatch,
            use_internal_features,
            feature_chunk_size,
            distance_backend,
        )
        for traj_idx, dcd_path in enumerate(dcd_files)
    ]

    blocks: list[TrajectoryBlock] = []
    actual_workers = min(max(1, workers), len(jobs))
    if actual_workers > 1:
        print(f"[IO] Loading trajectories with {actual_workers} worker threads.")
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_path = {executor.submit(load_trajectory_block, *job): job[1] for job in jobs}
            for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Processing trajectories"):
                try:
                    block = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed while processing {future_to_path[future]}") from exc
                if block is not None:
                    blocks.append(block)
        blocks.sort(key=lambda block: block.traj_idx)
    else:
        for job in tqdm(jobs, desc="Processing trajectories"):
            block = load_trajectory_block(*job)
            if block is not None:
                blocks.append(block)

    if not blocks:
        raise RuntimeError("No usable DCD/.colvars.traj pairs were loaded.")

    headers = blocks[0].headers
    for block in blocks[1:]:
        if block.headers != headers:
            raise ValueError("Colvars headers differ across trajectories.")

    colvars_all = np.vstack([block.colvars for block in blocks]).astype(np.float32)
    features_all = (
        np.vstack([block.features for block in blocks if block.features is not None]).astype(np.float32)
        if use_internal_features
        else np.zeros((colvars_all.shape[0], 0), dtype=np.float32)
    )
    traj_id = np.concatenate(
        [np.full(block.n_frames, block.traj_idx, dtype=np.int64) for block in blocks]
    )
    frame_counts = [block.n_frames for block in blocks]
    sources = [os.path.abspath(block.dcd_path) for block in blocks]
    return features_all, colvars_all, traj_id, headers, frame_counts, sources


def build_clustering_matrix(
    features: np.ndarray,
    cv: np.ndarray | None,
    cluster_space: str,
    standardize_features: bool,
    pca_cluster_dim: int,
) -> np.ndarray:
    _, PCA, StandardScaler = require_sklearn()
    cluster_space = str(cluster_space).lower()
    if cluster_space == "features":
        X = features
        use_pca = False
    elif cluster_space == "pca_highdim":
        X = features
        use_pca = True
    elif cluster_space == "cv":
        if cv is None:
            raise ValueError("cluster_space='cv' requires CV data.")
        X = cv
        use_pca = False
    elif cluster_space == "pca_cv":
        if cv is None:
            raise ValueError("cluster_space='pca_cv' requires CV data.")
        X = cv
        use_pca = True
    else:
        raise ValueError("cluster_space must be one of: features, pca_highdim, cv, pca_cv.")

    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError(f"Cannot cluster an empty feature matrix with shape {X.shape}.")

    X_work = X.astype(np.float64)
    if standardize_features:
        X_work = StandardScaler().fit_transform(X_work)
    if use_pca:
        ncomp = min(int(pca_cluster_dim), X_work.shape[1])
        X_work = PCA(n_components=ncomp).fit_transform(X_work)
    return X_work.astype(np.float32)


def assign_clusters_random_centers(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    if n_clusters >= X.shape[0]:
        raise ValueError("riteweight.n_clusters must be smaller than the number of frames.")
    centers = X[rng.choice(X.shape[0], size=n_clusters, replace=False)]
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X @ centers.T)
    return np.argmin(d2, axis=1).astype(np.int32)


def build_transition_matrix(start_labels, end_labels, weights, n_clusters, eps=1e-15):
    trans_num = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    cluster_weight = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(cluster_weight, start_labels, weights)
    np.add.at(trans_num, (start_labels, end_labels), weights)
    trans = trans_num / (cluster_weight[:, None] + eps)
    zero_rows = cluster_weight <= eps
    if np.any(zero_rows):
        trans[zero_rows, :] = 0.0
        trans[zero_rows, np.where(zero_rows)[0]] = 1.0
    row_sum = trans.sum(axis=1, keepdims=True)
    return trans / (row_sum + eps), cluster_weight


def stationary_distribution(trans: np.ndarray) -> np.ndarray:
    eig = require_scipy_eig()
    vals, vecs = eig(trans.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    vec = np.abs(np.real(vecs[:, idx]))
    norm = float(vec.sum())
    if norm <= 0:
        raise RuntimeError("Failed to compute a valid stationary distribution in RiteWeight.")
    return vec / norm


def riteweight(
    X: np.ndarray,
    seg_start_idx: np.ndarray,
    seg_end_idx: np.ndarray,
    n_clusters: int,
    n_iter: int,
    tol: float,
    tol_window: int,
    avg_last: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_segments = int(seg_start_idx.size)
    seg_weights = np.full(n_segments, 1.0 / max(1, n_segments), dtype=np.float64)
    prev_weights = seg_weights.copy()
    delta_history: list[float] = []
    stable_steps = 0
    trailing: list[np.ndarray] = []

    for step in tqdm(range(1, int(n_iter) + 1), desc="RiteWeight"):
        labels = assign_clusters_random_centers(X, int(n_clusters), rng)
        trans, cluster_weight = build_transition_matrix(
            labels[seg_start_idx],
            labels[seg_end_idx],
            seg_weights,
            int(n_clusters),
        )
        pi = stationary_distribution(trans)
        scale = pi / (cluster_weight + 1e-15)
        new_weights = seg_weights * scale[labels[seg_start_idx]]
        new_weights = np.clip(new_weights, 0.0, np.inf)
        new_weights /= np.sum(new_weights) + 1e-300

        delta = float(np.sum(np.abs(new_weights - prev_weights)))
        delta_history.append(delta)
        stable_steps = stable_steps + 1 if delta < float(tol) else 0
        if step > max(1, int(n_iter) - int(avg_last)):
            trailing.append(new_weights.copy())
        prev_weights = seg_weights
        seg_weights = new_weights
        if stable_steps >= int(tol_window):
            break

    if trailing:
        seg_weights = np.mean(np.stack(trailing, axis=0), axis=0)
        seg_weights = np.clip(seg_weights, 0.0, np.inf)
        seg_weights /= np.sum(seg_weights) + 1e-300

    weight_diff = np.zeros(X.shape[0] + 1, dtype=np.float64)
    count_diff = np.zeros(X.shape[0] + 1, dtype=np.float64)
    np.add.at(weight_diff, seg_start_idx, seg_weights)
    valid_end = seg_end_idx + 1
    inside = valid_end < weight_diff.shape[0]
    np.add.at(weight_diff, valid_end[inside], -seg_weights[inside])
    np.add.at(count_diff, seg_start_idx, 1.0)
    np.add.at(count_diff, valid_end[inside], -1.0)

    touch_weight = np.cumsum(weight_diff[:-1])
    touch_count = np.cumsum(count_diff[:-1])
    frame_weights = np.zeros(X.shape[0], dtype=np.float64)
    mask = touch_count > 0
    frame_weights[mask] = touch_weight[mask] / touch_count[mask]
    frame_weights = np.clip(frame_weights, 0.0, np.inf)
    frame_weights /= np.sum(frame_weights) + 1e-300
    return frame_weights.astype(np.float32), np.asarray(delta_history, dtype=np.float32)


def build_segment_indices(frame_counts: list[int], lag: int) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    offset = 0
    for n_frames in frame_counts:
        if n_frames > lag:
            s = np.arange(offset, offset + n_frames - lag, dtype=np.int64)
            starts.append(s)
            ends.append(s + lag)
        offset += n_frames
    if not starts:
        raise RuntimeError(f"No valid RiteWeight segments for lag={lag}.")
    return np.concatenate(starts), np.concatenate(ends)


def choose_k_elbow(
    X: np.ndarray,
    kmin: int,
    kmax: int,
    method: str,
    random_state: int,
    n_init: str | int,
    out_png: str | None = None,
) -> tuple[int, list[int], list[float]]:
    KMeans, _, _ = require_sklearn()
    ks = list(range(int(kmin), int(kmax) + 1))
    if not ks:
        raise ValueError("kmin/kmax produced no candidate k values.")
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        km.fit(X)
        inertias.append(float(km.inertia_))
    y = np.asarray(inertias, dtype=np.float64)

    if len(ks) == 1:
        best_k = ks[0]
    elif str(method).lower() == "second_derivative" and len(ks) >= 3:
        d2 = np.zeros_like(y)
        d2[1:-1] = y[:-2] - 2.0 * y[1:-1] + y[2:]
        best_k = ks[int(np.argmax(d2[1:-1]) + 1)]
    else:
        x = np.asarray(ks, dtype=np.float64)
        x1, y1 = x[0], y[0]
        x2, y2 = x[-1], y[-1]
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        dist = np.abs(a * x + b * y + c) / (np.sqrt(a * a + b * b) + 1e-12)
        best_k = ks[int(np.argmax(dist))]

    if out_png is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(ks, inertias, marker="o")
        plt.axvline(best_k, linestyle="--")
        plt.xlabel("k")
        plt.ylabel("KMeans inertia")
        plt.title(f"Elbow curve (best k = {best_k})")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
    return int(best_k), ks, inertias


def kmeans_metastable_labeling(
    X: np.ndarray,
    n_clusters: int,
    quantile: float,
    random_state: int,
    n_init: str | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    KMeans, _, _ = require_sklearn()
    km = KMeans(n_clusters=int(n_clusters), n_init=n_init, random_state=random_state)
    cluster_labels = km.fit_predict(X).astype(np.int64)
    centers = km.cluster_centers_
    dists = np.linalg.norm(X - centers[cluster_labels], axis=1).astype(np.float32)
    thresholds = np.zeros(int(n_clusters), dtype=np.float32)
    for state in range(int(n_clusters)):
        mask = cluster_labels == state
        thresholds[state] = np.quantile(dists[mask], float(quantile)) if np.any(mask) else np.inf
    meta_state = cluster_labels.copy()
    meta_state[dists > thresholds[cluster_labels]] = -1
    return meta_state.astype(np.int64), dists, thresholds, cluster_labels


def resolve_centroid_plot_indices(cv_headers: list[str], centroid_plot_cvs: list[str]) -> tuple[list[int], list[str]]:
    if not cv_headers:
        raise ValueError("CV headers are required for centroid plotting.")
    if not centroid_plot_cvs:
        raise ValueError("centroid_plot_cvs must list 2 or 3 CV column names.")
    if len(centroid_plot_cvs) not in {2, 3}:
        raise ValueError("centroid_plot_cvs must contain exactly 2 or 3 CV names.")
    missing = [name for name in centroid_plot_cvs if name not in cv_headers]
    if missing:
        raise ValueError(f"centroid_plot_cvs contains unknown CV columns: {missing}")
    return [cv_headers.index(name) for name in centroid_plot_cvs], list(centroid_plot_cvs)


def plot_kmeans_centroids_on_cvs(
    cv_data: np.ndarray,
    cv_headers: list[str],
    cluster_labels: np.ndarray,
    centroid_plot_cvs: list[str],
    out_prefix: str,
) -> dict[str, Any]:
    indices, names = resolve_centroid_plot_indices(cv_headers, centroid_plot_cvs)
    X_plot = cv_data[:, indices].astype(np.float32)
    n_clusters = int(np.max(cluster_labels)) + 1 if cluster_labels.size > 0 else 0
    if n_clusters <= 0:
        raise ValueError("No cluster labels available for centroid plotting.")

    centroids = np.zeros((n_clusters, len(indices)), dtype=np.float32)
    for state in range(n_clusters):
        mask = cluster_labels == state
        if np.any(mask):
            centroids[state] = X_plot[mask].mean(axis=0)

    pairs = [(0, 1)] if len(indices) == 2 else [(0, 1), (0, 2), (1, 2)]
    paths = []
    cmap = plt.get_cmap("tab10", max(n_clusters, 1))
    for i, j in pairs:
        plt.figure(figsize=(6.5, 5.5))
        plt.scatter(
            X_plot[:, i],
            X_plot[:, j],
            c=cluster_labels,
            cmap=cmap,
            s=6,
            alpha=0.28,
            linewidths=0,
            rasterized=True,
        )
        plt.scatter(
            centroids[:, i],
            centroids[:, j],
            c=np.arange(n_clusters),
            cmap=cmap,
            s=180,
            marker="X",
            edgecolors="black",
            linewidths=0.8,
        )
        for state in range(n_clusters):
            plt.annotate(
                str(state),
                (float(centroids[state, i]), float(centroids[state, j])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        plt.title(f"KMeans centroids on CVs: {names[i]} vs {names[j]}")
        plt.tight_layout()
        out_png = f"{out_prefix}_{names[i]}_vs_{names[j]}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        paths.append(out_png)

    return {"cvs": names, "centroids": centroids, "paths": paths}


def plot_state_centers_on_cvs(
    cv_data: np.ndarray,
    cv_headers: list[str],
    meta_state: np.ndarray,
    centers: np.ndarray,
    center_cv_names: list[str],
    centroid_plot_cvs: list[str],
    out_prefix: str,
    title_prefix: str,
) -> dict[str, Any]:
    indices, names = resolve_centroid_plot_indices(cv_headers, centroid_plot_cvs)
    center_indices, _ = resolve_centroid_plot_indices(center_cv_names, centroid_plot_cvs)
    X_plot = cv_data[:, indices].astype(np.float32)
    centers_plot = np.asarray(centers, dtype=np.float32)[:, center_indices]
    n_states = centers_plot.shape[0]

    pairs = [(0, 1)] if len(indices) == 2 else [(0, 1), (0, 2), (1, 2)]
    paths = []
    cmap = plt.get_cmap("tab10", max(n_states, 1))
    labels = np.asarray(meta_state, dtype=np.int64)
    intermediate = labels == -1
    labeled = labels >= 0

    for i, j in pairs:
        plt.figure(figsize=(6.5, 5.5))
        if np.any(intermediate):
            plt.scatter(
                X_plot[intermediate, i],
                X_plot[intermediate, j],
                c="lightgray",
                s=6,
                alpha=0.2,
                linewidths=0,
                rasterized=True,
                label="intermediate",
            )
        if np.any(labeled):
            plt.scatter(
                X_plot[labeled, i],
                X_plot[labeled, j],
                c=labels[labeled],
                cmap=cmap,
                s=8,
                alpha=0.45,
                linewidths=0,
                rasterized=True,
            )
        plt.scatter(
            centers_plot[:, i],
            centers_plot[:, j],
            c=np.arange(n_states),
            cmap=cmap,
            s=180,
            marker="X",
            edgecolors="black",
            linewidths=0.8,
        )
        for state in range(n_states):
            plt.annotate(
                str(state),
                (float(centers_plot[state, i]), float(centers_plot[state, j])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        plt.title(f"{title_prefix}: {names[i]} vs {names[j]}")
        plt.tight_layout()
        out_png = f"{out_prefix}_{names[i]}_vs_{names[j]}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        paths.append(out_png)

    return {"cvs": names, "centers": centers_plot, "paths": paths}


def coerce_basin_size(size, ndim: int) -> np.ndarray:
    if np.isscalar(size):
        out = np.full(ndim, float(size), dtype=np.float32)
    else:
        out = np.asarray(size, dtype=np.float32)
    if out.shape != (ndim,):
        raise ValueError(f"Basin size must be scalar or length {ndim}; got shape {out.shape}.")
    if np.any(out <= 0):
        raise ValueError("Basin sizes must be positive.")
    return out


def parse_user_defined_basins(config: dict[str, Any], cv_headers: list[str]) -> dict[str, Any]:
    basin_cfg = config.get("user_defined_basins", {})
    if not isinstance(basin_cfg, dict):
        raise ValueError("user_defined_basins must be a mapping.")
    cvs_to_label = basin_cfg.get("cvs_to_label", [])
    if not cvs_to_label:
        raise ValueError("user_defined_basins.cvs_to_label must list CV columns.")
    missing = [name for name in cvs_to_label if name not in cv_headers]
    if missing:
        raise ValueError(f"Unknown basin CV columns: {missing}")
    assignment_mode = str(basin_cfg.get("assignment_mode", "box")).lower()
    if assignment_mode not in {"box", "distance_cutoff"}:
        raise ValueError("assignment_mode must be 'box' or 'distance_cutoff'.")

    basins_raw = basin_cfg.get("basins")
    if basins_raw is None:
        basin_a = basin_cfg.get("basin_A")
        basin_b = basin_cfg.get("basin_B")
        basin_size = basin_cfg.get("basin_size")
        if basin_a is None or basin_b is None or basin_size is None:
            raise ValueError("Define user_defined_basins.basins or legacy basin_A/basin_B/basin_size.")
        basins_raw = [{"center": basin_a, "size": basin_size}, {"center": basin_b, "size": basin_size}]
    if not isinstance(basins_raw, list) or not basins_raw:
        raise ValueError("user_defined_basins.basins must be a non-empty list.")

    ndim = len(cvs_to_label)
    basins = []
    for idx, basin in enumerate(basins_raw):
        if not isinstance(basin, dict) or "center" not in basin:
            raise ValueError(f"Basin #{idx} must define a center.")
        center = np.asarray(basin["center"], dtype=np.float32)
        if center.shape != (ndim,):
            raise ValueError(f"Basin #{idx} center must have length {ndim}.")
        item: dict[str, Any] = {"label": int(basin.get("label", idx)), "center": center}
        if assignment_mode == "box":
            item["size"] = coerce_basin_size(basin["size"], ndim)
        else:
            cutoff = float(basin.get("cutoff", basin.get("size", 0.0)))
            if cutoff <= 0:
                raise ValueError(f"Basin #{idx} cutoff must be positive.")
            item["cutoff"] = cutoff
        basins.append(item)

    labels = sorted(item["label"] for item in basins)
    expected = list(range(len(basins)))
    if labels != expected:
        raise ValueError(f"Basin labels must be contiguous 0..k-1; got {labels}, expected {expected}.")
    basins.sort(key=lambda item: item["label"])
    return {
        "cvs_to_label": list(cvs_to_label),
        "cv_indices": [cv_headers.index(name) for name in cvs_to_label],
        "assignment_mode": assignment_mode,
        "basins": basins,
    }


def user_defined_basin_labeling(cv_data: np.ndarray, basin_spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X = cv_data[:, basin_spec["cv_indices"]].astype(np.float32)
    basins = basin_spec["basins"]
    centers = np.stack([item["center"] for item in basins], axis=0)
    n_states = len(basins)
    meta_state = np.full(X.shape[0], -1, dtype=np.int64)
    dist_to_centroid = np.full(X.shape[0], np.inf, dtype=np.float32)
    thresholds = np.ones(n_states, dtype=np.float32)

    if basin_spec["assignment_mode"] == "box":
        sizes = np.stack([item["size"] for item in basins], axis=0)
        scaled = np.abs(X[:, None, :] - centers[None, :, :]) / sizes[None, :, :]
        norm_dist = np.max(scaled, axis=2).astype(np.float32)
        closest = np.argmin(norm_dist, axis=1)
        dist_to_centroid[:] = norm_dist[np.arange(X.shape[0]), closest]
        for basin in basins:
            label = basin["label"]
            inside = np.all(np.abs(X - basin["center"][None, :]) <= basin["size"][None, :], axis=1)
            meta_state[inside & (meta_state == -1)] = label
        return meta_state, dist_to_centroid, thresholds, {"centers": centers, "sizes": sizes}

    cutoffs = np.asarray([item["cutoff"] for item in basins], dtype=np.float32)
    euclid = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2).astype(np.float32)
    closest = np.argmin(euclid, axis=1)
    closest_dist = euclid[np.arange(X.shape[0]), closest]
    dist_to_centroid[:] = closest_dist
    thresholds[:] = cutoffs
    inside = closest_dist <= cutoffs[closest]
    meta_state[inside] = closest[inside].astype(np.int64)
    return meta_state, dist_to_centroid, thresholds, {"centers": centers, "cutoffs": cutoffs}


def report_metastate_sizes(meta_state: np.ndarray, n_states: int) -> None:
    total = int(meta_state.shape[0])
    for state in range(int(n_states)):
        count = int(np.sum(meta_state == state))
        print(f"[META] State {state}: {count}/{total} frames ({100.0 * count / max(1, total):.2f}%)")
    count = int(np.sum(meta_state == -1))
    print(f"[META] Intermediate (-1): {count}/{total} frames ({100.0 * count / max(1, total):.2f}%)")


def label_metastates(
    config: dict[str, Any],
    features: np.ndarray,
    cv_data: np.ndarray | None,
    cv_headers: list[str],
    elbow_png: str | None = None,
    centroid_plot_prefix: str | None = None,
) -> dict[str, Any]:
    method = str(config.get("labeling_method", "kmeans")).lower()
    if method == "user_defined_basins":
        if cv_data is None:
            raise ValueError("user_defined_basins requires CV data.")
        basin_spec = parse_user_defined_basins(config, cv_headers)
        meta_state, dist, thresholds, details = user_defined_basin_labeling(cv_data, basin_spec)
        n_states = len(basin_spec["basins"])
        centroid_plot_info = None
        if bool(config.get("plot_kmeans_centroids", False)):
            if centroid_plot_prefix is None:
                raise ValueError("centroid_plot_prefix is required when plot_kmeans_centroids=true.")
            centroid_plot_info = plot_state_centers_on_cvs(
                cv_data=cv_data,
                cv_headers=cv_headers,
                meta_state=meta_state,
                centers=details["centers"],
                center_cv_names=basin_spec["cvs_to_label"],
                centroid_plot_cvs=config.get("centroid_plot_cvs", basin_spec["cvs_to_label"]),
                out_prefix=centroid_plot_prefix,
                title_prefix="User-defined metastates on CVs",
            )
            for path in centroid_plot_info["paths"]:
                print(f"[PLOT] Saved centroid plot: {path}")
        report_metastate_sizes(meta_state, n_states)
        return {
            "labeling_method": method,
            "meta_state": meta_state,
            "dist_to_centroid": dist,
            "thresholds": thresholds,
            "n_states": n_states,
            "details": {
                "cvs_to_label": basin_spec["cvs_to_label"],
                "assignment_mode": basin_spec["assignment_mode"],
                "basins": [
                    {
                        "label": int(item["label"]),
                        "center": item["center"].tolist(),
                        **({"size": item["size"].tolist()} if "size" in item else {}),
                        **({"cutoff": float(item["cutoff"])} if "cutoff" in item else {}),
                    }
                    for item in basin_spec["basins"]
                ],
                "fit": {key: np.asarray(value).tolist() for key, value in details.items()},
                "centroid_plot": (
                    {
                        "cvs": centroid_plot_info["cvs"],
                        "paths": centroid_plot_info["paths"],
                        "centers": centroid_plot_info["centers"].tolist(),
                    }
                    if centroid_plot_info is not None else None
                ),
            },
            "cluster_space": None,
            "elbow": None,
            "centroid_plot": (
                {
                    "cvs": centroid_plot_info["cvs"],
                    "paths": centroid_plot_info["paths"],
                    "centers": centroid_plot_info["centers"].tolist(),
                }
                if centroid_plot_info is not None else None
            ),
        }
    if method != "kmeans":
        raise ValueError("labeling_method must be 'kmeans' or 'user_defined_basins'.")

    cluster_space = str(config.get("cluster_space", "features")).lower()
    X_cluster = build_clustering_matrix(
        features=features,
        cv=cv_data,
        cluster_space=cluster_space,
        standardize_features=bool(config.get("standardize_features", True)),
        pca_cluster_dim=int(config.get("pca_cluster_dim", 20)),
    )
    manual_k = config.get("n_clusters", None)
    if manual_k is None:
        best_k, ks, inertias = choose_k_elbow(
            X_cluster,
            kmin=int(config.get("kmin", 2)),
            kmax=int(config.get("kmax", 12)),
            method=str(config.get("elbow_method", "knee")),
            random_state=int(config.get("kmeans_random_state", 0)),
            n_init=config.get("kmeans_n_init", "auto"),
            out_png=elbow_png,
        )
    else:
        best_k = int(manual_k)
        ks = [best_k]
        inertias = []
    meta_state, dist, thresholds, cluster_labels = kmeans_metastable_labeling(
        X_cluster,
        n_clusters=best_k,
        quantile=float(config.get("intermediate_quantile", 0.9)),
        random_state=int(config.get("kmeans_random_state", 0)),
        n_init=config.get("kmeans_n_init", "auto"),
    )
    centroid_plot_info = None
    if bool(config.get("plot_kmeans_centroids", False)):
        if cv_data is None:
            raise ValueError("plot_kmeans_centroids requires CV data to be available.")
        if centroid_plot_prefix is None:
            raise ValueError("centroid_plot_prefix is required when plot_kmeans_centroids=true.")
        centroid_plot_info = plot_kmeans_centroids_on_cvs(
            cv_data=cv_data,
            cv_headers=cv_headers,
            cluster_labels=cluster_labels,
            centroid_plot_cvs=config.get("centroid_plot_cvs", []),
            out_prefix=centroid_plot_prefix,
        )
        for path in centroid_plot_info["paths"]:
            print(f"[PLOT] Saved centroid plot: {path}")
    report_metastate_sizes(meta_state, best_k)
    return {
        "labeling_method": method,
        "meta_state": meta_state,
        "dist_to_centroid": dist,
        "thresholds": thresholds,
        "n_states": int(best_k),
        "details": {
            "cluster_labels": cluster_labels.tolist(),
            "centroid_plot": (
                {
                    "cvs": centroid_plot_info["cvs"],
                    "paths": centroid_plot_info["paths"],
                    "centroids": centroid_plot_info["centroids"].tolist(),
                }
                if centroid_plot_info is not None else None
            ),
        },
        "cluster_space": cluster_space,
        "elbow": {
            "ks": ks,
            "inertias": [float(x) for x in inertias],
            "method": str(config.get("elbow_method", "knee")),
        },
        "centroid_plot": (
            {
                "cvs": centroid_plot_info["cvs"],
                "paths": centroid_plot_info["paths"],
                "centroids": centroid_plot_info["centroids"].tolist(),
            }
            if centroid_plot_info is not None else None
        ),
    }


def save_dataset(
    dataset_path: str,
    save_format: str,
    features: np.ndarray,
    weights: np.ndarray,
    meta_state: np.ndarray,
    dist_to_centroid: np.ndarray,
    thresholds: np.ndarray,
    meta: dict[str, Any],
    cv_data: np.ndarray | None,
    traj_id: np.ndarray | None,
) -> None:
    ensure_dir(os.path.dirname(dataset_path) or ".")
    if str(save_format).lower() == "pt":
        pack: dict[str, Any] = {
            "features": torch.from_numpy(features.astype(np.float32)),
            "weights": torch.from_numpy(weights.astype(np.float32)),
            "meta_state": torch.from_numpy(meta_state.astype(np.int64)),
            "dist_to_centroid": torch.from_numpy(dist_to_centroid.astype(np.float32)),
            "thresholds": torch.from_numpy(thresholds.astype(np.float32)),
            "meta": meta,
        }
        if cv_data is not None:
            pack["cv"] = torch.from_numpy(cv_data.astype(np.float32))
        if traj_id is not None:
            pack["traj_id"] = torch.from_numpy(traj_id.astype(np.int64))
        torch.save(pack, dataset_path)
        return

    meta_yaml = np.array([__import__("yaml").safe_dump(meta, sort_keys=False)], dtype=object)
    out = {
        "features": features.astype(np.float32),
        "weights": weights.astype(np.float32),
        "meta_state": meta_state.astype(np.int64),
        "dist_to_centroid": dist_to_centroid.astype(np.float32),
        "thresholds": thresholds.astype(np.float32),
        "meta_yaml": meta_yaml,
    }
    if cv_data is not None:
        out["cv"] = cv_data.astype(np.float32)
    if traj_id is not None:
        out["traj_id"] = traj_id.astype(np.int64)
    np.savez_compressed(dataset_path, **out)


def save_weights_csv(
    path: str,
    cv_data: np.ndarray,
    cv_headers: list[str],
    weights: np.ndarray,
    meta_state: np.ndarray,
    dist_to_centroid: np.ndarray,
    traj_id: np.ndarray | None,
) -> None:
    df = pd.DataFrame(cv_data, columns=cv_headers)
    df.insert(0, "frame", np.arange(cv_data.shape[0], dtype=np.int64))
    if traj_id is not None:
        df["traj_id"] = traj_id
    df["weight"] = weights
    df["meta_state"] = meta_state
    df["is_intermediate"] = (meta_state == -1).astype(np.int8)
    df["dist_to_centroid"] = dist_to_centroid
    df.to_csv(path, index=False)


def write_concat_dcd(config: dict[str, Any], sources: list[str], every: int, out_dir: str) -> None:
    if not bool(config.get("write_concat_dcd", False)):
        return
    Universe, DCDWriter, _, _, _ = require_mdanalysis()
    topology_file = config["topology_file"]
    selection = config.get("sel_output", config.get("sel_weights", "protein and not name H*"))
    out_dcd = os.path.join(out_dir, "concat.dcd")
    u0 = Universe(topology_file, sources[0])
    sel0 = u0.select_atoms(selection)
    with DCDWriter(out_dcd, sel0.n_atoms) as writer:
        for dcd_path in tqdm(sources, desc="Writing concat DCD"):
            u = Universe(topology_file, dcd_path)
            sel = u.select_atoms(selection)
            for _ in u.trajectory[::every]:
                writer.write(sel)
            if hasattr(u.trajectory, "close"):
                u.trajectory.close()
    if hasattr(u0.trajectory, "close"):
        u0.trajectory.close()
    print(f"[DCD] Saved concatenated DCD: {out_dcd}")


def build_from_trajectories(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(config["output_dir"])
    save_format = str(config.get("save_format", "pt")).lower()
    dataset_path = config.get("dataset_path", os.path.join(out_dir, "dataset.pt" if save_format == "pt" else "dataset.npz"))

    riteweight_space = str(config.get("riteweight_space", "features")).lower()
    cluster_space = str(config.get("cluster_space", "features")).lower()
    use_internal = riteweight_space == "features" or cluster_space in {"features", "pca_highdim"}

    features_raw, colvars_all, traj_id, headers, frame_counts, sources = load_trajectory_blocks(config, use_internal)
    cv_data, cv_headers = prepare_cv_matrix(
        colvars_all,
        headers,
        cvs_to_use=config.get("cvs_to_save", config.get("cvs_to_use", [])),
        periodic=bool(config.get("periodic", False)),
    )
    if not use_internal:
        features = cv_data.copy()
        stored_feature_space = "cv"
    else:
        features = features_raw
        stored_feature_space = "features"

    n_frames, feature_dim = features.shape
    print(f"[INFO] frames={n_frames}, feature_dim={feature_dim}, stored_feature_space={stored_feature_space}")

    do_reweight = bool(config.get("reweight", True))
    rw_cfg = config.get("riteweight", {}) or {}
    if do_reweight:
        rw_lag = int(rw_cfg.get("lag", config.get("lag", 1)))
        seg_start, seg_end = build_segment_indices(frame_counts, rw_lag)
        X_rw = features if riteweight_space == "features" else cv_data
        max_clusters = max(1, min(seg_start.size - 1, n_frames - 1))
        n_clusters = rw_cfg.get("n_clusters", config.get("rw_n_clusters", min(100, max_clusters)))
        n_clusters = min(int(n_clusters), max_clusters)
        weights, delta_history = riteweight(
            X_rw,
            seg_start,
            seg_end,
            n_clusters=n_clusters,
            n_iter=int(rw_cfg.get("n_iter", config.get("rw_n_iter", 200))),
            tol=float(rw_cfg.get("tol", config.get("rw_tol", 1e-6))),
            tol_window=int(rw_cfg.get("tol_window", config.get("rw_tol_window", 5))),
            avg_last=int(rw_cfg.get("avg_last", config.get("rw_avg_last", 20))),
            seed=int(rw_cfg.get("seed", config.get("rw_seed", 2026))),
        )
    else:
        rw_lag = None
        n_clusters = 0
        seg_start = np.array([], dtype=np.int64)
        weights = np.ones(n_frames, dtype=np.float32) / max(1, n_frames)
        delta_history = np.asarray([], dtype=np.float32)
        print("[INFO] Reweighting disabled; using uniform normalized weights.")

    elbow_png = os.path.join(out_dir, "elbow_kmeans.png") if bool(config.get("write_diag_plots", True)) else None
    centroid_plot_prefix = os.path.join(out_dir, "kmeans_centroids")
    label_result = label_metastates(
        config,
        features,
        cv_data,
        cv_headers,
        elbow_png=elbow_png,
        centroid_plot_prefix=centroid_plot_prefix,
    )
    meta_state = label_result["meta_state"]
    dist_to_centroid = label_result["dist_to_centroid"]
    thresholds = label_result["thresholds"]

    if bool(config.get("save_cv", True)):
        save_weights_csv(
            os.path.join(out_dir, "weights_and_labels.csv"),
            cv_data,
            cv_headers,
            weights,
            meta_state,
            dist_to_centroid,
            traj_id,
        )

    meta = {
        "n_frames": int(n_frames),
        "feature_dim": int(feature_dim),
        "k_selected": int(label_result["n_states"]),
        "cv_headers": cv_headers,
        "traj_sources": sources,
        "config": config,
        "weighting_method": "riteweight" if do_reweight else "uniform",
        "reweight": bool(do_reweight),
        "riteweight": (
            {
                "n_clusters": int(n_clusters),
                "lag": int(rw_lag),
                "n_segments": int(seg_start.size),
                "delta_history": delta_history.tolist(),
            }
            if do_reweight else None
        ),
        "riteweight_space": riteweight_space,
        "stored_feature_space": stored_feature_space,
        "labeling_method": label_result["labeling_method"],
        "cluster_space": label_result["cluster_space"],
        "elbow": label_result["elbow"],
        "centroid_plot": label_result.get("centroid_plot"),
        "label_details": label_result["details"],
        "notes": "TensorQ dataset: labels support both next-hit and pair-wise committors.",
    }
    save_dataset(dataset_path, save_format, features, weights, meta_state, dist_to_centroid, thresholds, meta, cv_data, traj_id)
    np.save(os.path.join(out_dir, "label_thresholds.npy"), thresholds.astype(np.float32))
    write_concat_dcd(config, sources, int(config.get("every", 1)), out_dir)
    print(f"[DATASET] Saved dataset: {dataset_path}")
    return {"dataset_path": os.path.abspath(dataset_path), "n_states": int(label_result["n_states"])}


def relabel_existing_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = config.get("dataset_path")
    if dataset_path is None:
        raise KeyError("--relabel-only requires LABEL.dataset_path.")
    pack = load_dataset(dataset_path)
    features = pack.features.numpy().astype(np.float32)
    cv_data = pack.cv.numpy().astype(np.float32) if pack.cv is not None else None
    cv_headers = list(pack.meta.get("cv_headers", []))
    if cv_data is None and pack.meta.get("stored_feature_space") == "cv":
        cv_data = features
        if not cv_headers:
            cv_headers = [f"cv_{idx}" for idx in range(cv_data.shape[1])]
    if cv_data is None:
        raise RuntimeError("Relabeling requires saved CV data or stored_feature_space='cv'.")

    out_dir = ensure_dir(config.get("output_dir", os.path.dirname(dataset_path) or "."))
    elbow_png = os.path.join(out_dir, "elbow_kmeans.png") if bool(config.get("write_diag_plots", True)) else None
    centroid_plot_prefix = os.path.join(out_dir, "kmeans_centroids")
    result = label_metastates(
        config,
        features,
        cv_data,
        cv_headers,
        elbow_png=elbow_png,
        centroid_plot_prefix=centroid_plot_prefix,
    )
    meta = dict(pack.meta)
    meta.update(
        {
            "k_selected": int(result["n_states"]),
            "config": config,
            "labeling_method": result["labeling_method"],
            "cluster_space": result["cluster_space"],
            "elbow": result["elbow"],
            "centroid_plot": result.get("centroid_plot"),
            "label_details": result["details"],
            "notes": "TensorQ dataset: labels support both next-hit and pair-wise committors.",
        }
    )
    save_format = "pt" if os.path.splitext(dataset_path)[1].lower() in {".pt", ".pth"} else "npz"
    save_dataset(
        dataset_path,
        save_format,
        features,
        pack.weights.numpy().astype(np.float32),
        result["meta_state"],
        result["dist_to_centroid"],
        result["thresholds"],
        meta,
        cv_data,
        pack.traj_id.numpy().astype(np.int64) if pack.traj_id is not None else None,
    )
    if bool(config.get("save_cv", True)):
        save_weights_csv(
            os.path.join(out_dir, "weights_and_labels.csv"),
            cv_data,
            cv_headers,
            pack.weights.numpy().astype(np.float32),
            result["meta_state"],
            result["dist_to_centroid"],
            pack.traj_id.numpy().astype(np.int64) if pack.traj_id is not None else None,
        )
    print(f"[RELABEL] Updated dataset labels in: {dataset_path}")
    return {"dataset_path": os.path.abspath(dataset_path), "n_states": int(result["n_states"])}


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def run(config: dict[str, Any], relabel_only: bool = False) -> dict[str, Any]:
    out_dir = ensure_dir(config.get("output_dir", config.get("out_dir", "./label_out")))
    config["output_dir"] = out_dir
    log_path = os.path.join(out_dir, "label.log")
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        tee_out = TeeStream(sys.stdout, log_file)
        tee_err = TeeStream(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[LOG] Writing label log to: {log_path}")
            summary = relabel_existing_dataset(config) if relabel_only else build_from_trajectories(config)
            summary["log"] = os.path.abspath(log_path)
            write_yaml(summary, os.path.join(out_dir, "summary.yaml"))
            return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or relabel a TensorQ dataset shared by next-hit and pair-wise committors.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--relabel-only", action="store_true", help="Update labels in an existing dataset.")
    parser.add_argument("--trajectory-workers", type=int, default=None, help="Override LABEL.trajectory_workers.")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = select_section(raw, "TENSORQ_LABEL", "NEXT_HIT_LABEL", "LABEL", "MultiState")
    if args.trajectory_workers is not None:
        cfg["trajectory_workers"] = int(args.trajectory_workers)
    run(cfg, relabel_only=args.relabel_only)


if __name__ == "__main__":
    main()

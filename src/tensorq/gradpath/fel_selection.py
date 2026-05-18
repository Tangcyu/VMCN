from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import warnings
from typing import Any, Sequence

import numpy as np

try:
    from .selection import ChannelSelection
except ImportError:  # pragma: no cover - supports direct file execution in tests/examples.
    from selection import ChannelSelection


@dataclass(frozen=True)
class FelKdeSelectionResult:
    """Grid/KDE channel selection result in CV coordinates."""

    selected_cv_points: np.ndarray
    selected_q: np.ndarray
    selected_weights: np.ndarray
    selected_cluster_labels: np.ndarray
    selected_grid_indices: np.ndarray
    grid_centers: np.ndarray
    grid_density: np.ndarray
    grid_free_energy: np.ndarray
    grid_q: np.ndarray
    grid_q_product: np.ndarray
    active_grid_indices: np.ndarray
    active_cluster_labels: np.ndarray
    n_clusters: int
    k_values: np.ndarray
    inertias: np.ndarray


def _require_sklearn():
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FEL KDE selection needs scikit-learn for KMeans clustering.") from exc
    return KMeans


def _gaussian_filter(values: np.ndarray, sigma: Sequence[float], periodic: Sequence[bool]) -> np.ndarray:
    if all(float(item) <= 0.0 for item in sigma):
        return values.astype(np.float64, copy=True)
    try:
        from scipy.ndimage import gaussian_filter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FEL KDE selection needs scipy.ndimage.gaussian_filter for KDE smoothing.") from exc
    modes = ["wrap" if bool(item) else "nearest" for item in periodic]
    return gaussian_filter(values.astype(np.float64, copy=False), sigma=list(sigma), mode=modes)


def _as_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_dim).")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_1d_weights(weights: np.ndarray | None, n_samples: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_samples, dtype=np.float64)
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != n_samples:
        raise ValueError(f"weights must have shape ({n_samples},).")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("weights must be finite and nonnegative.")
    return array


def _validate_pair(pair: tuple[int, int], n_states: int) -> tuple[int, int]:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("pair must be a two-item tuple/list: (state_i, state_j).")
    i, j = int(pair[0]), int(pair[1])
    if i == j:
        raise ValueError("pair states must be distinct.")
    if not (0 <= i < n_states and 0 <= j < n_states):
        raise ValueError("pair contains a state index outside the q-vector dimension.")
    return i, j


def _normalize_bins(bins: int | Sequence[int], dim: int) -> list[int]:
    if isinstance(bins, (int, np.integer)):
        out = [int(bins)] * dim
    else:
        out = [int(item) for item in bins]
    if len(out) != dim or any(item < 2 for item in out):
        raise ValueError("bins must be an integer >=2 or one integer >=2 per CV dimension.")
    return out


def _normalize_bandwidth(bandwidth_bins: float | Sequence[float], dim: int) -> list[float]:
    if isinstance(bandwidth_bins, (int, float, np.integer, np.floating)):
        out = [float(bandwidth_bins)] * dim
    else:
        out = [float(item) for item in bandwidth_bins]
    if len(out) != dim or any(item < 0.0 for item in out):
        raise ValueError("bandwidth_bins must be a nonnegative scalar or one value per CV dimension.")
    return out


def _normalize_periodic(periodic: bool | Sequence[bool] | None, dim: int) -> list[bool]:
    if periodic is None:
        return [False] * dim
    if isinstance(periodic, bool):
        return [bool(periodic)] * dim
    out = [bool(item) for item in periodic]
    if len(out) != dim:
        raise ValueError("periodic must be a bool or one bool per CV dimension.")
    return out


def _default_ranges(cv_points: np.ndarray, periodic: Sequence[bool], periodic_units: str) -> list[tuple[float, float]]:
    units = str(periodic_units).lower()
    periodic_range = (-np.pi, np.pi) if units in {"radians", "radian", "rad"} else (-180.0, 180.0)
    ranges = []
    for dim, is_periodic in enumerate(periodic):
        if is_periodic:
            ranges.append(periodic_range)
        else:
            lo = float(np.nanmin(cv_points[:, dim]))
            hi = float(np.nanmax(cv_points[:, dim]))
            pad = max((hi - lo) * 0.02, 1.0e-12)
            ranges.append((lo - pad, hi + pad))
    return ranges


def _normalize_ranges(
    cv_points: np.ndarray,
    ranges: Sequence[Sequence[float]] | None,
    periodic: Sequence[bool],
    periodic_units: str,
) -> list[tuple[float, float]]:
    if ranges is None:
        return _default_ranges(cv_points, periodic, periodic_units)
    out = [(float(item[0]), float(item[1])) for item in ranges]
    if len(out) != cv_points.shape[1] or any(hi <= lo for lo, hi in out):
        raise ValueError("ranges must contain [min, max] for each CV dimension.")
    return out


def _grid_centers(edges: list[np.ndarray]) -> np.ndarray:
    centers_1d = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
    mesh = np.meshgrid(*centers_1d, indexing="ij")
    return np.stack([item.ravel() for item in mesh], axis=1)


def _cluster_coordinates(points: np.ndarray, periodic: Sequence[bool], periodic_units: str) -> np.ndarray:
    units = str(periodic_units).lower()
    scale = 1.0 if units in {"radians", "radian", "rad"} else np.pi / 180.0
    columns = []
    for dim, is_periodic in enumerate(periodic):
        values = points[:, dim]
        if is_periodic:
            angle = values * scale
            columns.extend([np.sin(angle), np.cos(angle)])
        else:
            span = max(float(np.nanmax(values) - np.nanmin(values)), 1.0e-12)
            columns.append((values - float(np.nanmean(values))) / span)
    return np.stack(columns, axis=1)


def _evaluate_grid_q(
    centers: np.ndarray,
    q_arr: np.ndarray,
    q_evaluator: Callable[[np.ndarray], np.ndarray] | None,
    *,
    n_states: int,
    edges: list[np.ndarray],
    sigma: Sequence[float],
    periodic: Sequence[bool],
    weights: np.ndarray,
    cv: np.ndarray,
    eps: float,
) -> np.ndarray:
    if q_evaluator is not None:
        q_grid = _as_2d_float("q_evaluator(centers)", q_evaluator(centers))
        if q_grid.shape != (centers.shape[0], n_states):
            raise ValueError(
                "q_evaluator must return an array with shape "
                f"({centers.shape[0]}, {n_states}), got {q_grid.shape}."
            )
        if np.min(q_grid) < -1.0e-6:
            raise ValueError("q_evaluator returned negative values beyond numerical tolerance.")
        return np.clip(q_grid, 0.0, 1.0)

    warnings.warn(
        "select_fel_kde_centers was called without q_evaluator; falling back to "
        "the legacy KDE-smoothed conditional q estimate. The gradpath runner "
        "passes q_evaluator so q_i*q_j is evaluated directly at grid centers.",
        RuntimeWarning,
        stacklevel=2,
    )
    hist_shape = tuple(len(edge) - 1 for edge in edges)
    density_hist, _ = np.histogramdd(cv, bins=edges, weights=weights)
    density = _gaussian_filter(density_hist, sigma, periodic)
    q_grids = []
    for state in range(n_states):
        weighted_q_hist, _ = np.histogramdd(cv, bins=edges, weights=weights * q_arr[:, state])
        smooth_weighted_q = _gaussian_filter(weighted_q_hist, sigma, periodic)
        q_grids.append(smooth_weighted_q / np.maximum(density, float(eps)))
    q_grid = np.stack(q_grids, axis=-1).reshape((int(np.prod(hist_shape)), n_states))
    return np.clip(q_grid, 0.0, 1.0)


def _choose_k_elbow(
    points: np.ndarray,
    sample_weight: np.ndarray,
    *,
    kmin: int,
    kmax: int,
    random_state: int,
    n_init: str | int,
) -> tuple[int, np.ndarray, np.ndarray]:
    KMeans = _require_sklearn()
    n_points = int(points.shape[0])
    if n_points < 2:
        return 1, np.asarray([1], dtype=np.int64), np.asarray([0.0], dtype=np.float64)
    lo = min(max(1, int(kmin)), n_points)
    hi = min(max(lo, int(kmax)), n_points)
    ks = np.arange(lo, hi + 1, dtype=np.int64)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=int(k), n_init=n_init, random_state=int(random_state))
        km.fit(points, sample_weight=sample_weight)
        inertias.append(float(km.inertia_))
    inertias_arr = np.asarray(inertias, dtype=np.float64)
    if ks.size <= 2:
        return int(ks[np.argmin(inertias_arr)]), ks, inertias_arr

    x = (ks.astype(np.float64) - float(ks[0])) / max(float(ks[-1] - ks[0]), 1.0)
    y_span = max(float(np.nanmax(inertias_arr) - np.nanmin(inertias_arr)), 1.0e-12)
    y = (inertias_arr - float(inertias_arr[-1])) / y_span
    start = np.array([x[0], y[0]], dtype=np.float64)
    end = np.array([x[-1], y[-1]], dtype=np.float64)
    line = end - start
    denom = max(float(np.linalg.norm(line)), 1.0e-12)
    distances = np.abs(np.cross(line, np.stack([x, y], axis=1) - start)) / denom
    return int(ks[int(np.argmax(distances))]), ks, inertias_arr


def _kmeans_labels(
    points: np.ndarray,
    sample_weight: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    n_init: str | int,
) -> np.ndarray:
    if int(n_clusters) <= 1:
        return np.ones(points.shape[0], dtype=np.int64)
    KMeans = _require_sklearn()
    km = KMeans(n_clusters=int(n_clusters), n_init=n_init, random_state=int(random_state))
    labels = km.fit_predict(points, sample_weight=sample_weight)
    return labels.astype(np.int64) + 1


def select_fel_kde_centers(
    cv_points: np.ndarray,
    q: np.ndarray,
    pair: tuple[int, int],
    *,
    weights: np.ndarray | None = None,
    bins: int | Sequence[int] = 40,
    bandwidth_bins: float | Sequence[float] = 1.0,
    ranges: Sequence[Sequence[float]] | None = None,
    periodic: bool | Sequence[bool] | None = None,
    periodic_units: str = "degrees",
    threshold: float = 0.20,
    min_density: float = 0.0,
    kmin: int = 1,
    kmax: int = 12,
    n_clusters: int | None = None,
    points_per_cluster: int = 5,
    selection_power: float = 0.0,
    selection_method: str = "weighted",
    q_evaluator: Callable[[np.ndarray], np.ndarray] | None = None,
    random_state: int = 0,
    kmeans_n_init: str | int = "auto",
    eps: float = 1.0e-12,
) -> FelKdeSelectionResult:
    """
    Select synthetic CV-space channel centers from a binned KDE/FEL.

    The sampled CV distribution is smoothed with a Gaussian kernel on a regular
    grid to estimate density/FEL. q_i*q_j is evaluated at the same grid centers
    from q_evaluator when provided; the gradpath runner uses the committor model
    for this, so the reaction probability field is independent of KDE weights
    and bandwidth. Centers with q_i*q_j above threshold are clustered by KMeans,
    with k chosen by an elbow rule unless n_clusters is provided.
    Within each cluster, centers are sampled without replacement using KDE
    density weights, optionally multiplied by (q_i*q_j)**selection_power for
    the sampling probabilities. Set selection_method="top" to recover the old
    top-density behavior.
    """

    cv = _as_2d_float("cv_points", cv_points)
    q_arr = _as_2d_float("q", q)
    if q_arr.shape[0] != cv.shape[0]:
        raise ValueError("cv_points and q must have the same number of rows.")
    if np.min(q_arr) < -1.0e-6:
        raise ValueError("q contains negative values beyond numerical tolerance.")
    q_arr = np.clip(q_arr, 0.0, 1.0)
    n_frames, n_states = q_arr.shape
    state_i, state_j = _validate_pair(pair, n_states)
    w = _as_1d_weights(weights, n_frames)
    dim = cv.shape[1]
    bins_list = _normalize_bins(bins, dim)
    sigma = _normalize_bandwidth(bandwidth_bins, dim)
    periodic_flags = _normalize_periodic(periodic, dim)
    ranges_list = _normalize_ranges(cv, ranges, periodic_flags, periodic_units)
    edges = [np.linspace(lo, hi, n_bin + 1) for (lo, hi), n_bin in zip(ranges_list, bins_list)]

    density_hist, _ = np.histogramdd(cv, bins=edges, weights=w)
    density = _gaussian_filter(density_hist, sigma, periodic_flags)
    density_flat = density.ravel()
    density_norm = density_flat / max(float(np.nanmax(density_flat)), float(eps))
    free_energy = -np.log(np.maximum(density_norm, float(eps)))
    centers = _grid_centers(edges)
    q_flat = _evaluate_grid_q(
        centers,
        q_arr,
        q_evaluator,
        n_states=n_states,
        edges=edges,
        sigma=sigma,
        periodic=periodic_flags,
        weights=w,
        cv=cv,
        eps=float(eps),
    )
    q_product_flat = q_flat[:, state_i] * q_flat[:, state_j]

    active_mask = np.isfinite(q_product_flat) & (q_product_flat > float(threshold)) & (density_flat > float(min_density))
    active_ids = np.flatnonzero(active_mask)
    if active_ids.size == 0:
        raise RuntimeError(
            f"No KDE grid centers satisfy q_{state_i} * q_{state_j} > {float(threshold):g} "
            f"and density > {float(min_density):g}."
        )

    active_centers = centers[active_ids]
    active_weights = density_flat[active_ids]
    active_q_product = q_product_flat[active_ids]
    cluster_features = _cluster_coordinates(active_centers, periodic_flags, periodic_units)
    cluster_weight = active_weights * np.power(np.maximum(active_q_product, 0.0), float(selection_power))
    if np.sum(cluster_weight) <= 0.0:
        cluster_weight = np.ones(active_ids.size, dtype=np.float64)

    if n_clusters is None:
        best_k, ks, inertias = _choose_k_elbow(
            cluster_features,
            cluster_weight,
            kmin=int(kmin),
            kmax=int(kmax),
            random_state=int(random_state),
            n_init=kmeans_n_init,
        )
    else:
        best_k = max(1, min(int(n_clusters), active_ids.size))
        ks = np.asarray([best_k], dtype=np.int64)
        inertias = np.asarray([], dtype=np.float64)
    active_labels = _kmeans_labels(
        cluster_features,
        cluster_weight,
        n_clusters=best_k,
        random_state=int(random_state),
        n_init=kmeans_n_init,
    )

    selected_local_ids = []
    selected_labels = []
    per_cluster = int(points_per_cluster)
    if per_cluster <= 0:
        raise ValueError("points_per_cluster must be positive.")
    method = str(selection_method).lower()
    if method not in {"weighted", "sample", "sample_weighted", "top", "ranked"}:
        raise ValueError("selection_method must be 'weighted' or 'top'.")
    rng = np.random.default_rng(int(random_state))
    for label in sorted(np.unique(active_labels)):
        members = np.flatnonzero(active_labels == label)
        member_score = cluster_weight[members]
        n_keep = min(per_cluster, members.size)
        if method in {"top", "ranked"}:
            order = np.lexsort((-active_q_product[members], -member_score))
            keep = members[order[:n_keep]]
        else:
            total_score = float(np.sum(member_score))
            if total_score > 0.0 and np.all(np.isfinite(member_score)):
                probs = member_score / total_score
            else:
                probs = None
            keep = rng.choice(members, size=n_keep, replace=False, p=probs)
            keep = np.sort(keep)
        selected_local_ids.extend(keep.tolist())
        selected_labels.extend([int(label)] * keep.size)
    selected_local = np.asarray(selected_local_ids, dtype=np.int64)
    selected_grid_ids = active_ids[selected_local]

    return FelKdeSelectionResult(
        selected_cv_points=centers[selected_grid_ids].astype(np.float64, copy=True),
        selected_q=q_flat[selected_grid_ids].astype(np.float64, copy=True),
        selected_weights=active_weights[selected_local].astype(np.float64, copy=True),
        selected_cluster_labels=np.asarray(selected_labels, dtype=np.int64),
        selected_grid_indices=selected_grid_ids.astype(np.int64, copy=True),
        grid_centers=centers.astype(np.float64, copy=False),
        grid_density=density_flat.astype(np.float64, copy=True),
        grid_free_energy=free_energy.astype(np.float64, copy=True),
        grid_q=q_flat.astype(np.float64, copy=True),
        grid_q_product=q_product_flat.astype(np.float64, copy=True),
        active_grid_indices=active_ids.astype(np.int64, copy=True),
        active_cluster_labels=active_labels.astype(np.int64, copy=True),
        n_clusters=int(best_k),
        k_values=ks.astype(np.int64, copy=True),
        inertias=inertias.astype(np.float64, copy=True),
    )


def channel_selection_from_fel_result(
    result: FelKdeSelectionResult,
    model_points: np.ndarray,
    pair: tuple[int, int],
    *,
    threshold: float,
) -> ChannelSelection:
    """Convert selected KDE/FEL centers into a gradpath ChannelSelection."""

    model_points = _as_2d_float("model_points", model_points)
    if model_points.shape[0] != result.selected_cv_points.shape[0]:
        raise ValueError("model_points must have one row per selected CV center.")
    state_i, state_j = _validate_pair(pair, result.selected_q.shape[1])
    return ChannelSelection(
        indices=np.arange(model_points.shape[0], dtype=np.int64),
        points=model_points.astype(np.float64, copy=True),
        q=result.selected_q.astype(np.float64, copy=True),
        weights=result.selected_weights.astype(np.float64, copy=True),
        channel_score=(result.selected_q[:, state_i] * result.selected_q[:, state_j]).astype(np.float64, copy=True),
        state_i=state_i,
        state_j=state_j,
        threshold=float(threshold),
    )


def save_fel_selection_npz(path: str, result: FelKdeSelectionResult) -> None:
    """Persist the large grid arrays and selected-center metadata."""

    np.savez_compressed(
        path,
        selected_cv_points=result.selected_cv_points,
        selected_q=result.selected_q,
        selected_weights=result.selected_weights,
        selected_cluster_labels=result.selected_cluster_labels,
        selected_grid_indices=result.selected_grid_indices,
        grid_centers=result.grid_centers,
        grid_density=result.grid_density,
        grid_free_energy=result.grid_free_energy,
        grid_q=result.grid_q,
        grid_q_product=result.grid_q_product,
        active_grid_indices=result.active_grid_indices,
        active_cluster_labels=result.active_cluster_labels,
        n_clusters=np.asarray([result.n_clusters], dtype=np.int64),
        k_values=result.k_values,
        inertias=result.inertias,
    )


def _axis_ids(axis_names: Sequence[str] | None, axes: Sequence[int | str]) -> list[int]:
    out = []
    for axis in axes:
        if isinstance(axis, str):
            if axis_names is None or axis not in axis_names:
                raise ValueError(f"Unknown CV axis {axis!r}.")
            out.append(list(axis_names).index(axis))
        else:
            out.append(int(axis))
    return out


def _project_grid_to_2d(
    centers: np.ndarray,
    density: np.ndarray,
    q_product: np.ndarray,
    axes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = centers[:, axes[0]]
    y = centers[:, axes[1]]
    x_values = np.unique(x)
    y_values = np.unique(y)
    nx, ny = x_values.size, y_values.size
    x_lookup = {float(value): idx for idx, value in enumerate(x_values)}
    y_lookup = {float(value): idx for idx, value in enumerate(y_values)}
    density_proj = np.zeros((nx, ny), dtype=np.float64)
    q_sum = np.zeros((nx, ny), dtype=np.float64)
    q_count = np.zeros((nx, ny), dtype=np.float64)
    for idx in range(centers.shape[0]):
        ix = x_lookup[float(x[idx])]
        iy = y_lookup[float(y[idx])]
        w = float(density[idx])
        density_proj[ix, iy] += w
        q_sum[ix, iy] += float(q_product[idx])
        q_count[ix, iy] += 1.0
    q_proj = q_sum / np.maximum(q_count, 1.0)
    max_density = max(float(np.nanmax(density_proj)), 1.0e-300)
    fel = -np.log(np.maximum(density_proj / max_density, 1.0e-300))
    return x_values, y_values, fel, q_proj


def _optional_float_config(config: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = config.get(key, None)
        if value is not None:
            return float(value)
    return None


def plot_fel_projection(
    result: FelKdeSelectionResult,
    *,
    axes: Sequence[int | str],
    axis_names: Sequence[str] | None,
    save_path: str,
    config: dict[str, Any] | None = None,
) -> str:
    """Project the KDE/FEL grid onto the requested plot CV axes."""

    import matplotlib.pyplot as plt

    cfg = {} if config is None else dict(config)
    axis_ids = _axis_ids(axis_names, axes)
    labels = [str(axis_names[idx]) if axis_names is not None else f"axis {idx}" for idx in axis_ids]
    if len(axis_ids) == 2:
        x, y, fel, q_product = _project_grid_to_2d(
            result.grid_centers,
            result.grid_density,
            result.grid_q_product,
            axis_ids,
        )
        f_max = _optional_float_config(cfg, "F_max", "fel_F_max", "fel_f_max", "free_energy_max")
        if f_max is not None:
            fel = np.minimum(fel, float(f_max))
        fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=160)
        fel_plot = axs[0].pcolormesh(
            x,
            y,
            fel.T,
            shading="nearest",
            cmap=str(cfg.get("fel_cmap", "viridis")),
            vmin=0.0,
            vmax=f_max,
        )
        axs[0].scatter(
            result.selected_cv_points[:, axis_ids[0]],
            result.selected_cv_points[:, axis_ids[1]],
            c=result.selected_cluster_labels,
            s=24.0,
            cmap="tab10",
            edgecolors="black",
            linewidths=0.25,
        )
        axs[0].set_xlabel(labels[0])
        axs[0].set_ylabel(labels[1])
        axs[0].set_title("Projected FEL")
        fig.colorbar(fel_plot, ax=axs[0], label="F / kT")

        q_plot = axs[1].pcolormesh(
            x,
            y,
            q_product.T,
            shading="nearest",
            cmap=str(cfg.get("reaction_tube_cmap", "magma")),
            vmin=0.0,
            vmax=cfg.get("reaction_tube_vmax", 0.25),
        )
        axs[1].scatter(
            result.selected_cv_points[:, axis_ids[0]],
            result.selected_cv_points[:, axis_ids[1]],
            c=result.selected_cluster_labels,
            s=24.0,
            cmap="tab10",
            edgecolors="black",
            linewidths=0.25,
        )
        axs[1].set_xlabel(labels[0])
        axs[1].set_ylabel(labels[1])
        axs[1].set_title("Projected q_i q_j")
        fig.colorbar(q_plot, ax=axs[1], label="q_i q_j")
        if cfg.get("xlim", None) is not None:
            for ax in axs:
                ax.set_xlim(float(cfg["xlim"][0]), float(cfg["xlim"][1]))
        if cfg.get("ylim", None) is not None:
            for ax in axs:
                ax.set_ylim(float(cfg["ylim"][0]), float(cfg["ylim"][1]))
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    if len(axis_ids) == 3:
        ids = np.flatnonzero(result.grid_density > 0.0)
        max_points = int(cfg.get("fel_projection_max_points", cfg.get("max_3d_points", 50000)))
        if max_points > 0 and ids.size > max_points:
            order = np.argsort(result.grid_density[ids])[-max_points:]
            ids = ids[order]
        fig = plt.figure(figsize=(5.8, 5.0), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            result.grid_centers[ids, axis_ids[0]],
            result.grid_centers[ids, axis_ids[1]],
            result.grid_centers[ids, axis_ids[2]],
            c=result.grid_q_product[ids],
            s=float(cfg.get("3d_point_size", 4.0)),
            cmap=str(cfg.get("reaction_tube_cmap", "magma")),
            vmin=0.0,
            vmax=cfg.get("reaction_tube_vmax", 0.25),
            alpha=float(cfg.get("3d_alpha", 0.45)),
            linewidths=0,
            depthshade=bool(cfg.get("3d_depthshade", False)),
        )
        ax.scatter(
            result.selected_cv_points[:, axis_ids[0]],
            result.selected_cv_points[:, axis_ids[1]],
            result.selected_cv_points[:, axis_ids[2]],
            c=result.selected_cluster_labels,
            s=30.0,
            cmap="tab10",
            edgecolors="black",
            linewidths=0.25,
            depthshade=bool(cfg.get("3d_depthshade", False)),
        )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.set_title("FEL grid projected onto plot CVs")
        if cfg.get("xlim", None) is not None:
            ax.set_xlim(float(cfg["xlim"][0]), float(cfg["xlim"][1]))
        if cfg.get("ylim", None) is not None:
            ax.set_ylim(float(cfg["ylim"][0]), float(cfg["ylim"][1]))
        if cfg.get("zlim", None) is not None:
            ax.set_zlim(float(cfg["zlim"][0]), float(cfg["zlim"][1]))
        ax.view_init(elev=float(cfg.get("3d_elev", 24.0)), azim=float(cfg.get("3d_azim", -60.0)))
        fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.75, label="q_i q_j")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    raise ValueError("FEL projection supports two or three plot axes.")


def weighted_average_paths_by_fel_cluster(
    paths: Sequence[np.ndarray],
    cluster_labels: np.ndarray,
    weights: np.ndarray,
    *,
    periods: Sequence[float | None] | None = None,
) -> dict[int, np.ndarray]:
    """Return one weighted average pathway per selected FEL/KDE cluster."""

    path_array = np.asarray(paths, dtype=np.float64)
    labels = np.asarray(cluster_labels, dtype=np.int64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    if path_array.ndim != 3:
        raise ValueError("paths must have shape (n_paths, n_images, n_dim).")
    if labels.shape[0] != path_array.shape[0] or weight_arr.shape[0] != path_array.shape[0]:
        raise ValueError("cluster_labels and weights must match the number of paths.")
    period_list = [None] * path_array.shape[2] if periods is None else [None if item is None else float(item) for item in periods]
    if len(period_list) != path_array.shape[2]:
        raise ValueError(f"periods must have length {path_array.shape[2]}.")
    out: dict[int, np.ndarray] = {}
    for label in sorted(np.unique(labels)):
        ids = np.flatnonzero(labels == label)
        local_weights = weight_arr[ids]
        total = float(np.sum(local_weights))
        if total <= 0.0:
            local_weights = np.ones(ids.size, dtype=np.float64) / max(ids.size, 1)
        else:
            local_weights = local_weights / total
        local_paths = path_array[ids]
        if any(period is not None and period > 0.0 for period in period_list):
            reference = local_paths[0]
            aligned = local_paths.copy()
            for dim, period in enumerate(period_list):
                if period is not None and float(period) > 0.0:
                    p = float(period)
                    delta = ((local_paths[:, :, dim] - reference[None, :, dim] + 0.5 * p) % p) - 0.5 * p
                    aligned[:, :, dim] = reference[None, :, dim] + delta
            out[int(label)] = np.average(aligned, axis=0, weights=local_weights)
        else:
            out[int(label)] = np.average(local_paths, axis=0, weights=local_weights)
    return out


def _example() -> None:
    rng = np.random.default_rng(4)
    cv_a = rng.normal(loc=(-0.7, 0.0), scale=0.20, size=(250, 2))
    cv_b = rng.normal(loc=(0.7, 0.0), scale=0.20, size=(250, 2))
    cv = np.vstack([cv_a, cv_b])
    q0 = 1.0 / (1.0 + np.exp(5.0 * cv[:, 0]))
    q1 = 1.0 - q0
    q = np.stack([q0, q1], axis=1)
    result = select_fel_kde_centers(
        cv,
        q,
        (0, 1),
        bins=30,
        bandwidth_bins=1.0,
        threshold=0.20,
        points_per_cluster=3,
        kmax=4,
    )
    print(f"n_clusters: {result.n_clusters}")
    print("selected_cv_points:")
    print(result.selected_cv_points)


if __name__ == "__main__":
    warnings.simplefilter("default")
    _example()

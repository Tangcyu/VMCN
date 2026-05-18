from __future__ import annotations

from typing import Any

import numpy as np

from ..common.data import CommittorDatasetPack, cv_headers_for_pack


def _list_or_empty(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if value is True:
        raise ValueError("Expected an explicit list, got true.")
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _angle_scale(units: str) -> float:
    value = str(units).lower()
    if value in {"degrees", "degree", "deg"}:
        return 180.0 / np.pi
    if value in {"radians", "radian", "rad"}:
        return 1.0
    raise ValueError("periodic_cv_units must be 'degrees' or 'radians'.")


def _angle_to_radians(values: np.ndarray, units: str) -> np.ndarray:
    value = str(units).lower()
    if value in {"degrees", "degree", "deg"}:
        return values * (np.pi / 180.0)
    if value in {"radians", "radian", "rad"}:
        return values
    raise ValueError("periodic_cv_units must be 'degrees' or 'radians'.")


def has_periodic_cv_projection(input_meta: dict[str, Any]) -> bool:
    """Return true when model inputs can be reprojected to angular CV columns."""

    return (
        str(input_meta.get("model_input_space", "")).lower() == "cv"
        and len(_list_or_empty(input_meta.get("model_periodic_cvs", []))) > 0
    )


def projected_axis_names(input_meta: dict[str, Any]) -> list[str]:
    """Coordinate names used after optional periodic CV reprojection."""

    if has_periodic_cv_projection(input_meta):
        return _list_or_empty(input_meta.get("model_cvs_to_use", input_meta.get("model_cv_headers", [])))
    names = input_meta.get("model_feature_names", None)
    return [str(name) for name in names] if names is not None else []


def _feature_index(feature_names: list[str], name: str) -> int:
    if name not in feature_names:
        raise ValueError(f"Missing model feature {name!r}; cannot convert periodic CV coordinates.")
    return feature_names.index(name)


def model_inputs_to_projected_cv(
    points: np.ndarray,
    input_meta: dict[str, Any],
    *,
    unwrap: bool = False,
) -> np.ndarray:
    """
    Convert model-input coordinates back to selected CV coordinates.

    Periodic CVs are reconstructed with atan2(sin(cv), cos(cv)). Nonperiodic
    CV columns are copied from their model-input column. When no periodic CVs
    were used, the input coordinates are returned unchanged.
    """

    values = np.asarray(points, dtype=np.float64)
    original_ndim = values.ndim
    if original_ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("points must have shape (n_points, n_dim).")
    if not has_periodic_cv_projection(input_meta):
        return values[0].copy() if original_ndim == 1 else values.copy()

    cv_names = projected_axis_names(input_meta)
    feature_names = [str(name) for name in input_meta.get("model_feature_names", [])]
    periodic = set(_list_or_empty(input_meta.get("model_periodic_cvs", [])))
    units = str(input_meta.get("model_periodic_cv_units", "degrees"))
    columns: list[np.ndarray] = []
    for name in cv_names:
        if name in periodic:
            sin_col = values[:, _feature_index(feature_names, f"sin{name}")]
            cos_col = values[:, _feature_index(feature_names, f"cos{name}")]
            angle = np.arctan2(sin_col, cos_col)
            if unwrap:
                angle = np.unwrap(angle)
            columns.append(angle * _angle_scale(units))
        else:
            columns.append(values[:, _feature_index(feature_names, name)])
    out = np.stack(columns, axis=1)
    return out[0] if original_ndim == 1 else out


def projected_cv_to_model_inputs(points: np.ndarray, input_meta: dict[str, Any]) -> np.ndarray:
    """
    Convert selected CV coordinates to model-input coordinates.

    This is mainly used for periodic gradpath endpoints and for re-evaluating
    q along saved angular pathways.
    """

    values = np.asarray(points, dtype=np.float64)
    original_ndim = values.ndim
    if original_ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("points must have shape (n_points, n_cv).")
    if not has_periodic_cv_projection(input_meta):
        return values[0].copy() if original_ndim == 1 else values.copy()

    cv_names = projected_axis_names(input_meta)
    if values.shape[1] != len(cv_names):
        raise ValueError(
            "Periodic CV coordinates must have one column per selected CV "
            f"({len(cv_names)} expected, got {values.shape[1]})."
        )
    periodic = set(_list_or_empty(input_meta.get("model_periodic_cvs", [])))
    units = str(input_meta.get("model_periodic_cv_units", "degrees"))
    columns: list[np.ndarray] = []
    for idx, name in enumerate(cv_names):
        column = values[:, idx]
        if name in periodic:
            angle = _angle_to_radians(column, units)
            columns.extend([np.sin(angle), np.cos(angle)])
        else:
            columns.append(column)
    out = np.stack(columns, axis=1)
    return out[0] if original_ndim == 1 else out


def selected_cv_points(pack: CommittorDatasetPack, input_meta: dict[str, Any]) -> np.ndarray:
    """Return dataset points in selected raw CV coordinates."""

    if pack.cv is None:
        raise RuntimeError("Periodic CV reprojection requires a dataset with saved CV data.")
    cv_headers = cv_headers_for_pack(pack)
    cv_names = projected_axis_names(input_meta)
    missing = [name for name in cv_names if name not in cv_headers]
    if missing:
        raise ValueError(f"Selected CV columns are missing from dataset: {missing}")
    ids = [cv_headers.index(name) for name in cv_names]
    return pack.cv[:, ids].detach().cpu().double().numpy()

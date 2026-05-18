from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelSelection:
    """Points selected in one direct i-j transition channel."""

    indices: np.ndarray
    points: np.ndarray
    q: np.ndarray
    weights: np.ndarray
    channel_score: np.ndarray
    state_i: int
    state_j: int
    threshold: float


def _as_2d_float(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_dim).")
    return array


def _as_1d_float(name: str, value: np.ndarray, n: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},).")
    return array


def normalize_weights(weights: np.ndarray, *, allow_zero: bool = False) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if np.any(~np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if allow_zero:
        if np.any(weights < 0.0):
            raise ValueError("weights must be nonnegative.")
    elif np.any(weights <= 0.0):
        raise ValueError("weights must be positive.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass.")
    return weights / total


def select_channel_points(
    points: np.ndarray,
    q: np.ndarray,
    state_i: int,
    state_j: int,
    *,
    threshold: float = 0.20,
    weights: np.ndarray | None = None,
    max_points: int | None = None,
    seed: int | None = None,
    sample_with_replacement: bool = False,
    selection_power: float = 1.0,
) -> ChannelSelection:
    """
    Select direct-channel points where q_i q_j exceeds a threshold.

    If max_points is provided, points are sampled with probabilities
    proportional to Boltzmann weight * (q_i q_j)**selection_power. The returned
    point weights are the original Boltzmann weights for the selected frames.
    """

    points = _as_2d_float("points", points)
    q = _as_2d_float("q", q)
    if q.shape[0] != points.shape[0]:
        raise ValueError("points and q must have the same number of rows.")
    n_samples, n_states = q.shape
    state_i = int(state_i)
    state_j = int(state_j)
    if state_i == state_j:
        raise ValueError("state_i and state_j must be different.")
    if not (0 <= state_i < n_states and 0 <= state_j < n_states):
        raise ValueError("state_i/state_j are outside the q-vector dimension.")

    if weights is None:
        weights = np.ones(n_samples, dtype=np.float64)
    else:
        weights = _as_1d_float("weights", weights, n_samples)
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative.")

    channel_score = q[:, state_i] * q[:, state_j]
    mask = np.isfinite(channel_score) & (channel_score >= float(threshold)) & (weights > 0.0)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise RuntimeError(
            f"No points satisfy q_{state_i} * q_{state_j} >= {float(threshold):g} with positive weight."
        )

    if max_points is not None and int(max_points) > 0 and indices.size > int(max_points):
        draw_weights = weights[indices] * np.power(channel_score[indices], float(selection_power))
        probs = normalize_weights(draw_weights)
        rng = np.random.default_rng(seed)
        replace = bool(sample_with_replacement)
        indices = rng.choice(indices, size=int(max_points), replace=replace, p=probs)

    return ChannelSelection(
        indices=np.asarray(indices, dtype=np.int64),
        points=points[indices].astype(np.float64, copy=True),
        q=q[indices].astype(np.float64, copy=True),
        weights=weights[indices].astype(np.float64, copy=True),
        channel_score=channel_score[indices].astype(np.float64, copy=True),
        state_i=state_i,
        state_j=state_j,
        threshold=float(threshold),
    )

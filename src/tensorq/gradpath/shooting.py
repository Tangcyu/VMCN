from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .selection import ChannelSelection


ArrayTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class GradientPath:
    """One stitched path connecting state_i to state_j through a channel point."""

    path: np.ndarray
    q_path: np.ndarray
    start_index: int
    weight: float
    channel_score: float
    state_i: int
    state_j: int
    model_path: np.ndarray | None = None


def _torch_nonzero_1d(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask, as_tuple=False).flatten()


def wrap_periodic(point: torch.Tensor, periodic: torch.Tensor | None, lower: float, upper: float) -> torch.Tensor:
    if periodic is None or not bool(torch.any(periodic)):
        return point
    width = float(upper - lower)
    wrapped = lower + torch.remainder(point - lower, width)
    return torch.where(periodic, wrapped, point)


def _model_q(model: torch.nn.Module, point: torch.Tensor) -> torch.Tensor:
    q = model(point.unsqueeze(0))[0]
    if q.ndim != 1:
        raise RuntimeError("Committor model must return shape (batch, n_states).")
    return q


def _expanded_scalar(q: torch.Tensor, target_state: int, eps: float) -> torch.Tensor:
    target = q[..., int(target_state)]
    target = torch.clamp(target, min=float(eps), max=1.0 - float(eps))
    return torch.logit(target)


def shoot_to_state(
    model: torch.nn.Module,
    start: np.ndarray,
    target_state: int,
    *,
    step_size: float = 0.05,
    max_steps: int = 300,
    target_q: float = 0.98,
    min_grad_norm: float = 1e-10,
    normalize_gradient: bool = True,
    expansion: bool = False,
    expansion_eps: float = 1e-6,
    basin_center: np.ndarray | None = None,
    basin_radius: float | None = None,
    noise_scale: float = 0.0,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    periodic: np.ndarray | None = None,
    periodic_lower: float = -180.0,
    periodic_upper: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Follow the gradient of q_target from one point until q_target is high.

    The path is generated in the model input space. Use model_input_space='cv'
    with nonperiodic CVs when the returned coordinates should be directly
    interpreted as CV pathways.
    """

    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    device = torch.device(device or "cpu")
    model = model.to(device=device, dtype=dtype)
    model.eval()

    point = torch.as_tensor(np.asarray(start, dtype=np.float64), dtype=dtype, device=device)
    if point.ndim != 1:
        raise ValueError("start must be one-dimensional.")
    periodic_mask = None
    if periodic is not None:
        periodic_mask = torch.as_tensor(periodic, dtype=torch.bool, device=device)
        if periodic_mask.shape != point.shape:
            raise ValueError("periodic must have the same shape as start.")

    rng = np.random.default_rng(seed)
    coords: list[np.ndarray] = []
    q_values: list[np.ndarray] = []
    target_state = int(target_state)

    for _ in range(int(max_steps) + 1):
        point = point.detach().clone().requires_grad_(True)
        q = _model_q(model, point)
        if target_state < 0 or target_state >= q.shape[0]:
            raise ValueError("target_state is outside the model output dimension.")
        coords.append(point.detach().cpu().double().numpy())
        q_values.append(q.detach().cpu().double().numpy())
        if bool(expansion):
            if basin_center is not None and basin_radius is not None:
                center = torch.as_tensor(np.asarray(basin_center, dtype=np.float64), dtype=dtype, device=device)
                if center.shape != point.shape:
                    raise ValueError("basin_center must have the same dimension as start.")
                if float(torch.linalg.norm(point.detach() - center).detach().cpu()) <= float(basin_radius):
                    break
        elif float(q[target_state].detach().cpu()) >= float(target_q):
            break

        scalar = _expanded_scalar(q, target_state, expansion_eps) if bool(expansion) else q[target_state]
        grad = torch.autograd.grad(scalar, point, retain_graph=False, create_graph=False)[0]
        grad_norm = torch.linalg.norm(grad)
        if not torch.isfinite(grad_norm) or float(grad_norm.detach().cpu()) < float(min_grad_norm):
            break
        direction = grad / grad_norm.clamp_min(float(min_grad_norm)) if normalize_gradient else grad
        if noise_scale > 0.0:
            noise = torch.as_tensor(rng.normal(size=point.shape), dtype=dtype, device=device)
            direction = direction + float(noise_scale) * noise
        with torch.no_grad():
            point = point + float(step_size) * direction
            point = wrap_periodic(point, periodic_mask, float(periodic_lower), float(periodic_upper))

    return np.asarray(coords, dtype=np.float64), np.asarray(q_values, dtype=np.float64)


def _torch_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _as_periodic_mask(periodic: np.ndarray | None, dim: int, device: torch.device) -> torch.Tensor | None:
    if periodic is None:
        return None
    mask = torch.as_tensor(periodic, dtype=torch.bool, device=device)
    if mask.ndim != 1 or mask.numel() != dim:
        raise ValueError("periodic must have shape (n_dim,).")
    return mask


def _shoot_batch_chunk(
    model: torch.nn.Module,
    starts: torch.Tensor,
    target_state: int,
    *,
    step_size: float,
    max_steps: int,
    target_q: float,
    min_grad_norm: float,
    normalize_gradient: bool,
    expansion: bool,
    expansion_eps: float,
    basin_center: torch.Tensor | None,
    basin_radius: float | None,
    noise_scale: float,
    generator: torch.Generator | None,
    periodic_mask: torch.Tensor | None,
    periodic_lower: float,
    periodic_upper: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    current = starts.detach().clone()
    n_paths = int(current.shape[0])
    active = torch.ones(n_paths, dtype=torch.bool, device=current.device)
    lengths = torch.full((n_paths,), int(max_steps) + 1, dtype=torch.long, device=current.device)
    coords_steps: list[torch.Tensor] = []
    q_steps: list[torch.Tensor] = []
    q_last: torch.Tensor | None = None
    target_state = int(target_state)

    for step in range(int(max_steps) + 1):
        active_ids = _torch_nonzero_1d(active)
        if active_ids.numel() == 0:
            break

        active_points = current.index_select(0, active_ids).detach().clone().requires_grad_(True)
        q_active = model(active_points)
        if q_active.ndim != 2:
            raise RuntimeError("Committor model must return shape (batch, n_states).")
        if target_state < 0 or target_state >= q_active.shape[1]:
            raise ValueError("target_state is outside the model output dimension.")
        if q_last is None:
            q_last = torch.zeros(n_paths, q_active.shape[1], dtype=q_active.dtype, device=q_active.device)
        q_last[active_ids] = q_active.detach()
        coords_steps.append(current.detach().clone())
        q_steps.append(q_last.detach().clone())

        if bool(expansion):
            if basin_center is not None and basin_radius is not None:
                distances = torch.linalg.norm(active_points.detach() - basin_center.view(1, -1), dim=1)
                hit = distances <= float(basin_radius)
            else:
                hit = torch.zeros(q_active.shape[0], dtype=torch.bool, device=q_active.device)
        else:
            hit = q_active[:, target_state] >= float(target_q)
        if bool(torch.any(hit)):
            hit_ids = active_ids[hit]
            lengths[hit_ids] = int(step) + 1
            active[hit_ids] = False
        if step == int(max_steps):
            break

        move = ~hit
        if not bool(torch.any(move)):
            continue
        scalar = _expanded_scalar(q_active, target_state, expansion_eps) if bool(expansion) else q_active[:, target_state]
        grad = torch.autograd.grad(
            scalar[move].sum(),
            active_points,
            retain_graph=False,
            create_graph=False,
        )[0][move]
        move_ids = active_ids[move]
        grad_norm = torch.linalg.norm(grad, dim=1)
        good = torch.isfinite(grad_norm) & (grad_norm >= float(min_grad_norm))
        if bool(torch.any(~good)):
            stop_ids = move_ids[~good]
            lengths[stop_ids] = int(step) + 1
            active[stop_ids] = False
        if not bool(torch.any(good)):
            continue

        good_ids = move_ids[good]
        direction = grad[good]
        if normalize_gradient:
            direction = direction / grad_norm[good].clamp_min(float(min_grad_norm)).unsqueeze(1)
        if noise_scale > 0.0:
            noise = torch.randn(
                direction.shape,
                dtype=direction.dtype,
                device=direction.device,
                generator=generator,
            )
            direction = direction + float(noise_scale) * noise
        with torch.no_grad():
            next_points = current.index_select(0, good_ids) + float(step_size) * direction
            next_points = wrap_periodic(next_points, periodic_mask, float(periodic_lower), float(periodic_upper))
            current[good_ids] = next_points

    if q_last is None:
        raise RuntimeError("No batch shooting steps were evaluated.")
    coords = torch.stack(coords_steps, dim=0).detach().cpu().double().numpy()
    q_values = torch.stack(q_steps, dim=0).detach().cpu().double().numpy()
    lengths_np = lengths.detach().cpu().numpy()
    return [coords[: int(lengths_np[i]), i, :] for i in range(n_paths)], [
        q_values[: int(lengths_np[i]), i, :] for i in range(n_paths)
    ]


def shoot_batch_to_state(
    model: torch.nn.Module,
    starts: np.ndarray,
    target_state: int,
    *,
    step_size: float = 0.05,
    max_steps: int = 300,
    target_q: float = 0.98,
    min_grad_norm: float = 1e-10,
    normalize_gradient: bool = True,
    expansion: bool = False,
    expansion_eps: float = 1e-6,
    basin_center: np.ndarray | None = None,
    basin_radius: float | None = None,
    noise_scale: float = 0.0,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    periodic: np.ndarray | None = None,
    periodic_lower: float = -180.0,
    periodic_upper: float = 180.0,
    integration_batch_size: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Batched velocity integration along grad q_target.

    The velocity field is evaluated with autograd on the requested torch device.
    Use device='cuda:0' (or the runner device config) to keep the integration on
    GPU. integration_batch_size chunks very large point sets without changing
    the returned path order.
    """

    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    starts_array = np.asarray(starts, dtype=np.float64)
    if starts_array.ndim != 2:
        raise ValueError("starts must have shape (n_paths, n_dim).")
    if starts_array.shape[0] == 0:
        return [], []

    device = torch.device(device or "cpu")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    starts_tensor = torch.as_tensor(starts_array, dtype=dtype, device=device)
    periodic_mask = _as_periodic_mask(periodic, starts_tensor.shape[1], device)
    basin_center_tensor = None
    if basin_center is not None:
        basin_center_tensor = torch.as_tensor(np.asarray(basin_center, dtype=np.float64), dtype=dtype, device=device)
        if basin_center_tensor.ndim != 1 or basin_center_tensor.numel() != starts_tensor.shape[1]:
            raise ValueError("basin_center must have shape (n_dim,).")
    if bool(expansion) and basin_center_tensor is not None and basin_radius is None:
        raise ValueError("expansion basin stopping requires basin_radius when basin_center is provided.")
    generator = _torch_generator(device, seed)
    batch_size = int(integration_batch_size or starts_tensor.shape[0])
    if batch_size < 1:
        raise ValueError("integration_batch_size must be positive.")

    all_paths: list[np.ndarray] = []
    all_q: list[np.ndarray] = []
    for start in range(0, starts_tensor.shape[0], batch_size):
        end = min(starts_tensor.shape[0], start + batch_size)
        chunk_paths, chunk_q = _shoot_batch_chunk(
            model,
            starts_tensor[start:end],
            target_state,
            step_size=float(step_size),
            max_steps=int(max_steps),
            target_q=float(target_q),
            min_grad_norm=float(min_grad_norm),
            normalize_gradient=bool(normalize_gradient),
            expansion=bool(expansion),
            expansion_eps=float(expansion_eps),
            basin_center=basin_center_tensor,
            basin_radius=None if basin_radius is None else float(basin_radius),
            noise_scale=float(noise_scale),
            generator=generator,
            periodic_mask=periodic_mask,
            periodic_lower=float(periodic_lower),
            periodic_upper=float(periodic_upper),
        )
        all_paths.extend(chunk_paths)
        all_q.extend(chunk_q)
    return all_paths, all_q


def _path_arc(path: np.ndarray) -> np.ndarray:
    if path.shape[0] == 1:
        return np.asarray([0.0], dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


def _reparameterize_values(values: np.ndarray, arc: np.ndarray, num_images: int) -> np.ndarray:
    if values.shape[0] == 1 or float(arc[-1]) <= 0.0:
        return np.repeat(values[:1], num_images, axis=0)
    target = np.linspace(0.0, float(arc[-1]), num_images)
    columns = [np.interp(target, arc, values[:, dim]) for dim in range(values.shape[1])]
    return np.stack(columns, axis=1)


def smooth_path(
    path: np.ndarray,
    *,
    iterations: int = 1,
    window: int = 3,
    preserve_endpoints: bool = True,
) -> np.ndarray:
    """Smooth a path with a centered moving average."""

    out = np.asarray(path, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError("path must have shape (n_images, n_dim).")
    iterations = int(iterations)
    window = int(window)
    if iterations <= 0 or window <= 1 or out.shape[0] <= 2:
        return out.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    for _ in range(iterations):
        padded = np.pad(out, ((half, half), (0, 0)), mode="edge")
        smoothed = np.empty_like(out)
        for idx in range(out.shape[0]):
            smoothed[idx] = np.mean(padded[idx : idx + window], axis=0)
        if preserve_endpoints:
            smoothed[0] = out[0]
            smoothed[-1] = out[-1]
        out = smoothed
    return out


def reparameterize_path(path: np.ndarray, num_images: int) -> np.ndarray:
    """Interpolate a path onto evenly spaced path arc-length images."""

    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2:
        raise ValueError("path must have shape (n_images, n_dim).")
    num_images = int(num_images)
    if num_images < 2:
        raise ValueError("num_images must be at least 2.")
    if path.shape[0] == num_images:
        return path.copy()
    if path.shape[0] == 1:
        return np.repeat(path, num_images, axis=0)
    delta = np.diff(path, axis=0)
    segment = np.linalg.norm(delta, axis=1)
    keep = np.concatenate([[True], segment > 1e-14])
    path = path[keep]
    if path.shape[0] == 1:
        return np.repeat(path, num_images, axis=0)
    arc = _path_arc(path)
    return _reparameterize_values(path, arc, num_images)


def finalize_stitched_path(
    path: np.ndarray,
    q_path: np.ndarray,
    *,
    num_images: int | None = None,
    endpoint_i: np.ndarray | None = None,
    endpoint_j: np.ndarray | None = None,
    smooth_iterations: int = 0,
    smooth_window: int = 3,
    reparameterize_after_smoothing: bool = True,
    state_i: int | None = None,
    state_j: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Attach optional endpoints and apply the standard path resampling steps."""

    path = np.asarray(path, dtype=np.float64)
    q_path = np.asarray(q_path, dtype=np.float64)
    if path.ndim != 2:
        raise ValueError("path must have shape (n_images, n_dim).")
    if q_path.ndim != 2 or q_path.shape[0] != path.shape[0]:
        raise ValueError("q_path must have shape (n_images, n_states).")
    q_dim = q_path.shape[1]
    if endpoint_i is not None:
        endpoint = np.asarray(endpoint_i, dtype=np.float64)
        if endpoint.shape != path[0].shape:
            raise ValueError("endpoint_i must have the same dimension as the path coordinates.")
        endpoint_q = np.zeros((1, q_dim), dtype=np.float64)
        endpoint_q[0, int(state_i if state_i is not None else np.argmax(q_path[0]))] = 1.0
        path = np.vstack([endpoint[None, :], path])
        q_path = np.vstack([endpoint_q, q_path])
    if endpoint_j is not None:
        endpoint = np.asarray(endpoint_j, dtype=np.float64)
        if endpoint.shape != path[-1].shape:
            raise ValueError("endpoint_j must have the same dimension as the path coordinates.")
        endpoint_q = np.zeros((1, q_dim), dtype=np.float64)
        endpoint_q[0, int(state_j if state_j is not None else np.argmax(q_path[-1]))] = 1.0
        path = np.vstack([path, endpoint[None, :]])
        q_path = np.vstack([endpoint_q, q_path])
    if num_images is not None:
        path_raw = path
        keep = np.concatenate([[True], np.linalg.norm(np.diff(path_raw, axis=0), axis=1) > 1e-14])
        path_raw = path_raw[keep]
        q_raw = q_path[keep]
        arc = _path_arc(path_raw)
        path = _reparameterize_values(path_raw, arc, int(num_images))
        q_path = _reparameterize_values(q_raw, arc, int(num_images))
    if int(smooth_iterations) > 0:
        path = smooth_path(
            path,
            iterations=int(smooth_iterations),
            window=int(smooth_window),
            preserve_endpoints=True,
        )
        if num_images is not None and bool(reparameterize_after_smoothing):
            path = reparameterize_path(path, int(num_images))
    return path, q_path


def stitch_channel_path(
    path_to_i: np.ndarray,
    q_to_i: np.ndarray,
    path_to_j: np.ndarray,
    q_to_j: np.ndarray,
    *,
    num_images: int | None = None,
    endpoint_i: np.ndarray | None = None,
    endpoint_j: np.ndarray | None = None,
    smooth_iterations: int = 0,
    smooth_window: int = 3,
    reparameterize_after_smoothing: bool = True,
    state_i: int | None = None,
    state_j: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.flip(np.asarray(path_to_i, dtype=np.float64), axis=0)
    left_q = np.flip(np.asarray(q_to_i, dtype=np.float64), axis=0)
    right = np.asarray(path_to_j, dtype=np.float64)
    right_q = np.asarray(q_to_j, dtype=np.float64)
    path = np.vstack([left, right[1:]]) if right.shape[0] > 1 else left
    q_path = np.vstack([left_q, right_q[1:]]) if right_q.shape[0] > 1 else left_q
    return finalize_stitched_path(
        path,
        q_path,
        num_images=num_images,
        endpoint_i=endpoint_i,
        endpoint_j=endpoint_j,
        smooth_iterations=smooth_iterations,
        smooth_window=smooth_window,
        reparameterize_after_smoothing=reparameterize_after_smoothing,
        state_i=state_i,
        state_j=state_j,
    )


def build_channel_paths(
    model: torch.nn.Module,
    selection: ChannelSelection,
    *,
    step_size: float = 0.05,
    max_steps: int = 300,
    target_q: float = 0.98,
    num_images: int | None = 50,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    normalize_gradient: bool = True,
    expansion: bool = False,
    expansion_eps: float = 1e-6,
    basin_radius_i: float | None = None,
    basin_radius_j: float | None = None,
    noise_scale: float = 0.0,
    seed: int | None = None,
    periodic: np.ndarray | None = None,
    integration_batch_size: int | None = None,
    endpoint_i: np.ndarray | None = None,
    endpoint_j: np.ndarray | None = None,
    smooth_iterations: int = 0,
    smooth_window: int = 3,
    reparameterize_after_smoothing: bool = True,
    attach_endpoints: bool = True,
) -> list[GradientPath]:
    """Shoot all selected points toward i and j, then stitch i -> j pathways."""

    paths: list[GradientPath] = []
    rng = np.random.default_rng(seed)
    seed_i = None if seed is None else int(rng.integers(0, np.iinfo(np.int32).max))
    seed_j = None if seed is None else int(rng.integers(0, np.iinfo(np.int32).max))
    paths_to_i, q_to_i = shoot_batch_to_state(
        model,
        selection.points,
        selection.state_i,
        step_size=step_size,
        max_steps=max_steps,
        target_q=target_q,
        device=device,
        dtype=dtype,
        normalize_gradient=normalize_gradient,
        expansion=expansion,
        expansion_eps=expansion_eps,
        basin_center=endpoint_i,
        basin_radius=basin_radius_i,
        noise_scale=noise_scale,
        seed=seed_i,
        periodic=periodic,
        integration_batch_size=integration_batch_size,
    )
    paths_to_j, q_to_j = shoot_batch_to_state(
        model,
        selection.points,
        selection.state_j,
        step_size=step_size,
        max_steps=max_steps,
        target_q=target_q,
        device=device,
        dtype=dtype,
        normalize_gradient=normalize_gradient,
        expansion=expansion,
        expansion_eps=expansion_eps,
        basin_center=endpoint_j,
        basin_radius=basin_radius_j,
        noise_scale=noise_scale,
        seed=seed_j,
        periodic=periodic,
        integration_batch_size=integration_batch_size,
    )
    for local_idx, (path_i, q_i, path_j, q_j) in enumerate(zip(paths_to_i, q_to_i, paths_to_j, q_to_j)):
        path, q_path = stitch_channel_path(
            path_i,
            q_i,
            path_j,
            q_j,
            num_images=num_images,
            endpoint_i=endpoint_i if bool(attach_endpoints) else None,
            endpoint_j=endpoint_j if bool(attach_endpoints) else None,
            smooth_iterations=smooth_iterations,
            smooth_window=smooth_window,
            reparameterize_after_smoothing=reparameterize_after_smoothing,
            state_i=selection.state_i,
            state_j=selection.state_j,
        )
        paths.append(
            GradientPath(
                path=path,
                q_path=q_path,
                start_index=int(selection.indices[local_idx]),
                weight=float(selection.weights[local_idx]),
                channel_score=float(selection.channel_score[local_idx]),
                state_i=selection.state_i,
                state_j=selection.state_j,
            )
        )
    return paths

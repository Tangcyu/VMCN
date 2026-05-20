# TensorQ Gradpath Scheme

This note describes the gradient path shooting and clustering workflow in `tensorq.gradpath`. It covers point selection, gradient-ascent path shooting, path clustering, and multi-state-pair orchestration.

## Current Philosophy

Gradpath generates reactive paths by following the gradient of a learned committor. Given a `NextHitCommittorNet` and a pair of states `(i, j)`, it selects points in the transition region (where `q_i * q_j` is large), then shoots gradient-ascent paths toward each basin. The forward and backward half-paths are stitched to form complete reactive trajectories.

The resulting paths are clustered by RMSD to produce representative pathway centers. This provides a structural interpretation of the committor: the model learns where the transition region is, and gradpath traces the actual geometric route through configuration space.

Gradpath operates on either raw model features or projected CV space. Periodic CVs are handled via sin/cos embedding in model-input space and via coordinate transformations for path post-processing.

## Entry Points

| Module | CLI | Purpose |
|---|---|---|
| `runner.py` | `python -m tensorq.gradpath.runner --config <yaml>` | Full workflow: selection → shooting → clustering → plotting |
| `plot_runner.py` | `python -m tensorq.gradpath.plot_runner --config <yaml> [--auto]` | Post-hoc plotting of saved paths colored by `q_i * q_j` |
| `state_p.py` | Called programmatically via `run_gradpath_for_state_pairs(config)` | Multi-state-pair orchestration from P_jump matrix |
| `selection.py` | (library) | Point selection based on channel score `q_i * q_j` |
| `fel_selection.py` | (library) | Alternative FEL/KDE-based point selection |
| `shooting.py` | (library) | Gradient-ascent path integration |
| `cluster.py` | (library) | RMSD-based path clustering |
| `coordinates.py` | (library) | Periodic CV coordinate transformations |
| `plot.py` | (library) | 2D/3D path visualization, dendrograms, colored paths |

## Shared Inputs

- **dataset**: `.pt`/`.npz` TensorQ dataset with features and optional CVs.
- **model**: Trained `NextHitCommittorNet` checkpoint (TorchScript or full checkpoint dict).
- **config**: YAML via `common.config.load_yaml` + `select_section`. Section aliases: `GRADPATH`, `GRADPATH_RUN`.
- **q_values**: Either loaded from cache (`Q.npy`) or computed via `next_hit.predict.infer_probabilities`.
- **state pair**: Specified as `(state_i, state_j)` in config, or derived from P_jump matrix for multi-pair runs.

## Point Selection

### Channel Score Selection (selection.py)

The default method. For a given state pair `(i, j)`:

1. Compute `channel_score = q_i * q_j` for all frames.
2. Filter frames where `channel_score >= threshold`.
3. Optionally downsample by `max_points` with probability proportional to `weight * (channel_score)^selection_power`.
4. Return `ChannelSelection` dataclass: indices, points, q values, weights, scores.

The `channel_score` is highest where both `q_i` and `q_j` are ~0.5 — the transition region.

### FEL/KDE Selection (fel_selection.py)

An alternative for when the committor landscape is diffuse:

1. Bin CV space on a regular grid.
2. Estimate density via Gaussian KDE, convert to free energy.
3. Evaluate the committor model at grid centers.
4. Filter grid cells where `q_i * q_j >= threshold`.
5. Cluster active grid centers via KMeans (k chosen by elbow method).
6. Sample points from each cluster.

Returns `FelKdeSelectionResult` with grid data, cluster labels, and selected points.

## Path Shooting

### Single-Path Shooting (shooting.py)

`shoot_to_state(model, start, target_state, ...)`:

1. Starting from `start` point, compute `q_target = model(start)`.
2. Compute gradient `dq_target/dx` via `torch.autograd.grad`.
3. Take a step in the gradient direction with step size `step_size`.
4. Optional gradient normalization and noise injection.
5. Stop when `q_target >= target_q` (in basin) or `max_steps` reached, or `grad_norm < min_grad_norm`.
6. Optional expansion mode: stop when distance from start exceeds `basin_radius` (for exploring the basin interior).

### Batch Shooting

`shoot_batch_to_state(model, starts, target_state, ...)`:

- Integrates multiple starting points concurrently on GPU.
- Uses `integration_batch_size` to chunk very large batches.
- Returns stacked coordinate arrays and q-value arrays.

### Path Stitching

`stitch_channel_path(path_to_i, q_to_i, path_to_j, q_to_j, ...)`:

1. Reverse the path to state i (so it goes from i to the transition region).
2. Append the path to j (removing duplicate midpoint).
3. Optionally reparameterize to uniform arc length with `num_images`.
4. Optionally smooth with centered moving average.
5. Attach exact endpoint coordinates if provided.

`build_channel_paths(model, selection, ...)`:

- High-level function: shoots to both i and j for each selected point, stitches each pair, returns list of `GradientPath` objects.

## Periodic CV Handling (coordinates.py)

For datasets where `model_input_space = "cv"` and some CVs are periodic (e.g., dihedral angles):

- **Forward**: `projected_cv_to_model_inputs` converts angle values to sin/cos pairs for model input.
- **Backward**: `model_inputs_to_projected_cv` reconstructs angles from sin/cos embeddings (via `atan2`).
- **Unwrap**: Optional `np.unwrap` for continuous paths across the periodic boundary.

This is used in `runner.py` via `_project_and_finalize_periodic_paths` to convert model-space paths back to interpretable CV-space paths.

## Path Clustering (cluster.py)

### Pairwise RMSD Matrix

`pairwise_rmsd_matrix(paths, periods)`:

- Reparameterizes all paths to the same number of images.
- Computes RMSD between each pair of paths, respecting periodic boundary conditions via minimum-image convention.
- Returns `(n_paths, n_paths)` distance matrix.

### Clustering

`cluster_paths(paths, weights, distance_threshold, ...)`:

1. Compute RMSD distance matrix.
2. Agglomerative clustering with weighted average linkage.
3. Compute weighted center path for each cluster.
4. Find medoid (path closest to center) for each cluster.
5. Return `PathCluster` objects.

`cluster_paths_with_linkage(paths, ...)`:

- Same as `cluster_paths` but additionally returns a scipy-format linkage matrix for dendrogram plotting.

### Weighted Center Path

`weighted_center_path(paths, weights, periods)`:

- Iteratively computes the weighted average of path coordinates, handling periodicity.
- Used as the representative path for each cluster.

## Full Workflow (runner.py)

`run_gradpath(config)`:

1. Load dataset, committor model, q values.
2. Select points (channel score or FEL/KDE).
3. `build_channel_paths` — shoot to both states, stitch.
4. For periodic CVs: `_project_and_finalize_periodic_paths` (convert model-space paths to CV-space).
5. Save individual paths to `paths/path_XXXX.txt`.
6. `cluster_paths_with_linkage` — cluster paths by RMSD.
7. Save cluster centers to `cluster_centers/cluster_XX_center_path.txt`.
8. Generate plots: selected points, paths with cluster coloring, dendrogram, FEL projection.
9. Write `summary.yaml`.

## Multi-State-Pair Orchestration (state_p.py)

`run_gradpath_for_state_pairs(config)`:

1. Load P_jump matrix (from next-hit rate estimation output).
2. `find_transitions_above_threshold` — find all `(i, j)` pairs with `P_jump[i,j] >= threshold`.
3. Parse state endpoint coordinates from config.
4. For each qualifying pair, call `run_gradpath` with that pair's configuration.
5. Each pair's output lands in a `state_i_j/` subdirectory.

## Plotting (plot.py, plot_runner.py)

**Runner-integrated plots** (called by `runner.py`):
- `plot_selected_points_2d/3d`: Show selected channel points in CV space.
- `plot_paths_2d/3d`: Show paths colored by cluster membership, with center paths highlighted.
- `plot_path_dendrogram`: SciPy dendrogram of the path clustering hierarchy.
- `plot_fel_projection`: FEL grid with active centers highlighted.

**Post-hoc colored path plots** (called by `plot_runner.py`):
- `plot_colored_paths_2d/3d`: Each path is a `LineCollection` colored continuously by `q_i * q_j` along its length. This reveals where along the path the committor changes most rapidly.

## Decision Meaning

- **Many similar paths in a cluster**: The transition is geometrically well-defined. The center path is a reliable reaction coordinate.
- **Diverse paths, many small clusters**: The transition region is broad or the committor is not sharply localized. Consider better CVs or higher model capacity.
- **Paths fail to reach basin**: `target_q` or `max_steps` may need tuning. Check if the model's q values are well-calibrated near the basin.
- **Periodic path discontinuities**: Check that `periodic_cvs` and `periodic_cv_units` are correctly configured in the dataset metadata and gradpath config.

## Important Constraints

- Path shooting uses `torch.autograd.grad` — the model must be in eval mode and gradients must flow through the full network.
- Batch shooting on GPU requires sufficient VRAM for `integration_batch_size` concurrent paths × model activations.
- RMSD clustering with many paths is O(n²). For >10,000 paths, consider subsampling or using a coarser `distance_threshold`.
- The `_project_and_finalize_periodic_paths` function in runner.py assumes the model input space uses sin/cos embedding. Paths shot in raw feature space cannot be projected this way.

## Key Config Knobs

**Selection**:
- `state_i`, `state_j`: which pair to target.
- `threshold`: minimum `q_i * q_j` for point selection.
- `max_points`: maximum selected points (with weighted sampling).
- `selection_power`: exponent for weighting by `channel_score`.
- `selection_method`: `"channel"` or `"fel_kde"`.

**Shooting**:
- `step_size`: gradient ascent step size.
- `max_steps`: maximum integration steps per half-path.
- `target_q`: stop when q_target reaches this value.
- `min_grad_norm`: stop when gradient norm falls below this.
- `normalize_gradient`: whether to use unit-length gradient steps.
- `expansion`, `expansion_eps`, `basin_center`, `basin_radius`: expansion mode settings.
- `noise_scale`: Gaussian noise added per step (for stochastic exploration).

**Stitching**:
- `num_images`: number of uniformly spaced points in final path.
- `smooth_iterations`, `smooth_window`: smoothing parameters.

**Clustering**:
- `distance_threshold`: maximum RMSD for merging clusters.
- `num_images`: images per path for distance computation.
- `periods`: periodic boundary values for RMSD computation.

**Multi-pair** (state_p.py):
- `P_jump_path`: path to P_jump CSV/NPY.
- `P_jump_threshold`: minimum P_jump to consider a pair.
- `state_endpoints.basins`: `{label: [coord1, coord2, ...]}` mapping.

## File Map

- `selection.py`: `ChannelSelection` dataclass, `select_channel_points`, `normalize_weights`.
- `shooting.py`: `GradientPath` dataclass, `shoot_to_state`, `shoot_batch_to_state`, `stitch_channel_path`, `build_channel_paths`, `smooth_path`, `reparameterize_path`, `finalize_stitched_path`.
- `cluster.py`: `PathCluster` dataclass, `path_array`, `pairwise_rmsd_matrix`, `weighted_center_path`, `cluster_paths`, `cluster_paths_with_linkage`.
- `coordinates.py`: `has_periodic_cv_projection`, `projected_axis_names`, `model_inputs_to_projected_cv`, `projected_cv_to_model_inputs`, `selected_cv_points`.
- `fel_selection.py`: `FelKdeSelectionResult` dataclass, `select_fel_kde_centers`, `channel_selection_from_fel_result`, `save_fel_selection_npz`, `plot_fel_projection`, `weighted_average_paths_by_fel_cluster`.
- `plot.py`: Style setup, discrete colormaps, `plot_selected_points_2d/3d`, `plot_paths_2d/3d`, `plot_path_dendrogram`, `plot_colored_paths_2d/3d`.
- `runner.py`: `run_gradpath` (full workflow orchestration), `main` CLI.
- `plot_runner.py`: `run_gradpath_plot`, `find_state_pairs`, `main` CLI.
- `state_p.py`: `parse_state_endpoints`, `load_p_jump`, `find_transitions_above_threshold`, `run_gradpath_for_state_pairs`.
- `__init__.py`: Lazy-loading exports of 22 public symbols via `__getattr__`.

## Current Known Rough Edges

- `ArrayTransform` type alias in `shooting.py:12` is defined but never used anywhere.
- `_project_paths_if_needed` in `runner.py:170` is defined but never called. The active code path uses `_project_and_finalize_periodic_paths` instead.
- `cluster_paths` and `cluster_paths_with_linkage` in `cluster.py` share ~100 lines of near-identical code. The only difference is that `cluster_paths_with_linkage` also records a scipy linkage matrix.
- FEL/KDE selection requires scikit-learn and scipy (lazy-imported). These are not listed as hard dependencies of TensorQ.
- `state_p.py` calls `run_gradpath` programmatically, which modifies global matplotlib state. Running many pairs sequentially can produce stale figure state.

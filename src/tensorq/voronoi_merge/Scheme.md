# TensorQ Voronoi Merge Scheme

This note describes the iterative Voronoi-based pathway merging workflow in `tensorq.voronoi_merge`. It covers sample loading, Voronoi assignment, iterative pathway expansion via exchange counts, and pathway network construction.

## Current Philosophy

Voronoi merge takes a set of pre-computed path images (typically from `gradpath` cluster centers) and refines them against MD trajectory data. The core idea is:

1. Assign each data point to the nearest path image (Voronoi tessellation in path-image space).
2. Count how often the trajectory crosses from one path's Voronoi cell to another's (exchange counts).
3. Merge path images that have high exchange counts into shared segments.
4. Relax path images by averaging with their exchange-connected neighbors.
5. Smooth, reparameterize, and repeat until convergence.

Unlike gradpath, voronoi_merge does not use a committor model. It is purely data-driven: the refinement is based on the actual MD trajectory sampling, not on a learned probability landscape. It answers the question: given candidate reaction paths, do the MD data support them as distinct pathways, or should they be merged?

## Entry Points

| Module | CLI | Purpose |
|---|---|---|
| `runner.py` | `python -m tensorq.voronoi_merge.runner --config <yaml> [--images ...] [--samples ...]` | Full workflow: load data → iterative expansion → final assignment → KLD |
| `core.py` | (library) | Voronoi assignment, distance computation, KL divergence, periodic geometry |
| `io.py` | (library) | Sample loading (`load_samples`), path image loading (`load_images`) |
| `iterative.py` | (library) | Core iterative algorithm: exchange counting, node merging, path relaxation |
| `plot.py` | (library) | Visualization: Voronoi assignments, shared segments, pathway networks |

## Shared Inputs

- **Path images**: Pre-computed path coordinates. Each path is a sequence of `(n_images, n_dims)` points. Loaded from files (`.txt`, `.npy`, `.npz`) or directories via `io.load_images`.
- **Samples**: MD trajectory data points with optional weights and trajectory IDs. Loaded via `io.load_samples` from `.pt`/`.pth`/`.npz`/`.npy` files.
- **Config**: YAML via `common.config`. Section aliases: `VORONOI_MERGE`, `VORONOI`.

### Data Formats

`io.load_samples` supports:
- TensorQ dataset packs (`.pt`/`.pth`) — extracts points from `features` or `cv`.
- NumPy archives (`.npz`) — tries dataset-pack keys first, falls back to raw arrays.
- Plain NumPy arrays (`.npy`).
- Optional separate files for weights and trajectory IDs.

`io.load_images` supports:
- Directory of per-path subdirectories (find medoid/center/all paths).
- Single `.npy` or `.npz` file with stacked path images.
- Optional subsampling via `image_stride` and `max_images_per_path`.

## Periodic Geometry

Both `core.py` and `iterative.py` handle periodic boundary conditions:

- **Minimum image convention**: `minimum_image_delta(diff, periods)` computes the shortest vector between two points across periodic boundaries.
- **Sin/cos embedding**: `_use_sincos_geometry` detects when `periodic_geometry` config requires sin/cos embedding for distance computation. Used when Euclidean distance in angle space is inappropriate (e.g., dihedral angles).
- **Periodic weighted mean**: `periodic_weighted_mean` and `periodic_sincos_weighted_mean` compute averages respecting periodicity.

## Voronoi Assignment (core.py)

`assign_voronoi_cells(points, centers, periods, ...)`:

1. Compute pairwise distances between all `(n_points, n_centers)`.
2. Respect periodic boundaries via minimum-image convention.
3. Optionally use sin/cos embedding for periodic dimensions.
4. Support both GPU (torch) and CPU (numpy) backends, selected automatically based on `device` config.
5. Return `(labels, distances)` — each point gets the index of its nearest center and the distance.

`voronoi_assignment(points, centers, weights, ...)`:

- One-stop function: calls `assign_voronoi_cells` + `cell_probabilities` (computing population per cell).
- Returns `VoronoiAssignment` dataclass: `labels`, `distances`, `probabilities`.

## Iterative Pathway Expansion (iterative.py)

This is the core algorithm, implemented in `run_iterative_pathway_expansion` (~1119 lines).

### Initialization

1. Reparameterize all input paths to a uniform number of images.
2. Optionally apply final image spacing (denser sampling after the last iteration).

### Per-Iteration Loop

Each iteration:

1. **Voronoi Assignment**: `assign_pathway_expansions` — assign each data point to the nearest path image (across all paths).

2. **Exchange Counting**: `cross_path_exchange_counts` — for each lagged pair `(t, t+lag)` within the same trajectory, check if the Voronoi assignment changed between paths. Count how many times trajectory visits cross from path A to path B.

3. **Shared Node Detection**: `shared_node_labels_from_counts` — use a Union-Find data structure to merge path images that have exchange counts above `min_exchange_count` and exchange probability above `min_exchange_probability`.

4. **Segment Decomposition**: `shared_segments_from_node_labels` — convert node-level shared labels to contiguous segments along each path. `decompose_pathway_segments` splits paths into "shared" and "unique" stretches.

5. **Path Relaxation**: `relax_images_by_dynamic_edges` — for each path image that has exchange edges to images on other paths, compute the weighted average of its connected neighbors. This pulls the path toward regions of high exchange.

6. **Smoothing and Reparameterization**: `smooth_reparameterize_paths_independently` — smooth each path independently (moving average), then reparameterize to uniform arc length. Periodic boundaries are respected.

7. **Convergence Check**: Compute `max_shift` (maximum displacement of any path image). If below `convergence_tol`, stop.

### Post-Iteration

- Build `PathwayNetwork`: graph representation with adjacency, start/end nodes, branch points.
- `find_all_reactive_pathways`: DFS enumeration of all routes through the network from start to end nodes.
- Save per-iteration outputs (paths, assignments, exchange edge tables, plots).

## Full Workflow (runner.py)

`run_voronoi_merge(config)`:

1. Load initial path images via `io.load_images`.
2. Load sample data via `io.load_samples`.
3. Determine periodic geometry from config.
4. If `pathway_iteration_enabled`:
   - Call `run_iterative_pathway_expansion` with the full iteration config.
   - This produces refined paths, exchange statistics, and a pathway network.
5. Perform final `voronoi_assignment` on the refined (or original) paths.
6. If previous probabilities are provided, compute KL divergence: `kl_divergence(p_current, p_previous)`.
7. Save output: refined paths, assignments, probabilities, pathway network, summary YAML.

## Plotting (plot.py)

- `plot_pathway_iteration_2d`: Color-coded Voronoi assignment of data points, with path images overlaid.
- `plot_shared_segments`: Paths drawn with shared segments in consistent colors, unique segments as gray dashed lines.
- `plot_pathway_network`: Full network visualization — data points (gray), paths (thin gray), exchange edges (colored by weight), branch points (red stars), start nodes (green triangles), end nodes (blue squares).

## Decision Meaning

- **High exchange count between two paths**: The MD data frequently transitions between these pathways — they likely represent the same physical route. They should be merged.
- **Low exchange, well-separated paths**: The paths represent genuinely distinct reaction mechanisms. Keep them separate.
- **Convergence after few iterations**: The initial path images are already well-aligned with the data.
- **No convergence after `max_iterations`**: The path topology may be fundamentally misaligned with the data. Consider regenerating initial paths with different gradpath parameters.
- **KL divergence between old and new assignments**: Measures how much the Voronoi partitioning changed. Large KLD means the refinement significantly altered the pathway decomposition.

## Important Constraints

- Exchange counts use lagged pairs — these must be trajectory-safe. The `traj_id` array ensures pairs never cross trajectory boundaries.
- The `lag` parameter should match the timescale of interest. Too short and exchange counts capture recrossing noise; too long and genuine pathway switching is missed.
- Periodic geometry must match between path images and sample data. Mismatched `periods` or `periodic_geometry` settings will produce nonsensical distances and assignments.
- GPU acceleration for Voronoi assignment (`device: "cuda"`) requires sufficient VRAM for `(n_points, n_path_images)` distance matrix. For very large datasets (>10M points), use chunked CPU path.
- The Union-Find step in `shared_node_labels_from_counts` can merge path images across the entire network. Large `min_exchange_count` and `min_exchange_probability` values prevent spurious merges.

## Key Config Knobs

**Input**:
- `images`: path to initial path images directory or file.
- `samples`: path to sample data file.
- `samples_stride`: optional stride for subsampling samples.

**Periodic geometry**:
- `periods`: list of period values per dimension (0 for non-periodic).
- `periodic_geometry`: `"minimum_image"` or `"sincos"`.
- `wrap_bounds`: optional `[[min, max], ...]` per dimension.

**Iteration** (`pathway_iteration` section):
- `lag`: frames between t and t+lag for exchange counting.
- `terminal_image_margin`: number of endpoint images excluded from exchange counting.
- `min_exchange_count`: minimum absolute exchange count for sharing two nodes.
- `min_exchange_probability`: minimum exchange probability for sharing two nodes.
- `exchange_weight_mode`: weighting for exchange counts (`"uniform"`, `"weight"`).
- `max_cell_distance`: maximum distance (in image indices) for two nodes to share.
- `max_iterations`: maximum iteration count.
- `convergence_tol`: convergence threshold on max image shift.
- `num_images`: number of images per path after reparameterization.
- `image_spacing`: image spacing mode (e.g., `"arc_length"`).
- `smooth_iterations`, `smooth_window`: smoothing parameters.
- `cell_relaxation`: relaxation strength (0 to 1).
- `fixed_endpoints`: whether path endpoints are fixed during relaxation.

**Output**:
- `out_dir`: output directory.
- `save_iterations`: whether to save per-iteration outputs.
- `save_final_images`: whether to save final refined path images.

## File Map

- `core.py`: `VoronoiAssignment` dataclass, `assign_voronoi_cells`, `cell_probabilities`, `voronoi_assignment`, `kl_divergence`, `minimum_image_delta`, `normalize_periods`, `periodic_weighted_mean`, `periodic_sincos_embed`, `periodic_sincos_project`, `periodic_sincos_weighted_mean`.
- `io.py`: `SampleData` dataclass, `load_samples`, `load_images`, `coarsen_path_images`.
- `iterative.py`: `PathwayIteration` dataclass, `IterativePathwayResult` dataclass, `PathwayNetwork` dataclass, `_UnionFind` class, `run_iterative_pathway_expansion`, `assign_pathway_expansions`, `cross_path_exchange_counts`, `shared_node_labels_from_counts`, `shared_segments_from_node_labels`, `decompose_pathway_segments`, `relax_images_by_dynamic_edges`, `smooth_reparameterize_paths_independently`, `build_pathway_network`, `find_all_reactive_pathways`, `exchange_edge_table`, `wrap_periodic_points`, `reparameterize_path_periodic`, `reparameterize_path_with_geometry`, `smooth_path_periodic`, `unwrap_path`.
- `runner.py`: `run_voronoi_merge`, `main` CLI.
- `plot.py`: `plot_pathway_iteration_2d`, `plot_shared_segments`, `plot_pathway_network`.
- `__init__.py`: Lazy-loading exports of 15 public symbols via `__getattr__`.

## Current Known Rough Edges

- `iterative.py` is the largest module in the project (~1119 lines). The plotting code is inlined in the iteration loop (lines ~976–1066) rather than delegated to `plot.py`. This mixes algorithm logic with visualization.
- `io.py` uses a conditional import of `..common.data` (inside `load_samples`) rather than a module-level import. This is fragile — if the common package structure changes, the error will only surface at runtime.
- `_use_sincos_geometry` in `core.py` has a `_` prefix suggesting it's private, but it is explicitly imported by `iterative.py`. Either make it public or refactor `iterative.py` to not depend on it.
- The sin/cos periodic geometry code in `core.py` overlaps conceptually with `gradpath/coordinates.py`. Both handle conversion between angle and sin/cos representations. Consider extracting shared periodic CV utilities to `common/`.

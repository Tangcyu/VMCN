# TensorQ Common Scheme

This note describes the shared infrastructure provided by `tensorq.common`. All other TensorQ subpackages depend on this layer for configuration loading, dataset I/O, model input selection, and flux computation.

## Current Philosophy

`tensorq.common` is a pure library layer. It contains no workflow-specific logic, no CLI entry points, and no assumptions about which committor variant (next-hit or pairwise) will consume its outputs. The three modules — config, data, flux — are deliberately independent: `config.py` depends only on external libraries, `data.py` depends on `config.torch_load`, and `flux.py` depends only on standard library types and torch/numpy.

## Entry Points

There are no CLI entry points. Every function is a library call consumed by higher-level subpackages.

### config.py

| Function | Purpose |
|---|---|
| `load_yaml(path)` | Load and validate a YAML config file |
| `select_section(raw, *names)` | Extract a named subsection by trying multiple keys |
| `ensure_dir(path)` | `mkdir -p` equivalent, returns path string |
| `setup_device(device_str)` | Resolve torch device, fallback to CPU |
| `set_seed(seed)` | Seed Python `random`, `numpy`, and `torch` |
| `torch_load(path, map_location="cpu")` | Load PyTorch checkpoint with `weights_only` fallback |
| `write_yaml(data, path)` | Serialize dict to YAML file |

### data.py

| Function / Class | Purpose |
|---|---|
| `CommittorDatasetPack` (dataclass) | Container for features, weights, state, traj_id, cv, meta |
| `load_dataset(path)` | Load `.pt`/`.pth`/`.npz` into a `CommittorDatasetPack` |
| `apply_stride(pack, stride)` | Subsample all tensors in a pack |
| `cv_headers_for_pack(pack)` | Extract CV column headers from metadata |
| `featurize_cv_inputs(cv, ...)` | Select CV columns, apply sin/cos for periodic CVs |
| `select_model_inputs(pack, config)` | Route to raw features or CV featurization |
| `infer_n_states(pack, requested)` | Determine n_states from config, metadata, or max label |
| `build_lagged_indices(n_frames, lag, traj_id)` | Build t and t+lag index pairs with trajectory safety |
| `split_train_val(n, val_ratio, seed)` | Random train/validation split |
| `LaggedCommittorDataset` (Dataset) | Yields `{z_t, z_tau, weight, state_t, state_tau}` |
| `PairwiseLaggedDataset` (Dataset) | Yields `{z_t, z_tau, weight, pair_label_t, pair_label_tau}` |
| `IndexSubset` (Dataset) | Wraps a base Dataset at specified indices |
| `pair_labels_from_state(state, n_states)` | Build pairwise endpoint labels (-1/0/1) |

### flux.py

| Function | Purpose |
|---|---|
| `unordered_pairs(n_states)` | All (i, j) pairs with i < j |
| `all_ordered_pairs(n_states)` | All (i, j) ordered pairs with i != j |
| `resolve_ordered_pairs(n_states, adjacency)` | Ordered pairs from adjacency or full set |
| `make_thresholds(values, n_thresholds, ...)` | Threshold tensor for flux binning |
| `reactive_current(q_t, q_tau, pairs)` | C_ij = q_j(t+tau) q_i(t) - q_j(t) q_i(t+tau) |
| `crossing_weight(q_t, q_tau, pairs, thresholds, eps)` | Smooth isocommittor crossing indicator |
| `flux_profiles(q_t, q_tau, pairs, thresholds, ...)` | Weighted flux profiles J_ij(c) and C_ij |
| `flux_consistency_loss(q_t, q_tau, pairs, ...)` | Variance-based flux consistency loss |

## Shared Inputs

All subpackages consume these inputs through `common`:

- **YAML config files**: Loaded via `load_yaml` + `select_section`. Multiple section name aliases are tried for backward compatibility (e.g., `NEXT_HIT_COMMITTOR`, `NEXT_HIT_TRAIN`, `TRAIN`).
- **TensorQ datasets**: `.pt`/`.pth` (PyTorch dict) or `.npz` (NumPy archive). Must contain `features` (2D float), `weights` (1D float), `meta_state` (1D int). Optional: `cv` (2D float), `traj_id` (1D int).
- **Model checkpoints**: PyTorch files loaded via `torch_load` with `weights_only` fallback for compatibility.

## Data Flow

```
YAML config → load_yaml() + select_section()
    ↓
dataset_path → load_dataset() → CommittorDatasetPack
    ↓
apply_stride() → subsampled pack
    ↓
select_model_inputs() → features tensor + input_meta dict
    ↓
infer_n_states() → n_states
    ↓
build_lagged_indices() → (idx_t, idx_tau)
    ↓
LaggedCommittorDataset / PairwiseLaggedDataset → DataLoader
```

During training, `flux_consistency_loss` is called on model outputs:
```
model(z_t), model(z_tau)
    ↓
flux_consistency_loss() → (J, variance, C)
    uses: make_thresholds(), resolve_ordered_pairs(), flux_profiles()
```

## Config Knobs (per module)

**config.py**: No knobs — pure utilities.

**data.py knobs** (read from higher-level config sections):
- `model_input_space`: `"features"` or `"cv"`
- `cvs_to_use`: list of CV column names or indices
- `periodic_cvs`, `periodic_cv_units`: sin/cos embedding configuration
- `dataset_stride`: integer stride for subsampling
- `lag`: number of frames between t and t+tau
- `allow_cross_traj_pairs`: whether lagged pairs may cross trajectory boundaries
- `require_labeled`: `"none"`, `"any"`, `"both"` for dataset filtering
- `val_ratio`: fraction of data for validation

**flux.py knobs**:
- `n_thresholds`, `flux_surface`, `eps`: threshold grid and crossing surface shape
- `scale_by_tau`: divide flux by lag time
- `flux_eps`: smoothing parameter for crossing indicator

## File Map

- `__init__.py`: Currently a docstring only. No re-exports.
- `config.py`: YAML, device, seed, checkpoint utilities. No internal dependencies.
- `flux.py`: Torch flux profiles, reactive current, crossing weight. Plus one unused numpy variant.
- `data.py`: Dataset loading, transformations, PyTorch Dataset classes. Depends on `config.torch_load`.

## Current Known Rough Edges

- `common/__init__.py` is effectively empty — consumers must import from submodules directly (`from tensorq.common.config import load_yaml` rather than `from tensorq.common import load_yaml`).
- `unordered_pairs` is duplicated in `data.py:324` and `flux.py:13`. Both implementations are identical. The `data.py` version exists for internal use by `pair_labels_from_state`, while `flux.py`'s is the canonical definition.
- `flux_profiles_numpy` in `flux.py:173` is defined but has zero callers in the entire codebase. It was likely a development prototype superseded by per-subpackage numpy flux implementations.
- `data.py` contains a lazy `import yaml` inside `load_dataset` for NPZ file handling, while `config.py` imports yaml at module level. This asymmetry is deliberate (NPZ path is rarely taken) but inconsistent.

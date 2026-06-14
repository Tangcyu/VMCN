# TensorQ Package Scheme

This note is the top-level overview of the `tensorq` package. It describes the package structure, the relationship between subpackages, and the typical workflow from raw MD trajectories to rate constants and reactive pathways.

## Current Philosophy

TensorQ implements a committor-vector approach to rare-event kinetics. Instead of building a Markov state model (MSM) and computing committors from the transition matrix, TensorQ trains neural networks to directly predict committor probabilities from molecular configurations. This is the "next-hit" formulation: `q_i(x)` is the probability that a trajectory initiated at configuration `x` will next hit state `i` before any other labeled state.

The package is organized as a pipeline:

1. **Label** trajectories into metastable states.
2. **Train** a committor neural network on lagged pairs.
3. **Infer** committor values on the full dataset.
4. **Estimate rates** from committor-derived flux profiles.
5. **Trace reaction paths** by following committor gradients.
6. **Refine paths** by aligning them with MD data via Voronoi tessellation.
7. **Diagnose and relabel** states when the committor disagrees with current labels.

## Package Structure

```
tensorq/
├── common/          Shared config, data I/O, flux computation
├── next_hit/        Multi-state next-hit committor (primary formulation)
├── pairwise/        Pair-wise committor (alternative formulation)
├── gradpath/        Gradient path shooting and clustering
├── voronoi_merge/   Iterative Voronoi-based pathway refinement
├── relabel/         Label diagnostics and relabeling
└── __init__.py      Lazy exports of NextHitCommittorNet, PairwiseCommittorNet
```

## Subpackage Relationships

```
                    ┌──────────────┐
                    │   common/    │
                    │ config, data,│
                    │    flux      │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼──────┐    │    ┌───────▼────────┐
     │   next_hit/   │    │    │   pairwise/    │
     │ train, infer, │    │    │ train, infer,  │
     │ rate, fit,    │    │    │ rate, plot     │
     │ plot, label   │    │    │                │
     └───────┬───────┘    │    └────────────────┘
             │            │
             │     ┌──────▼──────┐
             │     │  gradpath/  │
             │     │ shoot,      │
             │     │ cluster     │
             │     └──────┬──────┘
             │            │
             │     ┌──────▼───────┐
             │     │ voronoi_merge│
             │     │ iterative    │
             │     │ refinement   │
             │     └──────────────┘
             │
      ┌──────▼──────┐
      │  relabel/   │
      │ diagnostics,│
      │ relabel     │
      └─────────────┘
```

**Key dependency rules**:
- `common/` depends on nothing internal; it is the foundation.
- `next_hit/` and `pairwise/` depend only on `common/`. They do not import from each other.
- `gradpath/` depends on `common/` and `next_hit/` (needs a trained committor model for gradient shooting).
- `voronoi_merge/` depends on `common/` only. It takes path images as file inputs, not Python imports.
- `relabel/` depends on `common/` and `next_hit/` (needs committor predictions for label diagnostics).

## Typical Workflow

### Path A: Next-Hit Committor → Rates

```
MD trajectories
    ↓ next_hit/label.py
dataset.pt (features, meta_state, weights)
    ↓ next_hit/train.py
NextHitCommittorNet checkpoint
    ↓ next_hit/infer.py
Q.npy (q values for all frames)
    ↓ next_hit/rate_constant.py
rate_constants.csv, K.npy, MFPT.npy
    ↓ next_hit/fit_rate.py (optional SCF refinement)
K_fit.npy
```

### Path B: Committor → Reaction Paths

```
NextHitCommittorNet checkpoint
    ↓ gradpath/runner.py
paths/path_XXXX.txt, cluster_centers/
    ↓ voronoi_merge/runner.py
refined paths, pathway network, exchange statistics
```

### Path C: Label Diagnostics and Refinement

```
NextHitCommittorNet checkpoint + dataset
    ↓ relabel/main.py (diagnose)
diagnostic_summary.yaml
    ↓ relabel/relabel.py
relabel_summary.yaml, optional relabeled_dataset.pt
    ↓ retrain model with improved labels
```

## Choosing Between next_hit and pairwise

| Scenario | Recommendation |
|---|---|
| Standard use case | `next_hit` — simpler output, flux consistency loss, full error analysis |
| Many states (>10) | `pairwise` — independent modeling per pair can be more robust |
| Strong asymmetry between state pairs | `pairwise` — each pair gets its own model capacity |
| Need SCF rate fitting | `next_hit` only (`fit_rate.py` has no pairwise counterpart) |
| Need flux consistency during training | `next_hit` only |

## Config System

All subpackages use the same config pattern:

1. Load YAML via `common.config.load_yaml`.
2. Extract the relevant section via `common.config.select_section(raw, *aliases)`.
3. Multiple section name aliases are tried for backward compatibility (e.g., a training module tries `NEXT_HIT_COMMITTOR`, then `NEXT_HIT_TRAIN`, then `TRAIN`).

Example config structure:
```yaml
TENSORQ_LABEL:       # consumed by next_hit/label.py
  ...

NEXT_HIT_COMMITTOR:  # consumed by next_hit/train.py
  ...

NEXT_HIT_INFER:      # consumed by next_hit/infer.py
  ...

GRADPATH:            # consumed by gradpath/runner.py
  ...

VORONOI_MERGE:       # consumed by voronoi_merge/runner.py
  ...

RELABEL:             # consumed by relabel/
  ...
```

## Dataset Format

The canonical dataset is a dict (saved as `.pt`/`.pth`) or NPZ archive with these keys:

| Key | Shape | Type | Required |
|---|---|---|---|
| `features` | `(n_frames, n_features)` | float32 | Yes |
| `weights` | `(n_frames,)` | float32 | Yes |
| `meta_state` | `(n_frames,)` | int64 | Yes |
| `cv` | `(n_frames, n_cv)` | float32 | No |
| `traj_id` | `(n_frames,)` | int64 | No (default: single trajectory) |
| `meta` | dict | — | No (metadata like CV headers, periodicity) |

Label value `-1` means unlabeled. Lagged pairs never cross trajectory boundaries (enforced by `build_lagged_indices` in `common/data.py`).

## Model Architecture (Common Patterns)

Both `NextHitCommittorNet` and `PairwiseCommittorNet` share:
- MLP backbone with configurable hidden layers
- ELU activation (default)
- Optional dropout and batch normalization
- Adam optimizer with weight decay
- Early stopping on validation loss
- AMP (automatic mixed precision) support
- GPU-resident data option for large datasets

## GPU Acceleration

- **Training**: Both `GpuLaggedBatcher` (next_hit) and `GpuPairwiseLaggedBatcher` (pairwise) pre-load the entire dataset to GPU to eliminate per-batch CPU→GPU transfers.
- **Inference**: `infer_probabilities` / `infer_pairwise` support batched GPU inference.
- **Voronoi merge**: `assign_voronoi_cells` supports GPU-accelerated distance computation via torch.
- **Gradpath shooting**: `shoot_batch_to_state` integrates multiple paths concurrently on GPU via `torch.autograd.grad`.
- **Flux computation**: `flux_profiles` runs on GPU (torch); per-subpackage numpy implementations run on CPU.

## Important Constraints

- **Lagged pairs must never connect different trajectories.** All modules that build lagged pairs (`build_lagged_indices`, `cross_path_exchange_counts`) enforce this via `traj_id`.
- **Frames are not assumed to be sorted by trajectory.** Always use `traj_id` for trajectory-aware operations.
- **Unlabeled frames have `meta_state = -1`.** Downstream plots and analyses should handle or filter these.
- **Periodic CVs** must be configured consistently across labeling, training, gradpath, and voronoi_merge. Mismatched `periodic_cvs` / `periodic_cv_units` settings will produce incorrect results in path shooting and Voronoi distance computations.
- **The two committor formulations are independent.** A `PairwiseCommittorNet` cannot be used for gradpath shooting (which requires per-state q gradients).

## File Map

### Subpackage Scheme.md files
- [common/Scheme.md](common/Scheme.md): Shared config, data, and flux infrastructure.
- [next_hit/Scheme.md](next_hit/Scheme.md): Multi-state next-hit committor workflow.
- [pairwise/Scheme.md](pairwise/Scheme.md): Pair-wise committor workflow.
- [gradpath/Scheme.md](gradpath/Scheme.md): Gradient path shooting and clustering.
- [voronoi_merge/Scheme.md](voronoi_merge/Scheme.md): Iterative Voronoi-based pathway refinement.
- [relabel/Scheme.md](relabel/Scheme.md): Label diagnostics and relabeling.

### Subpackage Plan.md files
- [common/Plan.md](common/Plan.md): Dead code removal and import centralization plan.
- [next_hit/Plan.md](next_hit/Plan.md): Review notes.
- [pairwise/Plan.md](pairwise/Plan.md): Review notes.
- [gradpath/Plan.md](gradpath/Plan.md): Dead code removal and cluster deduplication plan.
- [voronoi_merge/Plan.md](voronoi_merge/Plan.md): Review notes and structural suggestions.
- [relabel/Plan.md](relabel/Plan.md): Review of existing relabel code.

## Current Known Rough Edges

- `common/__init__.py` is empty — all imports must specify submodules (`from tensorq.common.config import ...`).
- `common/flux.py` contains an unused `flux_profiles_numpy` function.
- `common/data.py` duplicates `unordered_pairs` from `flux.py`.
- `gradpath/shooting.py` has an unused `ArrayTransform` type alias.
- `gradpath/runner.py` has an uncalled `_project_paths_if_needed` function.
- `gradpath/cluster.py` has ~100 lines duplicated between `cluster_paths` and `cluster_paths_with_linkage`.
- `voronoi_merge/iterative.py` inlines plotting code in the main algorithm loop.
- Relabeling now uses a single `scripts/relabel.py` entry point.
- `next_hit/rate_constant.py` is ~1500 lines and mixes estimation, error analysis, CSV writing, and plotting.
- No test suite exists in the repository.

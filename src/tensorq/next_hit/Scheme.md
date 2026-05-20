# TensorQ Next-Hit Committor Scheme

This note describes the multi-state next-hit committor workflow in `tensorq.next_hit`. It covers dataset labeling, model training, inference, rate estimation, SCF fitting, and plotting.

## Current Philosophy

The next-hit committor model predicts `q_i(x)`, the probability that configuration `x` reaches state `i` before any other labeled state. The output is a normalized probability vector over all `n_states` (via softmax or positive-L1). This is the primary committor formulation in TensorQ — the pairwise formulation in `tensorq.pairwise` is an alternative approach.

The training loss combines three terms: a **Dirichlet loss** on lagged pairs that enforces the Chapman-Kolmogorov property, a **boundary loss** that anchors labeled frames to their known states, and an optional **flux consistency loss** that penalizes variance in the reactive flux profile across isocommittor surfaces.

## Entry Points

Each module has a `main()` function protected by `if __name__ == "__main__"`, making them invocable as `python -m tensorq.next_hit.<module> --config <yaml>`.

| Module | Config Section Aliases | Purpose |
|---|---|---|
| `label.py` | `TENSORQ_LABEL`, `NEXT_HIT_LABEL`, `LABEL` | Build datasets from trajectories or relabel existing ones |
| `train.py` | `NEXT_HIT_COMMITTOR`, `NEXT_HIT_TRAIN`, `TRAIN` | Train `NextHitCommittorNet` |
| `infer.py` | `NEXT_HIT_INFER`, `INFER` | Run inference: compute q values from a trained model |
| `predict.py` | (library only) | Load model, batch inference, probability checks |
| `rate_constant.py` | `NEXT_HIT_RATE`, `RATE_CONSTANT` | Estimate rate constants from q values |
| `fit_rate.py` | `NEXT_HIT_FIT_RATE`, `FIT_RATE` | SCF fit a rate generator to flux and branching observables |
| `plot.py` | `NEXT_HIT_PLOT`, `PLOT` | Plot q distributions, CV fields, reaction tubes, flux profiles |

## Shared Inputs

Every workflow loads:

- **dataset**: A `.pt`/`.npz` TensorQ dataset with model features, CVs, trajectory ids, weights, and `meta_state` labels.
- **config**: YAML file read via `common.config.load_yaml` + `select_section`. Multiple section name aliases are tried per module for backward compatibility.
- **model** (train/infer/plot/rate): A `NextHitCommittorNet` checkpoint, either as a TorchScript file or a full training checkpoint dict.
- **model_input_space / cvs_to_use**: Passed to `common.data.select_model_inputs`.

Model evaluation uses `predict.infer_probabilities`, which returns `q_values` of shape `(n_frames, n_states)`.

## Model Architecture

`NextHitCommittorNet` is an MLP defined in `model.py`:

- **Input**: `z` vector of dimension `in_dim` (raw features or selected CVs).
- **Hidden layers**: Configurable tuple, default `(256, 256, 128)`.
- **Activation**: ELU by default; ReLU, tanh, and SiLU also supported.
- **Regularization**: Optional dropout and batch normalization.
- **Output**: `n_states` logits followed by normalization — `"softmax"` (default) or `"positive_l1"`.
- **Lazy import**: The root `tensorq.__init__.py` exports `NextHitCommittorNet` via `__getattr__` for `from tensorq import NextHitCommittorNet`.

## Training Workflow

Code path: `train.py:main()` → `train_next_hit_committor(config)`.

1. Load dataset, apply stride, select model inputs, infer n_states.
2. Build `LaggedCommittorDataset` with lagged index pairs (trajectory-safe by default).
3. Split into train/validation via `IndexSubset`.
4. Instantiate `NextHitCommittorNet`, optimizer (Adam), optional AMP GradScaler.
5. Optionally pre-load all data to GPU via `GpuLaggedBatcher`.
6. For each epoch, `run_epoch()` computes:
   - `total_committor_loss`: `lambda_dir * dirichlet_loss + lambda_bc * boundary_loss + lambda_flux * flux_consistency_loss`
   - Optional `endpoint_boundary_loss` when both t and t+tau frames are labeled.
   - Metrics: `endpoint_boundary_accuracy`, `mean_entropy`, `normalization_error`.
7. Early stopping on validation loss with patience.
8. Save best checkpoint + TorchScript model export.

### Loss Components (losses.py)

- **Dirichlet loss**: Penalizes deviation from the Chapman-Kolmogorov identity `q(t) ≈ q(t+tau)` for unlabeled lagged pairs.
- **Boundary loss**: Anchors labeled frames to their state via cross-entropy or MSE on `q[state_label]`.
- **Endpoint boundary loss**: When both t and t+tau frames are labeled, enforces that the t+tau label matches the committor prediction at t.
- **Flux consistency loss** (flux.py → losses.py): For each ordered pair (i, j), bins the reactive current `C_ij` across isocommittor thresholds and penalizes variance. This enforces that the committor produces smooth, physically meaningful flux profiles.

## Inference Workflow

Code path: `infer.py:main()` → `run_inference(config)`.

1. Load dataset and model.
2. `predict.infer_probabilities` runs batched forward passes, returns `(n_frames, n_states)` numpy array.
3. Compute `q_argmax` per frame and optional boundary summary.
4. Save `Q.npy`, `destination_argmax.npy`, optional CSV with CV columns.

## Rate Estimation Workflow

Code path: `rate_constant.py:main()` → `run(config)` (the largest module at ~1500 lines).

1. Load or infer q values.
2. Build lagged pairs via `build_lagged_indices`.
3. Compute flux profiles `J_ij(c)` via `estimate_flux_profiles` (supports both torch and numpy paths).
4. Estimate equilibrium populations `pi_i` from committor-weighted counts or labeled state fractions.
5. Estimate transition hit matrix `T_hit` — fraction of transitions from i that hit j first.
6. `matrix_from_pair_values` assembles full `(n_states, n_states)` matrices.
7. `assemble_generator` builds the rate generator K from off-diagonal flux and populations.
8. `compute_mfpt_matrix` and `compute_jump_probabilities` derive MFPTs and jump probabilities.
9. Error analysis: slice-based standard deviations via `estimate_slice_rate_std`.
10. Output: CSV tables for k_direct, K, MFPT, P_jump, pi; flux profile plots; heatmaps.

## SCF Rate Fitting Workflow

Code path: `fit_rate.py:main()` → `run(config)`.

1. Takes flux profiles J_ij, populations pi, and branching matrix P_branch from `rate_constant.py` outputs.
2. Iteratively fits off-diagonal rate matrix elements to minimize flux + branching RMSE.
3. Optional detailed balance projection via `detailed_balance_projection`.
4. Output: fitted rate matrix, convergence history plots, scatter plots comparing observed vs fitted.

## Plotting Workflow

Code path: `plot.py:main()` → `run(config)`.

- **q distributions**: Histograms of `q_i` per state, reaction tube distributions (frames with `q_i > threshold`).
- **CV fields**: 2D/3D binned averages of `q_i` over CV space with basin overlays.
- **Reaction tube networks**: 2D/3D visualization of transitions between states in CV space.
- **Flux profiles**: J_ij(c) per pair vs isocommittor threshold.

## Dataset Labeling Workflow

Code path: `label.py:main()` → `run(config)`.

1. Load MD trajectories via MDAnalysis (DCD + topology) or read colvars files.
2. Optionally extract internal coordinates (distances, angles, dihedrals via min-Z-matrix).
3. Build clustering matrix from features or CVs (with optional PCA).
4. KMeans clustering with elbow method for k selection.
5. RiteWeight reweighting to correct non-equilibrium sampling.
6. User-defined basin labeling via CV thresholds.
7. Save dataset as `.pt` or `.npz` with features, weights, meta_state, cv, traj_id.

## Decision Meaning

- **High q_i with low entropy**: Frame is confidently assigned to state i.
- **q_i ~ q_j for i ≠ j**: Frame is in the transition region between i and j.
- **Diffuse q (high entropy)**: Frame is in a broad transition region, poorly described by current features, or belongs to an unlabeled state.
- **Flux profile sharpness**: A sharp J_ij(c) peak indicates a well-defined transition state ensemble; a broad/flat profile suggests a diffuse barrier.

## Important Constraints

- Lagged pairs must never connect different trajectories (enforced by `build_lagged_indices`).
- The model outputs must sum to 1 — `check_probability_rows` validates this after inference.
- `require_labeled` filtering in `LaggedCommittorDataset` ensures correct handling of partially labeled datasets.
- GPU-resident data (`gpu_resident_data: true`) requires sufficient GPU memory for the entire dataset.

## Key Config Knobs

**Model**:
- `in_dim`, `n_states`, `hidden`, `activation`, `dropout`, `batch_norm`
- `output_activation`: `"softmax"` or `"positive_l1"`

**Training**:
- `lr`, `weight_decay`, `epochs`, `patience`, `batch_size`
- `use_amp`, `gpu_resident_data`, `num_workers`
- `lag`, `require_labeled`, `allow_cross_traj_pairs`
- `val_ratio`, `val_metric`

**Loss weights**:
- `lambda_dir`, `lambda_bc`, `lambda_flux`
- `weighted_dirichlet`, `weighted_boundary`, `weighted_flux`
- `boundary_mode`: `"ce"` or `"mse"`

**Rate estimation**:
- `lag`, `lag_reference`, `frame_time`, `time_unit`
- `n_thresholds`, `flux_surface`, `flux_eps`, `pi_mode`
- `chunk_size`, `divide_by_tau`, `weighted`

**SCF fitting**:
- `max_iter`, `tol`, `lambda_reg`, `enforce_detailed_balance`

## File Map

- `model.py`: `NextHitCommittorNet` MLP.
- `train.py`: Training loop, checkpointing, GPU batcher, CLI.
- `predict.py`: Model loading, batch inference, probability checks.
- `infer.py`: Inference orchestration, boundary summary, CSV output.
- `losses.py`: Dirichlet, boundary, endpoint boundary, total loss.
- `metrics.py`: Entropy, normalization error, boundary accuracy.
- `rate_constant.py`: Flux profiles, transition hit matrix, MFPT, rate matrices, error analysis.
- `fit_rate.py`: SCF generator fitting with detailed balance option.
- `plot.py`: q distributions, CV fields, reaction tubes, flux profiles.
- `label.py`: Trajectory loading, KMeans, RiteWeight, dataset serialization.

## Current Known Rough Edges

- `rate_constant.py` is very large (~1500 lines) and mixes flux estimation, matrix assembly, error propagation, CSV writing, and plotting. Consider splitting into `flux_estimation.py`, `rate_matrix.py`, and `rate_plot.py`.
- `label.py` has hard dependencies on MDAnalysis and scikit-learn that are lazily imported — runtime errors occur only when those code paths are hit.
- The flux computation has both torch and numpy code paths with subtle differences in chunking and weighting. The numpy path in `next_hit/rate_constant.py` is distinct from the unused `flux_profiles_numpy` in `common/flux.py`.
- `fit_rate.py` has no pairwise counterpart — the SCF fitting is only available for the next-hit formulation.

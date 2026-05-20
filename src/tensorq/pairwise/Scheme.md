# TensorQ Pairwise Committor Scheme

This note describes the pair-wise committor workflow in `tensorq.pairwise`. It runs parallel to `tensorq.next_hit` as an alternative committor formulation.

## Current Philosophy

The pairwise committor model predicts `q_ij(x)`, the probability that configuration `x` reaches state `j` before state `i`, for each unordered pair `(i, j)` with `i < j`. Each pair has an independent sigmoid output — there is no normalization constraint across pairs. Full state probabilities `P_j(x)` are reconstructed from the pairwise values via a linear pseudo-inverse system.

The pairwise formulation can be advantageous when:
- The number of states is large (training one model for all pairs vs one model for all states).
- Pairs have very different transition barriers and benefit from independent modeling.
- The softmax normalization of next-hit committors is too constraining for diffuse transition regions.

## Entry Points

Each module has a `main()` function invocable as `python -m tensorq.pairwise.<module> --config <yaml>`.

| Module | Config Section Aliases | Purpose |
|---|---|---|
| `train.py` | `PAIRWISE_COMMITTOR`, `PAIRWISE_TRAIN`, `TRAIN` | Train `PairwiseCommittorNet` |
| `infer.py` | `PAIRWISE_INFER`, `INFER` | Run inference: compute pairwise Q and reconstruct P |
| `predict.py` | (library only) | Load model, batch inference, state probability reconstruction |
| `rate_constant.py` | `PAIRWISE_RATE`, `RATE_CONSTANT` | Estimate rate constants from P |
| `plot.py` | `PAIRWISE_PLOT`, `PLOT` | Plot Q and P distributions and CV fields |

## Shared Inputs

Same as `next_hit`:

- **dataset**: `.pt`/`.npz` TensorQ dataset.
- **config**: YAML via `common.config`.
- **model**: `PairwiseCommittorNet` checkpoint.
- **model_input_space / cvs_to_use**: Passed to `common.data.select_model_inputs`.

Model evaluation uses `predict.infer_pairwise`, returning `Q` of shape `(n_frames, n_pairs)` where `n_pairs = n_states * (n_states - 1) / 2`.

## Model Architecture

`PairwiseCommittorNet` is defined in `model.py`:

- **Input**: `z` vector of dimension `in_dim`.
- **Hidden layers**: Same architecture as `NextHitCommittorNet`, default `(256, 256, 128)`.
- **Output**: `n_pairs` logits with sigmoid activation (default) or identity.
- **No normalization across pairs**: Each `q_ij` is independently in `[0, 1]`.
- **Lazy import**: The root `tensorq.__init__.py` exports `PairwiseCommittorNet` via `__getattr__`.

## Training Workflow

Code path: `train.py:main()` → `train_pairwise_committor(config)`.

Structurally parallel to `next_hit/train.py` but uses:
- `PairwiseLaggedDataset` instead of `LaggedCommittorDataset`.
- `GpuPairwiseLaggedBatcher` for GPU-resident data.
- `total_pairwise_committor_loss` from `losses.py`.

### Loss Components (losses.py)

- **Dirichlet loss**: Same structure as next-hit but applied to pairwise q values. Penalizes deviation from `q_ij(t) ≈ q_ij(t+tau)`.
- **Endpoint loss**: MSE between predicted `q_ij` and the known endpoint label (+1 for state j, 0 for state i) on labeled frames.

Note: There is no flux consistency loss in the pairwise formulation. This is a deliberate simplification — the pairwise q_ij do not satisfy a global normalization that makes flux profiles meaningful in the same way.

## Inference Workflow

Code path: `infer.py:main()` → `run_inference(config)`.

1. Load dataset and model.
2. `predict.infer_pairwise` runs batched forward passes, returns `Q` of shape `(n_frames, n_pairs)`.
3. `predict.reconstruct_state_probabilities` converts Q to P via:
   - Logit transform: undo sigmoid to get logits.
   - Solve linear system: `A * logits_P = logits_Q` where A is the encoding matrix mapping state logits to pair logits.
   - Softmax normalization with `anchor_state` fixing one degree of freedom.
   - Chunked computation for memory efficiency on large trajectories.
4. Compute `q_argmax` from P.
5. Save `Q.npy`, `P.npy`, `destination_argmax.npy`, optional CSV.

## State Probability Reconstruction

This is the critical bridge from pairwise to state probabilities (in `predict.py`):

- For `n_states`, there are `n_pairs = n_states * (n_states - 1) / 2` pairs.
- The relationship is: `logit(q_ij) = logit(P_j) - logit(P_i)` for unordered pair (i, j).
- Form matrix A of shape `(n_pairs, n_states)` where each row has -1 at column i and +1 at column j.
- Solve `A * x = b` via pseudo-inverse (or least squares), where `b` is the vector of pairwise logits.
- Apply softmax to x to get P, with anchor state pinned for numerical stability.

## Rate Estimation Workflow

Code path: `rate_constant.py:main()` → `run(config)`.

Simpler than `next_hit/rate_constant.py` — no error analysis, no slicing, no std propagation:

1. Load or infer P values.
2. Build lagged pairs.
3. `estimate_flux_profiles` computes J_ij(c) from P (numpy path only).
4. `estimate_pi` computes equilibrium populations.
5. `assemble_generator` builds K from flux.
6. `compute_mfpt_matrix` derives MFPTs.
7. Output: rate constant CSV, matrix heatmaps.

## Plotting Workflow

Code path: `plot.py:main()` → `run(config)`.

- **Q distributions**: Histograms of `q_ij` per pair.
- **P distributions**: Histograms of `P_i` per state.
- **CV fields**: 2D/3D binned averages of Q_ij and P_i over CV space.

## Relationship to next_hit

The two subpackages are independent and parallel. They share:

- The same `common` infrastructure (config, data, flux).
- The same dataset format (`.pt`/`.npz` with `meta_state` labels).
- The same `label.py` in `next_hit/` (which builds datasets for both formulations).

They differ in:

| Aspect | next_hit | pairwise |
|---|---|---|
| Output | `q_i(x)` with sum = 1 | `q_ij(x)` per pair, independent |
| Model output units | `n_states` | `n_pairs = n*(n-1)/2` |
| Normalization | Softmax / positive-L1 | Sigmoid (per pair) |
| Loss terms | Dirichlet + boundary + flux consistency | Dirichlet + endpoint MSE |
| Rate estimation | Full error analysis (slicing, std) | Basic (flux + MFPT) |
| SCF fitting | Yes (`fit_rate.py`) | No |

## Decision Meaning

- **`q_ij ≈ 0.5`**: Configuration is on the transition state between i and j.
- **`q_ij` near 0 or 1**: Configuration is deep in basin i or j for this pair.
- **Reconstructed P_i diffuse**: Multiple q_ij pairs give conflicting information — possible unlabeled state or poor features.
- **Reconstructed P_i sharp**: All pairwise committors are self-consistent and point to the same state.

## Important Constraints

- The encoding matrix A is rank `n_states - 1` (one degree of freedom from normalization). The anchor state pins this gauge freedom.
- For `n_states = 2`, the pairwise and next-hit formulations are equivalent (one pair, one degree of freedom).
- Reconstructed P may have small negative values due to numerical noise in the pseudo-inverse — these are clipped.
- No flux consistency loss means the pairwise model does not directly learn to produce smooth flux profiles.

## Key Config Knobs

**Model / Training**: Same as `next_hit` (hidden, activation, dropout, lr, epochs, patience, etc.).

**Pairwise-specific**:
- `n_pairs`: derived from `n_states`, not directly configured.
- `output_activation`: `"sigmoid"` (default) or `"identity"`.
- `anchor_state`: which state to pin during P reconstruction (default 0).
- `reconstruct_chunk`: chunk size for memory-efficient reconstruction.

**Rate estimation**:
- `lag`, `frame_time`, `time_unit`.
- `n_thresholds`, `flux_surface`, `flux_eps`.

## File Map

- `model.py`: `PairwiseCommittorNet` MLP.
- `train.py`: Training loop, GPU batcher, checkpointing, CLI.
- `predict.py`: Model loading, pairwise inference, state probability reconstruction.
- `infer.py`: Inference orchestration, CSV output.
- `losses.py`: Dirichlet loss, endpoint MSE loss, total loss.
- `rate_constant.py`: Flux profiles, populations, generator, MFPT, CLI.
- `plot.py`: Q/P distributions, CV field plots, CLI.

## Current Known Rough Edges

- `reconstruct_state_probabilities` is duplicated between `predict.py` (primary) and `infer.py` (wrapper). The `infer.py` version is a thin caller — confirm it does not duplicate logic.
- No flux consistency loss means the rate estimation step is decoupled from training. The model has no incentive to produce physically consistent flux profiles during training.
- No SCF fitting counterpart (`fit_rate.py` only exists in `next_hit/`).
- `rate_constant.py` is significantly simpler than its `next_hit` counterpart — no slice-based error analysis or std propagation. This may be intentional (pairwise rates are less mature) or an omission.

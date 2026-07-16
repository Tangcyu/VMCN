# TensorQ

TensorQ now uses one shared package architecture:

- `tensorq.common`: YAML, dataset, CV featurization, lagged indexing, and rate helpers.
- `tensorq.next_hit`: next-hit committor training, inference, plotting, and rate estimation.
- `tensorq.pairwise`: pair-wise committor training, inference, plotting, and rate estimation.
- `tensorq.gradpath`: direct-channel gradient pathway shooting, weighted clustering, and plotting.
- `tensorq.voronoi_merge`: Voronoi shared-segment alignment and iterative KLD diagnostics.
- `tensorq.MSMlabel`: MSM/PCCA+ macrostate discovery and TensorQ core-label dataset export.

The shared dataset is produced by `scripts/label.py` from the `TENSORQ_LABEL` config section. Both committor families consume that same `.pt` or `.npz` dataset.

## Staged workflow

`run.py` is the single dispatcher for the four-stage workflow. The default
`config.yaml` points to one focused configuration file per stage:

- `configs/0.MSMcorelabel.yaml`: weighted MSM, PCCA+, diagnostics, and core labels.
- `configs/1.Committorvector.yaml`: next-hit training, inference, plotting, and rate constants.
- `configs/2.Gradpath.yaml`: gradient pathfinding, weighted path clustering, plotting, and optional Voronoi merging.
- `configs/3.Relabel.yaml`: diagnostics plus entropy (`H`) and Gini (`G`) relabeling.

Run from the repository root:

```bash
python run.py --step 0       # MSM -> PCCA+ -> core-label dataset
python run.py --step 1       # next-hit committor vector
python run.py --step 2       # gradpath and Voronoi workflow
python run.py --step 3       # diagnose, H relabel, G relabel
```

`--step` also accepts the names `msmcorelabel`, `committorvector`, `gradpath`,
`relabel`, and `all`. Use a direct stage YAML when desired:

```bash
python run.py --step 1 --config configs/1.Committorvector.yaml
python run.py --step 1 --substep rate_constant
python run.py --step 2 --substep merging
python run.py --step 3 --substep gini
```

The sections below describe every dispatcher stage and substep. Substep names
are the values accepted by `--substep` and by `pipeline.substeps` in each stage
configuration.

### Step 0: MSM core labeling (`msmcorelabel`)

```bash
python run.py --step 0
```

This stage calls the same complete `all` pipeline as
`tensorq.MSMlabel.cli`. It converts weighted trajectory/frame tables into a
TensorQ core-label dataset.

1. `data` — loads tables, raw colvars, or discovered colvars trajectories;
   preserves trajectory boundaries; applies stride and weight handling; and
   writes the unified frame table and modeling coordinates.
2. `cluster` — normalizes the selected coordinates and performs weighted
   k-means to assign microstates.
3. `msm` — constructs weighted MSMs for every value in `msm.lags`, computes
   implied timescales, and writes MSM diagnostics. The origin-frame weights and
   zero-weight-origin masking are controlled by the `msm` section.
4. `pcca` — selects `pcca.selected_lag`, computes PCCA+ macrostates for
   `pcca.m_values` or `pcca.single_m`, and evaluates macrostate transition and
   CK statistics.
5. `core` — estimates density/free-energy cores inside each macrostate and
   writes TensorQ-compatible labels. Core frames receive `meta_state >= 0`;
   intermediate frames receive `meta_state = -1`.

The principal output is the dataset under
`project.out_dir/06_core_labels/lag_<lag>/m_<m>/`. The native MSM CLI also
supports an independent `structures` command for exporting structures from an
already-created core-label dataset:

```bash
PYTHONPATH=src python -m tensorq.MSMlabel.cli structures configs/0.MSMcorelabel.yaml
```

### Step 1: next-hit committor vector (`committorvector`)

```bash
python run.py --step 1
```

The next-hit model predicts a probability vector
`q(x) = (q_0(x), ..., q_{N-1}(x))`, where `q_j(x)` is the probability that a
trajectory started at configuration `x` next reaches state `j`. The four
substeps use the corresponding `NEXT_HIT_*` sections in
`configs/1.Committorvector.yaml`.

1. `train` — reads the core-label dataset, builds trajectory-safe lagged
   pairs, trains the neural committor with the configured Dirichlet, boundary,
   simplex, and flux losses, and saves the best checkpoint under
   `NEXT_HIT_COMMITTOR.out_dir`.
2. `infer` — loads the best checkpoint, evaluates `q_j(x)` for every dataset
   frame, checks probability normalization, and writes `Q.npy`, destination
   labels, reactive weights, and optional CSV assignments under
   `NEXT_HIT_INFER.out_dir`.
3. `plot` — generates committor distributions, CV projections, reaction-tube
   projections, and flux-profile plots under `NEXT_HIT_PLOT.out_dir`.
4. `rate_constant` — estimates flux profiles, populations, jump
   probabilities, rate constants, generator matrices, and MFPT matrices using
   the trained next-hit model. Outputs are written under
   `NEXT_HIT_RATE.out_dir`, including `rate_constants.csv`, `K.npy`,
   `P_jump.npy`, and `MFPT.npy`.

There is intentionally no `rate_fit` or `fit_rate` substep in the committor
vector workflow. Rate estimation ends with `rate_constant`.

Run one substep independently when restarting from an existing checkpoint:

```bash
python run.py --step 1 --substep train
python run.py --step 1 --substep infer
python run.py --step 1 --substep plot
python run.py --step 1 --substep rate_constant
```

### Step 2: gradient pathways (`gradpath`)

```bash
python run.py --step 2
```

This stage uses the trained committor vector and the reactive channel score
`q_i(x) q_j(x)` to identify and refine pathways. Its configuration is split
between `GRADPATH` and `VORONOI_MERGE`.

1. `pathfinding` — selects channel points using either a manually configured
   state pair `(state_i, state_j)` or automatic pairs from `P_jump`; performs
   gradient shooting with `step_size`, `max_steps`, and `target_q`; smooths
   and reparameterizes the resulting paths; and writes path files.
2. `clustering` — performs weighted agglomerative clustering of the generated
   paths using `cluster_distance_threshold`, then writes cluster labels,
   linkage data, weighted center paths, and medoid paths. The native gradpath
   runner performs this substep in the same call as `pathfinding`, so the
   dispatcher reports it as complete rather than running the expensive
   shooting calculation twice.
3. `plot` — optional plotting of saved pathways, dendrograms, selected channel
   points, and pathways colored by `q_i q_j`. It can be run separately with
   `python run.py --step 2 --substep plot`.
4. `merging` — optional Voronoi shared-segment alignment and iterative pathway
   expansion/KLD analysis. Set `VORONOI_MERGE.enabled: true` and provide
   current and previous sample files before enabling this substep. It is
   disabled in the example configuration by default.

For automatic multi-pair pathfinding, set `GRADPATH.automatic_pairs: true`
and provide `GRADPATH.p_jump` (normally the `P_jump.csv` written by
`rate_constant`). The dispatcher then processes every pair above
`prob_threshold`, subject to `max_pairs`.

### Step 3: label diagnostics and relabeling (`relabel`)

```bash
python run.py --step 3
```

This stage uses the same dataset and next-hit checkpoint to identify uncertain
or kinetically inconsistent labels. All lagged analyses respect trajectory
boundaries. The default substeps are:

1. `diagnose` — computes confidence, current uncertainty, lagged uncertainty,
   label consistency, and basin kinetic groups, then writes a compact
   `diagnostic_summary.yaml`. The default uncertainty measure is normalized
   committor entropy,
   `H_norm(x) = -sum_j q_j(x) log(q_j(x)) / log(N)`.
2. `relabel_entropy` — applies the production entropy-based relabeling path:
   entropy screening, weighted density-core selection, and kinetic merge/
   split checks. It writes a relabeled dataset, plots, and
   `relabel_summary.yaml` under `entropy.output_dir`.
3. `relabel_gini` — runs the same relabeling and kinetic checks while replacing
   the entropy score with normalized Gini impurity,
   `G_norm(x) = (1 - sum_j q_j(x)^2) / (1 - 1/N)`. Its results are written
   separately under `gini.output_dir`.

Run individual relabeling substeps with:

```bash
python run.py --step 3 --substep diagnose
python run.py --step 3 --substep relabel_entropy
python run.py --step 3 --substep relabel_gini
```

The entropy and Gini runs share the thresholds, density settings, and
lagged-kinetic settings in `RELABEL`, but write separate outputs so their
label assignments can be compared safely.

<!-- ## Old Commands

## For generating datasets:
```bash
python scripts/label.py --config configs/example.yaml
```
## For training:
```bash
python scripts/next_hit_train.py --config configs/example.yaml
```
## Plotting and building kinetic models:
```bash
python scripts/next_hit_infer.py --config configs/example.yaml
python scripts/next_hit_plot.py --config configs/example.yaml
python scripts/next_hit_rate.py --config configs/example.yaml
```

## Path finding from committor vector gradient
```bash
python scripts/gradpath.py --config configs/gradpath.example.yaml
python scripts/gradpath_plot.py --config configs/gradpath.example.yaml
python scripts/gradpath.mergy.py --config configs/voronoi_merge.example.yaml # Merging based on Voronoi expansion
```
## Old pairwise committor related
```bash
python scripts/pairwise_train.py --config configs/example.yaml
python scripts/pairwise_infer.py --config configs/example.yaml
python scripts/pairwise_plot.py --config configs/example.yaml
python scripts/pairwise_rate.py --config configs/example.yaml
```

## MSM/PCCA+ core labels
```bash
PYTHONPATH=src python -m tensorq.MSMlabel.cli all src/tensorq/MSMlabel/config.template.yaml
``` -->

<!-- For inference, plotting, and rate estimation, prefer `*_checkpoint.pt` models because they preserve the model input metadata. -->

# Variational Multistate Committor Network (VMCN)

Variational Multistate Committor Network (VMCN) uses one shared package
architecture:

- `tensorq.common`: YAML, dataset, CV featurization, lagged indexing, and rate helpers.
- `tensorq.next_hit`: next-hit committor training, inference, plotting, and rate estimation.
- `tensorq.pairwise`: pair-wise committor training, inference, plotting, and rate estimation.
- `tensorq.gradpath`: direct-channel gradient pathway shooting, weighted clustering, and plotting.
- `tensorq.voronoi_merge`: Voronoi shared-segment alignment and iterative KLD diagnostics.
- `tensorq.MSMlabel`: MSM/PCCA+ macrostate discovery and VMCN core-label dataset export.

The staged workflow creates its shared `.pt` or `.npz` dataset during MSM core
labeling (step 0). For the alternative trajectory-labeling workflow, the
maintained wrapper is `scripts/dataset_label.py` (installed as
`tensorq-label`). Both committor families consume the same dataset format.

## Environment and dependencies

VMCN requires Python 3.10 or newer; Python 3.12 is used for current
development checks. Create an isolated environment and install the project
from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Core dependencies are declared in `pyproject.toml`: NumPy, pandas, PyYAML,
SciPy, scikit-learn, Matplotlib, PyTorch, tqdm, and setuptools for the build.
Install optional features only when they are needed:

```bash
python -m pip install -e ".[label]"       # MDAnalysis: DCD feature extraction
python -m pip install -e ".[structures]"  # MDTraj: PDB/DCD structure export
python -m pip install -e ".[pcca]"        # deeptime: PCCA+ (otherwise spectral fallback)
python -m pip install -e ".[all,test]"    # all optional runtime features + pytest
```

The default configs request CUDA where acceleration is useful. PyTorch is
still required on CPU-only systems; the main training/inference runners fall
back to CPU when CUDA is unavailable. For an NVIDIA GPU, install the PyTorch
build matching the host driver/CUDA runtime before installing this project.
Use `device: cpu`, `knn_backend: sklearn`, and `knn_device: cpu` explicitly
when a reproducible CPU-only run is desired.

Run the verification suite with:

```bash
python -m pytest -q
```

## Staged workflow

`run.py` is the single dispatcher for the four-stage workflow. The default
`config.yaml` points to one focused configuration file per stage:

- `configs/0.MSMcorelabel.yaml`: weighted MSM, PCCA+, diagnostics, and core labels.
- `configs/1.Committorvector.yaml`: next-hit training, inference, plotting, and rate constants.
- `configs/2.Gradpath.yaml`: gradient pathfinding, weighted path clustering, plotting, and optional Voronoi merging.
- `configs/3.Relabel.yaml`: diagnostics plus entropy (`H`) and Gini (`G`) relabeling.

The committed stage YAML files are editable templates. Replace placeholder
inputs such as `/path/to/frame_weights.csv` and confirm all dataset/model paths
before starting a production run. Relative paths are resolved from the
directory where the command is launched.

Run from the repository root:

```bash
python run.py --step 0       # MSM -> PCCA+ -> core-label dataset
python run.py --step 1       # next-hit committor vector
python run.py --step 2       # gradpath and Voronoi workflow
python run.py --step 3       # diagnose, H relabel, G relabel
```

The MSM/core-label stage can also be restarted at a specific substep:

```bash
python run.py --step 0 --substep data
python run.py --step 0 --substep cluster
python run.py --step 0 --substep msm
python run.py --step 0 --substep pcca
python run.py --step 0 --substep core
python run.py --step 0 --substep structures
```

When a substep is selected, prerequisite artifacts are reused even if
`project.force: true`; `force` applies only to the requested substep. Missing
prerequisites are built automatically for `data` through `core`. The
independent `structures` exporter requires an existing core-label dataset and
a configured topology/trajectory; it does not rebuild the MSM stages.

`--step` also accepts the names `msmcorelabel`, `committorvector`, `gradpath`,
`relabel`, and `all`. Use a direct stage YAML when desired:

```bash
python run.py --step 1 --config configs/1.Committorvector.yaml
python run.py --step 1 --substep rate_constant
python run.py --step 2 --substep merging
python run.py --step 3 --substep gini
```

The sections below describe every dispatcher stage and substep. All names are
accepted by `--substep`; stages 1–3 also read their default ordered lists from
`pipeline.substeps` in the corresponding stage configuration.

### Step 0: MSM core labeling (`msmcorelabel`)

```bash
python run.py --step 0
```

This stage calls the same complete `all` pipeline as
`tensorq.MSMlabel.cli`. It converts weighted trajectory/frame tables into a
VMCN core-label dataset.

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
   writes VMCN-compatible labels. Core frames receive `meta_state >= 0`;
   intermediate frames receive `meta_state = -1`.

The principal output is the dataset under
`project.out_dir/06_core_labels/lag_<lag>/m_<m>/`. The native MSM CLI also
supports an independent `structures` command for exporting structures from an
already-created core-label dataset:

```bash
PYTHONPATH=src python -m tensorq.MSMlabel.cli structures configs/0.MSMcorelabel.yaml
```

Configure `core_structures` before using that command. With `aligned_dcd`,
dataset row `i` must correspond exactly to DCD frame `i`; the exporter writes
the minimum-`dist_to_centroid` frame for each state as a PDB and can optionally
write every labeled state to a separate DCD with `write_state_dcds: true`.
Without `aligned_dcd`, it matches dataset CVs against the configured
DCD/colvars pairs within `tolerance`. Both modes require the `structures`
optional dependency.

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

   For shooting trajectories started outside equilibrium, set
   `NEXT_HIT_RATE.discard_first_n_frames: N`. The first `N` original saved
   dataset frames of every trajectory are excluded from population, flux,
   transition, rate, and MFPT estimates. The applied frame/pair counts are
   recorded under `trajectory_burn_in` in the rate `summary.yaml`.

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
   shooting calculation twice. Requesting `clustering` by itself invokes that
   bundled runner and therefore performs shooting as well.
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
`prob_threshold`, subject to `max_pairs`. In this mode the same threshold also
removes pathway clusters with fewer than
`ceil(prob_threshold * max_points)` generated paths; discarded paths receive
cluster label 0 and the details are saved in the gradpath summary.

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

As an optional final operation in either relabeling run,
`RELABEL.relabel.rate_merge_enabled: true` merges connected macrostates using
the `P_jump.npy` and `MFPT.npy` written by step 1. A directed edge must satisfy
both `P_ij > rate_merge_probability_cutoff` and
`MFPT_ij < rate_merge_mfpt_cutoff_frames`; bidirectional qualification can be
required. The groups, qualifying edges, and compacted label map are recorded
in `relabel_summary.yaml`.

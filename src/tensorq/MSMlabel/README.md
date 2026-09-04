# MSM to Committor-Vector State Labels

This project turns short unbiased shooting trajectories into candidate metastable state labels for downstream committor-vector training.

The workflow is checkpointed by stage:

1. Load old `riteweight.py` tables or raw `.colvars.traj` files while preserving `traj_id`.
2. Normalize the selected modeling coordinates and cluster them into many microstates with weighted k-means on CUDA when available.
3. Build weighted MSMs across lag times and plot implied timescales.
4. For each requested macrostate count `m`, run PCCA+ if `deeptime` is installed, otherwise use a spectral clustering fallback.
5. Compute macrostate transition matrices, residence and escape times, CK validation, and macrostate CV plots.
6. Estimate weighted kNN free-energy cores inside each PCCA macrostate and export TensorQ-style core-label datasets.
7. Optionally export PDB structures for the existing core-label dataset without rerunning earlier stages.

## Quick Start

```bash
cd /path/to/VMCN.v.0.1
python -m pip install -e .
tensorq-msmlabel all /path/to/msm_config.yaml
```

For a staged run:

```bash
PYTHONPATH=src python -m tensorq.MSMlabel.cli data config.template.yaml
PYTHONPATH=src python -m tensorq.MSMlabel.cli cluster config.template.yaml
PYTHONPATH=src python -m tensorq.MSMlabel.cli msm config.template.yaml
PYTHONPATH=src python -m tensorq.MSMlabel.cli pcca config.template.yaml
PYTHONPATH=src python -m tensorq.MSMlabel.cli core config.template.yaml
PYTHONPATH=src python -m tensorq.MSMlabel.cli structures config.template.yaml
```

Set `project.force: true` to recompute checkpoints after changing parameters. Otherwise existing outputs are reused.

## CVs vs High-Dimensional Features

`data.cvs` are always read from the frame table and kept for weights, inspection, and 2D projection plots. The coordinates used to build the MSM are controlled separately:

```yaml
data:
  cvs: ["ACH_1", "ACH_2", "coplanar_1", "Cloop_A", "Cloop_D", "theta"]
  model:
    source: features
    path: /path/to/features_internal_zmat.npz
    key: X
    dimensions: all
```

Use `data.model.source: cvs` for the old CV-space workflow. Use `source: features` for a RiteWeight feature cache such as `features_internal_zmat.npz`; `.npz` files are read from key `X` by default, and CSV caches with a `META=...` first line are also supported. The feature matrix must have the same row order and row count as the prepared frame table. `dimensions` can be `all`, integer feature indices, or names from `feature_columns`.

TensorQ `label.py` datasets can be used directly when saved as `.pt`:

```yaml
data:
  cvs: ["phi1", "phi2", "phi3", "sphi1", "cphi1", "sphi2", "cphi2", "sphi3", "cphi3"]
  tables:
    - /path/to/dataset.pt
  model:
    source: features
    path: /path/to/dataset.pt
    key: features
```

For `.pt` tables, `cv`, `weights`, `traj_id`, and `meta.cv_headers` are read from the dataset. The feature matrix is read from the `features` tensor. If the dataset was created with `save_cv: false`, provide a separate table containing the CV columns used for plotting.

Clustering uses `clustering.use_weights: true` by default, so `data.weight_column` affects k-means center updates. Zero-weight frames still receive microstate labels for projection and target-frame use, but they do not pull cluster centers.

## Important Outputs

- `01_data/frame_table.csv.gz`: unified frame table with `traj_id`, `frame_in_traj`, CVs, and weights.
- `01_data/features.npz`: selected modeling coordinates used for clustering/MSM construction.
- `02_microstates/microstates.npz`: microstate label for each frame and normalized cluster centers.
- `03_msms/lag_*/msm.npz`: weighted count matrix, transition matrix, stationary distribution, implied timescales.
- `05_plots/implied_timescales.png`: lag-time diagnostic for choosing the slow-process count.
- `05_plots/spectrum/`: eigenvalue/eigengap diagnostics for selecting candidate `m`.
- `04_pcca/lag_*/m_*/pcca.npz`: memberships, macro transition matrix, CK arrays, residence/escape times.
- `06_core_labels/lag_*/m_*/dataset.npz`: TensorQ-compatible dataset with `meta_state=-1` for intermediate frames.
- `06_core_labels/lag_*/m_*/weights_and_labels.csv`: inspectable CV/weight/core-label table.
- `06_core_labels/lag_*/m_*/macrostate_fes_centers.csv`: lowest-FES center selected for each macrostate.
- `07_core_structures/`: optional PDB files for frames assigned to each core state, plus match manifests.
- `summary.csv`: compact table to compare `m` values.

## Choosing `m`

Start from `05_plots/implied_timescales.png` and `implied_timescales.csv` to identify candidate slow processes. Then inspect:

- `05_plots/spectrum/selected_lag_*_spectrum.png` for the eigenvalue spectrum, timescale spectrum, and eigengap at the selected lag.
- `05_plots/spectrum/eigengap_candidate_m_vs_lag.png` and `candidate_m_by_lag.csv` for candidate `m = slow_processes + 1` across lags.
- `summary.csv` for self-transition probabilities, CK RMSD, and residence/escape times.
- `05_plots/macrostates/*.png` for physically interpretable CV-space regions.
- `05_plots/ck/*_ck_self_transitions.png` for observed-vs-predicted macrostate residence curves.
- `05_plots/ck/*_ck_observed_vs_predicted.png` for all macrostate transition probabilities against the ideal `y=x` line.
- `04_pcca/lag_*/m_*/pcca.npz` memberships for robust, non-fragmented assignments.

Pick the smallest `m` that gives clear metastability, robust memberships, reasonable CK validation, and interpretable regions.

The spectral-gap suggestion is a shortlist, not a final answer. If the largest eigengap suggests `m=4`, still check that the `m=4` CK and CV-space macrostates are physically meaningful.

## Reading CK Plots

For a good state assignment, the observed CK curves should track the predicted curves from `P(tau)^k`. In `*_ck_self_transitions.png`, compare each solid line to the dashed line of the same color. Large systematic gaps mean the chosen macrostates are not Markovian at that lag or that `m`.

In `*_ck_observed_vs_predicted.png`, points close to the diagonal are good. Broad scatter away from the diagonal means the macrostate transition matrix is not predictive at longer lags.

## Input Notes

Old `riteweight.py` output can be used through `data.tables`. If those files contain `traj_id` and `frame_in_traj`, trajectories stay separated. If they do not, each input table is treated as one trajectory.

For short shooting trajectories, prefer `data.colvars` or `data.folders`; each file becomes its own trajectory automatically.

For concatenated old tables without `traj_id`, enable `data.infer_trajectories`. The default `step_reset` mode splits a table whenever `step` decreases, so MSM lag pairs never cross trajectory boundaries. You can also set `frames_per_traj` for fixed-length blocks.

Set `msm.use_weights: true` to count transitions with origin-frame weights from `data.weight_column`. This is enabled by default. Set `msm.mask_zero_weight_origins: true` to allow `weight == 0` frames as `z(t+tau)` targets while excluding them as `z(t)` origins; this is also enabled by default. If zero-weight frames exist, reversible count symmetrization is disabled so masked target regions are not turned into artificial origins.

Set `pcca.exclude_zero_weight_microstates: true` to remove microstates supported only by `weight == 0` frames before selecting PCCA macrostates. Those inactive microstates receive `macro_by_micro = -1` and are skipped in macrostate CK tests.

Set `plotting.cv_pairs` to choose specific macrostate projections, or omit it to plot every unique pair from `data.cvs`. These plots remain in CV space even when `data.model.source: features` builds the MSM in high-dimensional coordinates.

## Core Labels

The PCCA assignment is used only as a candidate macrostate partition. The exported training labels are stricter: inside each macrostate, the code estimates a weighted k-nearest-neighbor density using positive-weight frames only; the lowest free-energy point is saved as the macrostate center, and only the densest `core_labeling.core_fraction` positive-weight frames are considered for labeling. Every other frame, including every `weight == 0` frame, is assigned `meta_state = -1` as intermediate.

By default `core_labeling.feature_source: model`, so high-dimensional `data.model` features are also used for core density and distances. Set `core_labeling.feature_source: cvs` to force core labeling back onto `core_labeling.feature_cvs`, while keeping the MSM built in another space. The saved TensorQ `features` tensor uses the same coordinates by default; set `core_labeling.output_features.source: features` to label in CV space but save the aligned full feature cache for training.

Set `core_labeling.selection_mode: density_connected` to label only the connected low-FES pocket around the lowest-FES center. Set `core_labeling.connected_core.split_disconnected_pockets: true` to split disconnected low-FES pockets into separate core labels instead of merging them into one state. Set `pcca.single_m` to test one macrostate count without editing `pcca.m_values`.

This follows the TensorQ next-hit dataset convention: `features`, `weights`, `meta_state`, `dist_to_centroid`, `thresholds`, `cv`, `traj_id`, and metadata are saved in the dataset file.

## Core Structures

Set `core_structures.enabled: true` and run `PYTHONPATH=src python -m tensorq.MSMlabel.cli structures config.yaml` from the Tensorq repository root to export structures from an existing `dataset.pt` or `dataset.npz`. This stage does not rerun data preparation, clustering, MSM, PCCA, or core labeling.

There are two export modes:

- Set `core_structures.aligned_dcd` when dataset row `i` is exactly DCD frame
  `i`. The exporter writes one `core_state_NNN.pdb` at the minimum
  `dist_to_centroid` frame per state. Set `write_state_dcds: true` to stream all
  frames with `meta_state >= 0` into one DCD per state.
- Otherwise, configure `folders`, `match_dcd`, `match_colvars`, `tag_regex`,
  `stride`, and `allow_skip_first_colvars`. The exporter matches labeled
  dataset CVs to colvars rows within `core_structures.tolerance` over
  `core_structures.match_cvs`, then writes the center PDB for each state.

`frames.csv`, `summary.csv`, and `counts.csv` record the source-frame mapping;
aligned state-DCD export also writes `state_dcds.csv`. MDTraj is required only
for this export step and is available through `pip install -e ".[structures]"`.

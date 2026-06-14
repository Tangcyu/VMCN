# TensorQ Relabeling Scheme

`tensorq.relabel` has two public workflows:

- `scripts/relabel_diagnose.py`: run label diagnostics and write
  `diagnostic_summary.yaml`.
- `scripts/relabel.py`: apply relabeling and write `relabel_summary.yaml`;
  it may also write a relabeled dataset for the next train -> relabel round.

Both workflows read the `RELABEL` or `TENSORQ_RELABEL` section from a YAML
config, load a TensorQ dataset plus a trained committor-vector checkpoint,
infer `q_values`, and use trajectory-safe lag pairs for lagged entropy or
kinetic analysis.

## Diagnostics

Diagnostics are implemented in `label_diagnostics.py` and serialized by
`diagnostics_io.py`.

The diagnostic summary contains:

- per-state committor consistency summaries
- relabel hints from entropy and label consistency
- optional lagged entropy classification
- optional basin-internal kinetic group summaries

The only diagnostic file written is:

- `diagnostic_summary.yaml`

Old CSV/JSON diagnostic files are removed from the output directory when a new
diagnostic run writes its YAML summary.

## Relabeling

Relabeling is implemented in `relabel.py`. The proposal returned by
`propose_relabeling` is intentionally compact:

- `proposed_labels`
- `changed_mask`
- `masks`
- `tables`
- `diagnostics`
- `scores`

Automatic relabeling follows three stages.

### 1. Entropy

Current normalized committor entropy `H(q(x)) / log(N)` marks unreliable frames.
Before per-frame pruning, each current label is checked as a whole. If most
frames in a label have low label consistency `q_label(x)` or high entropy, or
if too little of the label remains as stable high-confidence core, the entire
label is removed and its frames are set to `-1`.

After label-level pruning, high-entropy labeled frames are set to `-1` before
retraining so they no longer act as hard boundary labels.

Trajectory-safe lagged entropy then classifies those uncertain frames:

- high current entropy and high lagged entropy -> persistent uncertain region
- high current entropy and low lagged entropy -> transition-like review region
- insufficient lag statistics -> unresolved review region

Only persistent uncertain frames are eligible for automatic new-core detection.

### 2. Density

Persistent high-entropy candidates are sampled for the graph if needed, grouped
with kNN connected components, and trimmed to a weighted local-density core.
Density shells are returned to `-1`.

This is the only automatic new-label source.

### 3. Kinetics

Kinetic checks are applied after the entropy/density proposal:

- existing labels can be reshaped to high-confidence q-cores
- basin-internal kinetic groups can split a label when a supported slow mode is
  present
- split groups with less than `min_split_core_weight_fraction` of the parent
  state's weight are assigned to `-1` instead of becoming a new state
- final lagged checks can merge labels that behave as one metastate
- final kNN kinetic checks can split labels with weakly exchanging components

The basin-internal slow-mode check can be expensive on long trajectories. It
caches standardized features and same-state lag-pair positions, and
`basin_kinetic_groups.max_transition_pairs_per_state_lag` can cap the number of
transition pairs used per state and lag. A value of `0` keeps exact transition
counts; a positive value uses a reproducible sampled estimate.

## Outputs

Relabeling writes:

- `relabel_summary.yaml`
- optional relabeled dataset when `relabel.write_relabel_dataset: true`
- optional diagnostic plots when `relabel.make_plots: true`

Relabel CSV frame dumps are no longer written. Existing generated
`relabel_*.csv` files in the output directory are removed by `scripts/relabel.py`
to avoid stale diagnostics.

`relabel_summary.yaml` includes `removed_states`, so label-level inconsistency
deletions remain auditable without writing separate CSV files.

The relabel diagnostic plots are focused on the current pipeline:

- `relabel_entropy.png` with current entropy and time-lagged entropy panels
- `relabel_candidates.png`
- `relabel_labels_before_after.png`

## Config Sections

- `analysis`: shared thresholds for q consistency, entropy, lag list,
  persistence, eigengap, and group size.
- `relabel`: relabel-specific graph, density, kinetic-check, and dataset-output
  settings.
- `confidence`, `kinetics`, `uncertainty`, and `basin_kinetic_groups`: narrow
  legacy fallback sections still read by diagnostics and shared settings.

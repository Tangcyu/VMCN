# TensorQ Relabeling Scheme

`tensorq.relabel` has two public workflows:

- `scripts/relabel_diagnose.py`: run diagnostics and write confidence/kinetic
  summaries.
- `scripts/relabel.py`: apply the relabeling proposal and optionally write a
  relabeled dataset for the next train -> relabel round.

Both workflows read the `RELABEL` section from a YAML config, load a TensorQ
dataset plus trained committor-vector checkpoint, infer `q_values`, and use
trajectory-safe lag pairs when kinetic analysis is requested.

## Diagnostics

Diagnostics are implemented in `label_diagnostics.py`.

Per-frame outputs include:

- `q_argmax`, `q_max`, normalized entropy, and committor confidence.
- `label_consistency = q_{state_label}(x)` for currently labeled frames.
- optional lagged entropy and lagged q-max fields.
- `basin_kinetic_group`, when basin-internal kinetic grouping is enabled.

State-level outputs are written to:

- `state_confidence_summary.csv`
- `relabel_hints.csv`
- `basin_kinetic_state_summary.csv`
- `basin_kinetic_groups.csv`
- `diagnostic_summary.json`

The legacy split/merge/missing-state detectors remain disabled. The supported
automatic kinetic check is narrower: inside each label, collect high-confidence
q-core frames, cluster them into feature-space microstates, build a local
trajectory-safe transition matrix, and use the local MSM eigenvalue spectrum to
estimate how many metastable groups are present. A clear slow-mode eigengap
under one label suggests that the label may contain several kinetic metastates.

## Relabeling

Relabeling is implemented in `relabel.py` and exposed by `scripts/relabel.py`.
The old conservative/radical split has been removed; the former radical
confidence/kNN relabeler is now the single relabel path.

The default relabeler follows the same logic as diagnostics:

1. Removes whole labels only when most frames are unreliable.
2. Marks high-entropy / low-commitment frames as unlabeled.
3. Uses lagged behavior to separate transition-like review frames from
   persistent uncertain review frames.
4. Reshapes existing labels to high-confidence q cores.
5. Splits an existing label only when the local spectral MSM finds a supported
   slow-mode split.

Persistent high-entropy regions are review candidates by default, not automatic
new labels. The older kNN promotion path is still available for advanced runs
with `relabel.promote_persistent_candidates: true`, but it is off in the
recommended minimal config.

Main relabel outputs are written with the `relabel_` prefix:

- `relabel_summary.yaml`
- `relabel_changed_frames.csv`
- `relabel_removed_states.csv`
- `relabel_basin_kinetic_state_summary.csv`
- `relabel_reshaped_basin_groups.csv`
- `relabel_reshaped_basin_core_frames.csv`
- `relabel_reshaped_basin_shell_frames.csv`
- `relabel_transition_like_candidate_frames.csv`
- `relabel_persistent_candidate_review_frames.csv`

## Key Config Sections

- `analysis`: the main shared decision knobs for both diagnose and relabel:
  lag list, q/entropy/core cutoffs, minimum count, persistence fraction,
  eigengap, maximum group count, and minimum group size.
- `relabel`: dataset writing and whether to reshape/split existing labels.

Legacy `confidence`, `kinetics`, `uncertainty`, and `basin_kinetic_groups`
sections remain supported as fallbacks for older configs, but new configs should
prefer `analysis`.

# TensorQ Relabeling Scheme

This note is the current implementation frame for `tensorq.relabel`. It is meant
for agents or developers who need to understand the relabel workflow without
reading every source file.

## Current Philosophy

The relabel code is confidence-first. The committor-vector model predicts
`q_i(x)`, the probability that configuration `x` reaches state `i` before the
other labeled states. A good label `y(x) = i` should have high own-state
committor `q_i(x)` and usually low normalized entropy.

The current code does not trust automatic split, merge, or missing-state calls
from clustering. Those old candidate detectors are disabled because they were
too sensitive to feature choice, CV projection, lag time, and clustering
hyperparameters. The diagnostic stage now answers a narrower question:

> Where do the current labels disagree with the learned committor?

Actual split, merge, and new-state decisions still require CV/structure
inspection and ideally retraining.

## Entry Points

The wrappers currently present on disk are:

- `scripts/relabel_diagnose.py`: run diagnostics only.
- `scripts/relabe_conservel.py`: run conservative relabel application. The file
  name currently has a typo, but it imports `tensorq.relabel.apply`.
- `scripts/relabel_radical.py`: run radical relabel application.

All read the `RELABEL` section from a YAML config. The active Trialanine config
is `2.Trialanine/1.Committor_Vector/relabel.yaml`; the package example is
`configs/relabel.example.yaml`.

## Shared Inputs

Each workflow loads:

- `dataset`: a `.pt` or `.npz` TensorQ dataset with model features, optional CVs,
  trajectory ids, weights, and current `meta_state` labels.
- `model`: a trained committor-vector checkpoint.
- `model_input_space` / `cvs_to_use`: passed to `select_model_inputs`.
- `dataset_stride`: optional analysis stride. If an output dataset is written,
  it contains only the strided frames.

The model is evaluated in batches by `next_hit.predict.infer_probabilities`.
The resulting `q_values` array has shape `(n_frames, n_states)`.

## Per-Frame Metrics

Implemented in `label_diagnostics.py` and reused by `apply.py` / `radical.py`.

For each frame:

- `q_argmax = argmax_i q_i(x)`
- `q_max = max_i q_i(x)`
- `q_entropy = -sum_i q_i(x) log q_i(x)`
- `q_entropy_norm = q_entropy / log(n_states)`
- `committor_confidence = 1 - q_entropy_norm`
- `label_consistency = q_{state_label}(x)` for labeled frames
- `core_label_consistency = q_{core_state_label}(x)`

Important interpretation:

- Low `label_consistency` means the current label is not supported by the
  learned committor.
- High `q_entropy_norm` means the committor prediction is diffuse.
- High entropy alone is not proof of a new state. It can also mean transition
  region, bad descriptor space, insufficient training, or boundary
  contamination.

## Diagnostics Workflow

Code path:

- `scripts/relabel_diagnose.py`
- `tensorq.relabel.main.main`
- `tensorq.relabel.label_diagnostics.run_relabel`
- `StateLabelDiagnostics.run_all`
- `diagnostics_io.save_results`

State-level summaries include:

- `n_frames`
- `n_core_frames`
- `mean_q_own`
- `median_q_own`
- `fraction_low_consistency`
- `mean_entropy_norm`
- `median_entropy_norm`
- optional short-lag kinetic metrics if `diagnostics.compute_kinetics: true`

The diagnostics build `relabel_hints` with this logic:

- If a state is unreliable and low-consistency frames mostly point by
  `q_argmax` to one alternative state, flag
  `possible_reassignment_or_merge`.
- If a state is unreliable and high entropy, flag
  `ambiguous_or_missing_coordinate`.
- If a state is unreliable without a dominant destination, flag
  `possible_mislabel_or_broad_state`.
- If a state is mainly high entropy, flag
  `transition_like_or_low_confidence`.
- Otherwise flag `no_confidence_issue`.

The old methods `detect_split_candidates`, `detect_merge_candidates`, and
`detect_missing_state_candidates` intentionally return empty lists.

Main diagnostic outputs:

- `diagnostic_summary.json`
- `diagnose_summary.yaml`
- `state_confidence_summary.csv`
- `relabel_hints.csv`
- optional `label_consistency_by_frame.csv`
- optional `label_consistency_by_frame.npz`
- optional `Q.npy`
- diagnostic plots from `plot.py`

`diagnostics_io.save_results` removes stale old clustering outputs:

- `split_candidates.csv`
- `split_candidates_details.json`
- `merge_candidates.csv`
- `missing_state_candidates.csv`
- `cluster_statistics.csv`

Performance note: per-frame CSV can dominate runtime on large trajectories.
The current config defaults to `save_per_frame_csv: false` and
`save_per_frame_npz: false`.

## Conservative Relabel Workflow

Code path:

- `scripts/relabe_conservel.py`
- `tensorq.relabel.apply.main`
- `run_apply_relabel`
- `propose_relabeling`

This mode applies only conservative, confidence-supported changes.

For each current state `i`:

1. Find frames with `label_consistency < confidence.q_label_cutoff`.
2. Mark the state as unreliable if:
   - `fraction_low_consistency >= confidence.state_fraction_low_cutoff`, or
   - `mean_q_own < confidence.state_mean_q_cutoff`.
3. Among low-consistency frames, find the dominant non-self `q_argmax` state
   `j`.
4. Reassign only if:
   - state `i` is unreliable,
   - `j` exists,
   - the fraction of bad frames pointing to `j` is at least
     `confidence.reassign_dominance_cutoff`, and
   - those frames have `q_j >= relabel.reassign_q_cutoff`.

Mixed-destination low-consistency frames are not split automatically. They are
written for review as possible broad/mixed labels.

High-entropy weak-destination frames are computed as:

```text
q_entropy_norm >= confidence.entropy_cutoff_ambiguous
and q_max < relabel.missing_qmax_cutoff
```

If `relabel.mark_ambiguous_as_unlabeled: true`, those frames are assigned `-1`.
Otherwise they remain unchanged and are written as review signals.

Conservative outputs:

- relabeled dataset at `relabel.output_dataset`
- `relabel_apply_summary.yaml`
- `relabel_actions.csv`
- `relabel_changed_frames.csv`
- `relabel_review_mixed_frames.csv`
- `relabel_review_missing_signal_frames.csv`
- `relabeled_state_labels_2d/3d.<format>`
- `relabel_changed_review_2d/3d.<format>`

Plot note: `relabel.plot_unlabeled: false` hides `-1` unlabeled frames from the
state-label CV plot.

## Radical Relabel Workflow

Code path:

- `scripts/relabel_radical.py`
- `tensorq.relabel.radical.main`
- `run_radical_relabel`
- `propose_radical_relabeling`

This mode is intentionally more aggressive. It is for cases where whole labels
are badly contaminated and should be removed before looking for new cores.

### Step 1: Remove Bad States

For each state `i`, compute frames that are problematic:

```text
label_consistency < confidence.q_label_cutoff
or q_entropy_norm >= confidence.entropy_cutoff_ambiguous
```

If the problematic fraction is at least
`radical.remove_problem_fraction_cutoff`, the entire state is removed by setting
all its frames to `-1`.

Removed state ids are stored and ignored in the later surviving-state top-2
committor calculation.

### Step 2: Mark Ambiguous Frames Unlabeled

Candidate uncertainty is configured by:

```text
uncertain = q_entropy_norm >= radical.candidate_entropy_cutoff
weak_destination = q_max <= radical.candidate_qmax_cutoff
```

With `radical.candidate_logic: "and"`, both must be true. With `"or"`, either
is enough.

All candidate frames that still have a non-negative label are first set to `-1`.
This prevents old labels from surviving into the graph step.

### Step 3: Surviving-State Top-2 Selection

The graph pool is not based on the old global `q_argmax` if a state was removed.
Instead, `_surviving_top2` masks removed committor dimensions and finds the top
two states only among surviving labels.

A frame enters the pair-ambiguous graph pool only if:

```text
new_state == -1
and candidate uncertainty condition is true
and surviving top1/top2 states exist
and surviving top2 q >= radical.top2_min_probability
and surviving top1 q - surviving top2 q <= radical.top2_margin_cutoff
```

This implements the current rule: run kNN graphs only when the frame is
ambiguous between two surviving states, not merely because the old prediction
favored a state that has just been removed.

### Step 4: Pairwise kNN Components

For each unordered surviving-state pair `(a, b)`, build a kNN graph only on
candidate frames whose surviving top-2 pair is `(a, b)`.

Feature space comes from `radical.graph_space`:

- `"cv"`: use saved CV columns.
- `"features"` / `"model_features"`: use the model input features.

Features are standardized columnwise before graph construction. Large candidate
pools are sampled with probability proportional to:

```text
entropy_norm * (1 - q_max) * weight
```

Each connected component is promoted to a new state only if:

- `n_frames >= radical.min_new_core_size`, and
- weighted population is at least `radical.min_new_core_weight`.

Accepted components are candidate pools for new core labels. Unaccepted
pair-ambiguous frames remain `-1` and are written as review signals.

### Step 5: Iterative New-State Mixing Check

The kNN graph can split a single metastable basin into multiple geometric
components. After provisional new labels are assigned, radical relabeling checks
whether those new labels are dynamically distinct.

For each pair of provisional new labels `(i, j)`, the code computes
trajectory-safe label-indicator time correlations at configured lags:

```text
C_ij(tau) = <1[label_new(t) = i] 1[label_new(t+tau) = j]> / <1[label_new(t) = i]>
C_ji(tau) = <1[label_new(t) = j] 1[label_new(t+tau) = i]> / <1[label_new(t) = j]>
```

These are transition probabilities estimated only from lagged pairs within the
same trajectory. If the configured mixing rule is met, the two provisional new
labels are merged into the same new state. The scan is repeated until no pair
exceeds the mixing threshold or `radical.merge_max_iterations` is reached. This
reduces over-splitting when high-uncertainty conformations are connected by fast
exchange and therefore do not support separate metastable states.

Implementation note: the iteration computes one transition matrix per lag and
reuses it for all label pairs in that iteration.

### Step 6: Weighted Dense-Core Selection

A full final metastable basin can include transition-shell frames around the
actual core. After the iterative mixing check, radical relabeling estimates a
weighted local density inside each final new-state basin:

```text
density(x) = sum_{y in kNN(x)} weight(y) / radius_k(x)^p
```

where `radius_k(x)` is the distance to the configured kth neighbor in the same
standardized graph space and `p = radical.density_radius_power`. Only the
highest-density subset, controlled by `radical.density_core_fraction`, receives
the final new label. Lower-density frames from the same final basin stay `-1`
and are written to `radical_density_shell_frames.csv`.

Radical outputs:

- relabeled dataset at `radical.output_dataset`
- `radical_relabel_summary.yaml`
- `radical_removed_states.csv`
- `radical_removed_frames.csv`
- `radical_ambiguous_unlabeled_frames.csv`
- `radical_new_core_components.csv`
- `radical_merged_new_states.csv`
- `radical_new_core_frames.csv`
- `radical_density_shell_frames.csv`
- `radical_changed_frames.csv`
- `radical_pair_ambiguous_review_frames.csv`
- relabel CV plots via `apply.plot_relabel_cv`

The stale old file `radical_far_uncertain_review_frames.csv` is removed if it
exists.

When `radical.compact_labels: true` (default), the saved relabeled dataset
remaps all non-negative labels to contiguous ids `0..N-1` and keeps `-1`
unlabeled frames unchanged. The old-to-new label map is written under
`meta.relabel.label_mapping`.

## Decision Meaning

Use these rules when interpreting results:

- Low consistency plus one dominant alternative state means inspect/reassign
  those frames first. Only consider merge if retraining still cannot separate
  the two states.
- Low consistency plus multiple destinations means possible broad or mixed
  label. A split is plausible only after CV/structure inspection shows stable
  reproducible subregions.
- Persistent high entropy without a strong existing-state destination is the
  missing-state signal. Add a state only after those frames form a stable,
  reproducible region after inspection and retraining.
- Radical new cores are hypotheses, not final physical states. They should be
  checked in CV/structure space and then retrained.

## Important Constraints

- Lagged pairs must never connect different trajectories.
- Do not assume global frames are sorted by trajectory.
- Handle insufficient statistics by returning empty rows or `NaN`, not by
  raising avoidable errors.
- Do not write huge per-frame CSVs by default.
- Do not reintroduce automatic split/merge/missing-state clustering into
  diagnostics unless it is explicitly optional and clearly labeled exploratory.
- Keep unlabeled frames as `-1`; downstream plots should hide them unless
  explicitly configured otherwise.

## Key Config Knobs

Confidence:

- `confidence.q_label_cutoff`
- `confidence.entropy_cutoff_ambiguous`
- `confidence.state_fraction_low_cutoff`
- `confidence.state_mean_q_cutoff`
- `confidence.reassign_dominance_cutoff`

Diagnostics:

- `diagnostics.compute_kinetics`
- `diagnostics.profile_timing`
- `output.save_per_frame_csv`
- `output.save_per_frame_npz`
- `output.save_q_values`

Conservative relabel:

- `relabel.write_relabel_dataset`
- `relabel.output_dataset`
- `relabel.reassign_q_cutoff`
- `relabel.missing_qmax_cutoff`
- `relabel.mark_ambiguous_as_unlabeled`
- `relabel.plot_unlabeled`

Radical relabel:

- `radical.compact_labels`
- `radical.remove_problem_fraction_cutoff`
- `radical.candidate_entropy_cutoff`
- `radical.candidate_qmax_cutoff`
- `radical.candidate_logic`
- `radical.top2_min_probability`
- `radical.top2_margin_cutoff`
- `radical.graph_space`
- `radical.k_neighbors`
- `radical.max_graph_frames`
- `radical.min_new_core_size`
- `radical.min_new_core_weight`
- `radical.density_core_enabled`
- `radical.density_core_fraction`
- `radical.density_k_neighbors`
- `radical.density_radius_power`
- `radical.density_core_min_size`
- `radical.density_core_min_weight`
- `radical.density_core_max_size`
- `radical.merge_mixed_new_states`
- `radical.merge_lag_list`
- `radical.merge_transition_cutoff`
- `radical.merge_require_bidirectional`
- `radical.merge_min_valid_pairs`
- `radical.merge_max_iterations`

## File Map

- `THEORY.md`: conceptual background and validation workflow.
- `label_diagnostics.py`: confidence metrics, state summaries, relabel hints,
  optional kinetic summaries.
- `diagnostics_io.py`: diagnostic output writer and stale-output cleanup.
- `lag_pair_utils.py`: trajectory-safe lag-pair and kinetic helper functions.
- `main.py`: diagnostics command implementation.
- `apply.py`: conservative relabel implementation and relabel CV plotting.
- `radical.py`: radical whole-state removal and pair-ambiguous kNN core
  promotion.
- `plot.py`: diagnostic plots for confidence, entropy, labels, and legacy empty
  candidate overlays.

## Current Known Rough Edges

- The conservative wrapper is named `relabe_conservel.py` on disk. Agents should
  either use that name or rename wrappers deliberately in a separate cleanup.
- `plot.py` still contains legacy split/merge/missing overlay functions, but
  candidate lists are empty in the current diagnostics.
- Full execution requires the local Torch/scikit-learn/scipy environment used by
  TensorQ. In a lightweight shell, syntax checks may pass while runtime imports
  fail if those packages are absent.

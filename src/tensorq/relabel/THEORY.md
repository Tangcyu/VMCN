# Theory: Entropy -> Density -> Kinetics Relabeling

## Motivation

The committor-vector model predicts

```
q(x) = (q_1(x), ..., q_N(x))
```

where `q_i(x)` is the probability that a trajectory starting from configuration
`x` reaches labeled state `S_i` before the other labeled states.

Good labels should satisfy three practical conditions:

- frames inside state `S_i` should have high `q_i(x)`
- uncertain frames should not be forced to act as hard boundary labels
- each final label should be kinetically coherent

The relabeler therefore uses a deliberately narrow automatic pipeline:

```
entropy -> density -> kinetics
```

## 1. Entropy

Current committor entropy measures how committed the model is:

```
H(q(x)) = -sum_i q_i(x) * log(q_i(x))
H_norm(x) = H(q(x)) / log(N)
```

High `H_norm(x)` means the frame is ambiguous under the current state set.
Label-level inconsistency is checked first: if a state has too many
low-consistency or high-entropy frames, or too little stable high-confidence
core, the whole state is removed and its frames are set to `-1`.

After that state-level pruning, any remaining high-entropy labeled frame is set
to `-1` so it no longer constrains the next training round as a boundary frame.

Lagged entropy uses trajectory-safe pairs `(t, t + tau)` from the same
trajectory. It asks whether an uncertain frame quickly commits to an existing
basin:

- high current entropy and low lagged entropy: transition-like review frame
- high current entropy and high lagged entropy: persistent uncertain candidate
- insufficient valid lagged pairs: unresolved review frame

Only persistent uncertain candidates proceed to automatic new-label detection.

## 2. Density

Persistent uncertain candidates may contain true missing metastable cores, but
they can also contain diffuse transition regions. The density stage keeps the
automatic part conservative:

1. build kNN components in the configured graph space
2. reject tiny components
3. keep only the weighted local-density core
4. return density-shell frames to `-1`

The density core is the only source of new labels.

## 3. Kinetics

After entropy and density have reduced boundary contamination, kinetic checks
make the final labels more coherent:

- reshape existing labels to supported high-confidence q-cores
- split an existing label only when basin-internal lagged dynamics support a
  slow-mode separation
- merge labels that exchange like one metastate under lagged transition checks
- run a final kNN kinetic split check for weakly exchanging internal components

These checks depend on trajectory-safe lag pairs and must skip gracefully when
statistics are insufficient.

## What Was Removed

The relabeler still supports whole-state deletion from label inconsistency.
It no longer automatically promotes top-2 ambiguous frames, no longer runs
existing-core overlap filtering, and no longer writes per-frame CSV diagnostic
dumps. Those removed paths made the proposal difficult to audit and produced too
many stale side files.

## Outputs

Diagnostics and relabeling keep their records in YAML:

- diagnostics: `diagnostic_summary.yaml`
- relabeling: `relabel_summary.yaml`

The YAML summaries are the intended audit trail. Detailed frame inspection
should be generated explicitly in a separate analysis script when needed.

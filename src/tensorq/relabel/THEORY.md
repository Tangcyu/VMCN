# Theory: State-Label Confidence Diagnostics

## Motivation

The committor-vector model assumes a set of labeled core states S_1, ..., S_N. For each configuration x, the model predicts q(x) = (q_1(x), ..., q_N(x)), where q_i(x) is the probability that a trajectory starting from x reaches state S_i before the other labeled states.

If the current state labels are good:
- Frames inside core state S_i should have q_i(x) ≈ 1
- Frames inside different states should be distinguishable in committor space
- Each labeled state should be internally coherent
- Unlabeled regions should either be transition regions or should not show strong metastability

When these assumptions fail, we need diagnostics to decide whether to split, merge, or add states.

## Quality Decomposition

```
label quality = committor consistency + local kinetic stability + feature-space clustering
```

### 1. Committor Consistency

Per-frame entropy of the committor vector:

```
H(q(x)) = -sum_i q_i(x) * log(q_i(x))
H_norm(x) = H(q(x)) / log(N)
confidence(x) = 1 - H_norm(x)
```

High confidence means the model assigns a sharp committor value. Low confidence (high entropy) indicates ambiguity.

Label consistency: for a frame with label y(x), consistency = q_{y(x)}(x). If the model is consistent with the label, this should be close to 1.

### 2. Local Kinetic Stability

Estimated from short unbiased trajectories using lagged pairs (t, t+tau) within the same trajectory.

**Retention probability**: p_stay(C, tau) = P(x_{t+tau} in C | x_t in C). High retention indicates the region is kinetically stable at short timescales.

**Committor autocorrelation**: C_q^C(tau) measures how slowly the committor vector decorrelates inside region C. High autocorrelation at lag tau indicates slow dynamics within the region.

### 3. Confidence-Based Relabel Hints

The current implementation does not automatically call split, merge, or missing-state candidates. Those decisions are too sensitive to the choice of structural features, CVs, lag time, and clustering hyperparameters.

Instead, the diagnostic table flags states by:
- fraction of frames with q_label(x) below q_label_cutoff
- mean own-state committor q_i(x)
- mean normalized entropy
- dominant q-argmax destination among low-consistency frames

These signals answer a narrower and more reliable question: "Where do the current labels disagree with the learned committor?"

## Follow-Up Decision Workflow

### Possible Reassignment or Merge

If low-consistency frames from state S_i mostly have q-argmax S_j, inspect those frames in CV/structure space:
- If they occupy the S_j basin, relabel/reassign them.
- If S_i and S_j are not separable after retraining, consider merging.

### Possible Split

If a state has high low-consistency fraction but low-consistency frames point to multiple q-argmax destinations, inspect the distribution of q_i(x), entropy, and CVs inside that state. A split is plausible only if the uncertain frames form reproducible structural/kinetic subregions after retraining or targeted analysis.

### Possible Missing State

If high-entropy frames persist after obvious relabel/reassignment fixes and do not strongly point to any existing state, inspect whether they occupy a stable region in CV/structure space. A new state should be added only after confirming that the region is reproducible and kinetically meaningful.

## Important Constraints

1. Lagged pairs must not connect frames from different trajectories
2. Do not assume the global frame array is sorted by trajectory
3. All thresholds are configurable
4. Handle insufficient statistics robustly (return NaN, skip gracefully)
5. This is a diagnostic module only — it does not modify labels or retrain models

## Future Extensions (not implemented)

- Soft boundary conditions: confidence-weighted boundary loss for training
- Automatic relabeling based on diagnostic findings
- Targeted structural/CV inspection tools for low-confidence frames
- Optional, user-controlled clustering for exploratory visualization only
- Integration with active learning for state set refinement

## References

- The committor-vector framework follows the next-hit formulation: q_i(x) is the probability that state S_i is reached before all other labeled states.
- Kinetic stability estimation via lagged pairs is analogous to implied timescale analysis in Markov state models.

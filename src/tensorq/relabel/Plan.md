# Relabel Implementation Notes

Current relabel code paths:

- `main.py`: diagnostics CLI entry point used by `scripts/relabel_diagnose.py`.
- `label_diagnostics.py`: diagnostic feature construction and high-level hints.
- `diagnostics_io.py`: writes one compact `diagnostic_summary.yaml`.
- `relabel.py`: public relabel CLI/API, dataset writing, and final YAML summary.
- `entropy.py`: current and lagged entropy classification.
- `density.py` and `knn.py`: kNN components and weighted density-core selection.
- `kinetic_groups.py` and `kinetics.py`: basin-internal kinetic grouping plus
  final merge/split checks.
- `utils.py`: shared entropy scoring and relabeled-dataset writer.

Removed code:

- The old experimental `radical.py` and `relabel_new.py` implementations.
- Relabel CSV frame dumps and diagnostics CSV/JSON side outputs.
- Top-2 ambiguous relabel promotion and existing-core overlap filtering from
  `propose_relabeling`.

Relabeling now follows one automatic path:

1. **Entropy**: remove whole labels with too much inconsistency, mark
   high-current-entropy labeled frames as unlabeled, then use trajectory-safe
   lagged entropy to classify persistent uncertain candidates.
2. **Density**: build kNN components only from persistent high-entropy
   candidates and keep the weighted local-density core of each proposed basin.
3. **Kinetics**: reshape existing labels to supported q-cores, then apply the
   lagged merge/split consistency checks.

Kinetics performance notes:

- `kinetic_groups.py` standardizes features once per relabel proposal and
  reuses same-state lag-pair caches across labels.
- `basin_kinetic_groups.max_transition_pairs_per_state_lag: 0` keeps exact
  slow-mode transition counts; a positive value enables reproducible pair
  sampling for faster diagnostic runs.
- Final merge matrices use local label remapping plus `bincount`, avoiding the
  older `np.isin`/`add.at` path on every lag.

Expected outputs:

- Diagnose: `diagnostic_summary.yaml`
- Relabel: `relabel_summary.yaml`
- Optional relabeled dataset when `relabel.write_relabel_dataset: true`

Important follow-up checks after touching this package:

- `python3 -m py_compile src/tensorq/relabel/*.py`
- Import smoke test for `tensorq.relabel`
- A tiny synthetic `propose_relabeling` smoke test when changing the pipeline

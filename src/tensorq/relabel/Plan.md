# Relabel Implementation Notes

Current relabel code paths:

- `main.py`: diagnostics entry point used by `scripts/relabel_diagnose.py`.
- `label_diagnostics.py`: confidence, uncertainty, lagged entropy, and
  basin-internal kinetic group diagnostics.
- `kinetic_groups.py`: shared high-confidence q-core lag-component analysis.
- `relabel.py`: single relabeling implementation.
- `utils.py`: shared CSV, plotting, and relabeled-dataset writers.

The previous conservative/radical split has been removed. `scripts/relabel.py`
now calls `tensorq.relabel.relabel.main`.

Important follow-up checks after touching this package:

- `python -m py_compile` on all `src/tensorq/relabel/*.py` files.
- Import smoke test for `tensorq.relabel`.
- YAML parse check for `configs/relabel.example.yaml`.
- When possible, run the same checks in `conda myenv` on ERC005.

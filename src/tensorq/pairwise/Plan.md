# Cleanup Plan: tensorq.pairwise

## 1. Dead Code Removal

No dead code was found in the `pairwise` subpackage. All functions, classes, and imports are actively used.

## 2. Import Centralization

### Current pattern

All modules import from `tensorq.common` submodules individually. The most common pattern:

```python
from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import load_dataset, apply_stride, select_model_inputs, infer_n_states, unordered_pairs
from ..common.flux import make_thresholds, resolve_ordered_pairs
```

Note: `pairwise/*` imports `unordered_pairs` from `..common.data`, while `next_hit/*` imports it from `..common.flux`. This is because the function is duplicated in both modules (see `common/Plan.md` for de-duplication plan).

### Recommended migration

Once `common/__init__.py` is populated and `unordered_pairs` is de-duplicated:

```python
from ..common import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common import load_dataset, apply_stride, select_model_inputs, infer_n_states
from ..common import unordered_pairs, make_thresholds, resolve_ordered_pairs
```

**Affected files** (all 6 `.py` files except `__init__.py` and `losses.py`):
- `train.py`: 6 config + 9 data
- `infer.py`: 5 config + 5 data
- `predict.py`: 1 config + 1 data
- `rate_constant.py`: 5 config + 6 data + 2 flux
- `plot.py`: 5 config + 6 data

### unordered_pairs import path

After de-duplication in `common/Plan.md`, all `from ..common.data import unordered_pairs` should change to `from ..common.flux import unordered_pairs` (or use the top-level `from ..common import unordered_pairs`). This affects 5 files:
- `train.py`, `infer.py`, `predict.py`, `rate_constant.py`, `plot.py`

## 3. Structural Notes (non-urgent)

### reconstruct_state_probabilities

`predict.py` defines the primary implementation; `infer.py` calls it. Verify that `infer.py` does not contain a duplicate implementation — if it does, remove it and import from `predict.py`.

### rate_constant.py simplicity

Compared to `next_hit/rate_constant.py` (~1500 lines with full error analysis), `pairwise/rate_constant.py` is much simpler (~200 lines). Features missing from pairwise:
- Slice-based error estimation
- Standard deviation propagation through generator and MFPT
- Multiple flux computation backends (torch vs numpy)
- `sanitize_rate_matrix` with negative rate policies

These are feature gaps, not bugs. If pairwise rate analysis matures, consider extracting shared rate estimation infrastructure into `common/`.

### No flux consistency loss

The pairwise training loss does not include flux consistency. This is a design decision (pairwise q_ij lack the global normalization that makes flux profiles meaningful). No action needed, but worth documenting the rationale.

## 4. Unused Import Check

Verified: all imports in all 7 modules are used. No action needed.

# Cleanup Plan: tensorq.next_hit

## 1. Dead Code Removal

No dead code was found in the `next_hit` subpackage. All functions, classes, and imports are actively used within the package or imported by external consumers.

## 2. Import Centralization

### Current pattern

All modules import from `tensorq.common` submodules individually:

```python
from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import load_dataset, apply_stride, select_model_inputs, infer_n_states
from ..common.flux import make_thresholds, resolve_ordered_pairs
```

### Recommended migration

Once `common/__init__.py` is populated with re-exports (see `common/Plan.md`), update imports to:

```python
from ..common import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common import load_dataset, apply_stride, select_model_inputs, infer_n_states
from ..common import make_thresholds, resolve_ordered_pairs
```

**Affected files** (all 10 `.py` files except `__init__.py` and `model.py`):
- `train.py`: 6 config imports + 7 data imports + 2 flux imports
- `infer.py`: 5 config + 4 data + 1 flux
- `predict.py`: 1 config (`torch_load`)
- `losses.py`: 1 flux (`flux_consistency_loss`)
- `metrics.py`: no common imports
- `rate_constant.py`: 5 config + 4 data + 2 flux
- `fit_rate.py`: 5 config + 4 data + 2 flux
- `plot.py`: 5 config + 8 data + 3 flux
- `label.py`: 4 config + 1 data

**Migration strategy**: This is a cosmetic change only. Existing submodule-level imports continue to work since the original module bindings are not removed. Can be done incrementally or all at once.

## 3. Structural Notes (non-urgent)

### rate_constant.py size

At ~1500 lines, `rate_constant.py` is the largest module in the project. It mixes:
- Flux profile estimation (torch and numpy paths)
- Transition hit matrix computation
- Rate matrix assembly and sanitization
- MFPT and jump probability computation
- Error propagation (slice-based standard deviations)
- CSV and plot output writing

**Suggested split** (low priority, requires careful API design):
- `rate_constant/flux.py`: `estimate_flux_profiles`, `validate_rate_inputs`, `positive_weight_masks`
- `rate_constant/matrix.py`: `estimate_pi`, `estimate_transition_hit_matrix`, `matrix_from_pair_values`, `assemble_generator`, `sanitize_rate_matrix`, `compute_mfpt_matrix`, `compute_jump_probabilities`
- `rate_constant/errors.py`: `estimate_pi_std`, `estimate_transition_hit_matrix_std`, `estimate_slice_rate_std`, `propagate_generator_std`, `propagate_rate_matrix_std`
- `rate_constant/io.py`: all `write_*_csv` and `plot_*` functions

### fit_rate.py independence

`fit_rate.py` imports many specific functions from `rate_constant.py`. If `rate_constant.py` is split, update these imports accordingly.

### label.py lazy imports

`label.py` lazily imports MDAnalysis, sklearn, and scipy inside `require_*()` helper functions. This is deliberate to allow import of the module without those heavy dependencies. Do not change this pattern.

## 4. Unused Import Check

Verified: all imports in all 10 modules are used. No action needed.

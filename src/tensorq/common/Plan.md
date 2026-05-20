# Cleanup Plan: tensorq.common

## 1. Dead Code Removal

### P0 — `flux_profiles_numpy` in flux.py

- **Location**: `flux.py` lines ~173–217
- **Issue**: Defined but has zero callers anywhere in the codebase. The per-subpackage rate modules (`next_hit/rate_constant.py`, `pairwise/rate_constant.py`) each define their own `estimate_flux_profiles()` instead.
- **Action**: Remove the entire function body. The docstring also has a bug claiming 2 return values when the function returns 3 (J, var, C).
- **Risk**: None. Zero imports, zero callers.

### P0 — `unordered_pairs` duplication in data.py

- **Location**: `data.py` line ~324, `flux.py` line ~13
- **Issue**: Both files define identical `unordered_pairs(n_states)` functions. `data.py`'s version is used by `pair_labels_from_state` (same file) and imported by all `pairwise/*` modules. `flux.py`'s version is imported by `next_hit/*` modules.
- **Action**: In `data.py`, replace the definition with `from .flux import unordered_pairs`. This is safe because both implementations are identical list comprehensions.
- **Verification**: Check that all consumers still resolve correctly:
  - `data.py` internal: `pair_labels_from_state` uses `unordered_pairs` at lines 336–337
  - External consumers from `data`: `pairwise/train.py`, `pairwise/predict.py`, `pairwise/plot.py`, `pairwise/rate_constant.py`, `pairwise/infer.py`
  - External consumers from `flux`: `next_hit/rate_constant.py`, `next_hit/infer.py`, `next_hit/plot.py`
- **Risk**: Low. Only potential issue is if any consumer relies on `unordered_pairs` being in `data.py`'s namespace for pickle/serialization — but this is not the case for standard module imports.

## 2. Import Centralization

### Populate `__init__.py` with re-exports

**Current state**: `__init__.py` contains only a docstring. All consumers import directly from submodules:
```python
from tensorq.common.config import load_yaml, select_section, ensure_dir
from tensorq.common.data import load_dataset, select_model_inputs
from tensorq.common.flux import make_thresholds, resolve_ordered_pairs
```

**Target state**: `__init__.py` re-exports the public API so consumers can write:
```python
from tensorq.common import load_yaml, load_dataset, make_thresholds
```

**Symbols to re-export:**

From `config` (7):
- `load_yaml`, `select_section`, `ensure_dir`, `setup_device`, `set_seed`, `torch_load`, `write_yaml`

From `data` (12):
- `CommittorDatasetPack`, `load_dataset`, `apply_stride`, `cv_headers_for_pack`, `featurize_cv_inputs`, `select_model_inputs`, `infer_n_states`, `build_lagged_indices`, `LaggedCommittorDataset`, `PairwiseLaggedDataset`, `IndexSubset`, `split_train_val`, `pair_labels_from_state`

From `flux` (8):
- `unordered_pairs`, `all_ordered_pairs`, `resolve_ordered_pairs`, `make_thresholds`, `pair_index_tensors`, `reactive_current`, `crossing_weight`, `flux_profiles`, `flux_consistency_loss`

**Implementation pattern**:
```python
from .config import (
    load_yaml, select_section, ensure_dir, setup_device,
    set_seed, torch_load, write_yaml,
)
from .data import (
    CommittorDatasetPack, load_dataset, apply_stride,
    cv_headers_for_pack, featurize_cv_inputs, select_model_inputs,
    infer_n_states, build_lagged_indices, LaggedCommittorDataset,
    PairwiseLaggedDataset, IndexSubset, split_train_val, pair_labels_from_state,
)
from .flux import (
    unordered_pairs, all_ordered_pairs, resolve_ordered_pairs,
    make_thresholds, reactive_current, crossing_weight,
    flux_profiles, flux_consistency_loss,
)
```

**Migration strategy**: Existing `from tensorq.common.xxx import ...` patterns continue to work since module-level bindings are not removed. New code or refactored imports can use the shorter `from tensorq.common import ...` path. No urgency to update all call sites at once.

## 3. Notes

- `data.py` uses a lazy `import yaml` inside `load_dataset` (NPZ path only), while `config.py` imports yaml at module level. This is deliberate — the NPZ path is rarely taken. No action needed.
- `flux.py` has several internal-only helpers (`all_ordered_pairs`, `pair_index_tensors`, `reactive_current`, `crossing_weight`) that are called only within `flux.py` itself. They are properly scoped as implementation details. No action needed beyond re-exporting them for documentation completeness.

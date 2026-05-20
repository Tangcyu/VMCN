# Cleanup Plan: tensorq.relabel

## 1. Dead Code Removal

### Legacy candidate detection functions

- **Location**: `label_diagnostics.py`
- **Functions**: `detect_split_candidates`, `detect_merge_candidates`, `detect_missing_state_candidates`
- **Status**: Intentionally return empty lists per `Scheme.md` documentation ("disabled because they were too sensitive"). The functions still exist in the code.
- **Action**: No removal needed. These are documented as intentionally disabled, not dead. The empty-list return is a deliberate stub, not an oversight. If the functions are ever re-enabled, the stub bodies serve as reminders of the original intent.

### Legacy plot overlay functions

- **Location**: `plot.py`
- **Issue**: Per `Scheme.md`, `plot.py` "still contains legacy split/merge/missing overlay functions, but candidate lists are empty in the current diagnostics."
- **Action**: No removal needed now. These plot functions will be used if/when automatic candidate detection is re-enabled. Mark with a comment noting they are dormant, not dead.

## 2. Import Centralization

### Current pattern

```python
from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import load_dataset, apply_stride, select_model_inputs, infer_n_states
from ..next_hit.predict import infer_probabilities
```

### Recommended migration

Once `common/__init__.py` is populated (see `common/Plan.md`):

```python
from ..common import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common import load_dataset, apply_stride, select_model_inputs, infer_n_states
```

**Affected files**:
- `main.py`: config (5) + data (4) + next_hit.predict (1)
- `apply.py`: config (5) + data (4) + next_hit.predict (1)
- `radical.py`: config (5) + data (4) + next_hit.predict (1)
- `diagnostics_io.py`: config (3)
- `label_diagnostics.py`: no common imports (pure computation)
- `lag_pair_utils.py`: no common imports (pure computation)

## 3. Known Typo

### relabe_conservel.py

- **Location**: `scripts/relabe_conservel.py` (outside `src/`, in the scripts directory)
- **Issue**: File name has a typo: "conservel" should be "conservel" or "conservative".
- **Action**: Rename to `relabel_conservative.py`. Update any references in documentation or shell scripts.
- **Note**: The file correctly imports `tensorq.relabel.apply` internally — only the filename is affected.

## 4. Structural Notes

### diagnostics_io.py stale output cleanup

`diagnostics_io.py:save_results` removes stale clustering outputs:
- `split_candidates.csv`
- `split_candidates_details.json`
- `merge_candidates.csv`
- `missing_state_candidates.csv`
- `cluster_statistics.csv`

This cleanup is necessary because the old clustering-based diagnostics wrote these files. Since those candidate detectors are now disabled, the cleanup prevents stale files from misleading users. Keep this cleanup code.

### radical.py stale file removal

`radical.py` removes `radical_far_uncertain_review_frames.csv` if it exists. This is a legacy output file from an earlier version. Keep this cleanup until the next major version bump, then remove.

## 5. Unused Import Check

Based on the Explore agent analysis of the relabel subpackage (from the existing `Scheme.md` file map):
- `label_diagnostics.py` — verified all imports used
- `diagnostics_io.py` — verified all imports used
- `lag_pair_utils.py` — verified all imports used
- `main.py` — verified all imports used
- `apply.py` — verified all imports used
- `radical.py` — verified all imports used
- `plot.py` — verified all imports used

No unused imports detected in the relabel subpackage.

## 6. Summary

The relabel subpackage is the cleanest in the project:
- No dead functions (disabled functions are intentionally stubbed, not dead)
- No unused imports
- No duplicated code
- Well-documented in Scheme.md

The only actionable item is the `relabe_conservel.py` filename typo in `scripts/`.

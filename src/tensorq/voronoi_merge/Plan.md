# Cleanup Plan: tensorq.voronoi_merge

## 1. Dead Code Removal

No dead code (uncalled functions or unused imports) was found in the `voronoi_merge` subpackage. All defined functions, classes, and imports are actively used.

## 2. Import Centralization

### Current pattern

```python
from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
# io.py uses a conditional import inside load_samples:
# from ..common.data import apply_stride, load_dataset, select_model_inputs
```

### Recommended migration

Once `common/__init__.py` is populated (see `common/Plan.md`):

```python
from ..common import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common import apply_stride, load_dataset, select_model_inputs
```

**Affected files**:
- `runner.py`: 5 config imports
- `io.py`: conditional data imports (3 functions)

### io.py conditional import

`io.py` imports `..common.data` inside `load_samples()` rather than at module level:

```python
def load_samples(path, config, ...):
    from ..common.data import apply_stride, load_dataset, select_model_inputs
    ...
```

**Recommendation**: Move this to a module-level import. The conditional import pattern is fragile — if the import fails, the error only surfaces when `load_samples` is called with a `.pt`/`.npz` path, not at import time when the problem could be caught earlier. The `common` subpackage is always available in a valid TensorQ installation, so there is no circular import risk.

## 3. Structural Notes

### iterative.py size and inline plotting

`iterative.py` is ~1119 lines, making it the second-largest module in the project (after `next_hit/rate_constant.py` at ~1500 lines). A significant portion is plotting code inlined in the main iteration loop (~lines 976–1066).

**Suggested refactor** (low priority):
- Extract the per-iteration plot generation into `plot.py` as `plot_iteration_state(iteration, points, paths, labels, ...)`.
- The main loop in `run_iterative_pathway_expansion` should call this function rather than containing matplotlib code directly.

This would:
- Reduce `iterative.py` by ~100 lines.
- Make the algorithm logic easier to read.
- Allow the plotting code to be tested independently.

### _use_sincos_geometry visibility

- **Location**: `core.py` — defined with `_` prefix suggesting it's private.
- **Usage**: Explicitly imported by `iterative.py` via `from .core import _use_sincos_geometry`.
- **Action**: Either rename to `use_sincos_geometry` (making it public) or refactor `iterative.py` to use a public API. The function encapsulates a non-trivial decision about distance geometry that callers legitimately need.

### Periodic geometry overlap with gradpath

Both `voronoi_merge/core.py` and `gradpath/coordinates.py` handle periodic CVs via sin/cos embedding:

- `gradpath/coordinates.py`: `projected_cv_to_model_inputs` / `model_inputs_to_projected_cv`
- `voronoi_merge/core.py`: `periodic_sincos_embed` / `periodic_sincos_project`

These implementations are conceptually similar but differ in details (voronoi_merge uses scaled embedding, gradpath uses unscaled). Consider extracting shared periodic geometry primitives to `common/geometry.py` or similar. This would reduce duplication and ensure consistent handling across the project.

### json import in iterative.py

`iterative.py` has a lazy `import json` inside a save function. This is fine (json is stdlib), but for consistency with other modules, consider moving to module-level import.

## 4. Unused Import Check

Verified: all imports in all 6 modules are used. No action needed.

## 5. Testing Considerations

The iterative algorithm has many hyperparameters. A lightweight convergence test (small synthetic dataset, 2 paths, 5 iterations) would help catch regressions when refactoring. Currently there are no test files in the `voronoi_merge` directory.

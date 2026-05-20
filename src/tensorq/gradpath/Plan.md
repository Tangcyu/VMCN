# Cleanup Plan: tensorq.gradpath

## 1. Dead Code Removal

### P0 — `ArrayTransform` type alias in shooting.py

- **Location**: `shooting.py` line ~12
- **Issue**: Defined as `ArrayTransform = Callable[[np.ndarray], np.ndarray]` but never used anywhere in the codebase. The `Callable` import exists only for this alias.
- **Action**: Remove the `ArrayTransform` line and the `from typing import Callable` import if `Callable` has no other uses in the file.
- **Risk**: None. Zero references.

### P0 — `_project_paths_if_needed` in runner.py

- **Location**: `runner.py` lines ~170–199
- **Issue**: Defined but never called. The active code path in `run_gradpath()` uses `_project_and_finalize_periodic_paths` (line ~206) instead.
- **Action**: Remove the entire function body. Also check if its import of `model_inputs_to_projected_cv` from `.coordinates` is still needed (it is — `_project_and_finalize_periodic_paths` also uses it at line ~206).
- **Risk**: None. Zero callers. The newer `_project_and_finalize_periodic_paths` handles the same task with additional wrapping logic.

## 2. Code Consolidation

### P1 — Deduplicate `cluster_paths` and `cluster_paths_with_linkage`

- **Location**: `cluster.py` lines ~141–206 (`cluster_paths`) and ~209–311 (`cluster_paths_with_linkage`)
- **Issue**: Both functions share ~100 lines of identical code:
  - Calling `path_array()` and `path_weights()` (lines 141–149 ≈ 209–217)
  - Building the distance matrix (lines 151–161 ≈ 219–229)
  - The while-loop merging clusters using `_cluster_distance` (lines 163–183 ≈ 231–285)
  - Building `PathCluster` objects from final labels (lines 185–205 ≈ 287–310)
- **Difference**: `cluster_paths_with_linkage` additionally records a scipy linkage matrix during merging and includes `linkage_matrix` in its return value.

- **Action**: Extract shared logic into a private helper:
  ```python
  def _prepare_cluster_data(paths, weights, num_images, periods):
      """Return (path_array_3d, weights_array, distance_matrix)."""
      ...

  def _build_path_clusters(path_array_3d, weights_array, distance_matrix, labels, periods):
      """Build list[PathCluster] from cluster labels."""
      ...
  ```
  Then refactor both public functions to call these helpers, with `cluster_paths_with_linkage` adding only the linkage recording logic.

- **Risk**: Medium. Requires careful preservation of existing behavior. Both functions are part of the public API (exported in `__init__.py`). The internal refactor must not change return types or calling conventions.

## 3. Import Centralization

### Current pattern

Gradpath imports from `tensorq.common` submodules:

```python
from ..common.config import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common.data import load_dataset, apply_stride, select_model_inputs
```

Also imports from `next_hit`:
```python
from ..next_hit.predict import load_committor_model, infer_probabilities
```

### Recommended migration

Once `common/__init__.py` is populated (see `common/Plan.md`):

```python
from ..common import ensure_dir, load_yaml, select_section, setup_device, write_yaml
from ..common import load_dataset, apply_stride, select_model_inputs
```

**Affected files**:
- `runner.py`: config (5) + data (3) + next_hit.predict (2)
- `plot_runner.py`: config (5) + data (3) + next_hit.predict (2)
- `state_p.py`: config (5)
- `coordinates.py`: data (1: `cv_headers_for_pack`)

### Cross-subpackage imports

`runner.py` and `plot_runner.py` import from `..next_hit.predict`. This is a legitimate cross-subpackage dependency (gradpath needs a committor model to shoot paths). No action needed.

## 4. Notes

### fel_selection.py lazy imports

`fel_selection.py` lazily imports `scipy.ndimage.gaussian_filter` and `sklearn.cluster.KMeans` inside `_require_sklearn()`. This is deliberate — most gradpath runs use channel selection, not FEL/KDE selection. Keep this pattern.

### _example() in fel_selection.py

The `_example()` function at line ~698 creates synthetic 2D data and runs `select_fel_kde_centers`. It is only called under `if __name__ == "__main__"`. This is a development/demo function — consider moving it to a separate example script or adding a comment that it's for manual testing only.

### plot.py size

`plot.py` is ~999 lines and covers style setup, discrete colormaps, 2D/3D path plots, dendrogram plots, colored path plots, and periodic segment splitting. Consider splitting into `plot_style.py` and `plot_paths.py` if it grows beyond ~1200 lines.

## 5. Unused Import Check

Verified: all imports in all 10 modules are used (except `Callable` which is only used for the dead `ArrayTransform` type alias — see P0 above). No action needed beyond the P0 removal.

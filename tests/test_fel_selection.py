from pathlib import Path
import importlib.util
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GRADPATH_ROOT = ROOT / "src" / "tensorq" / "gradpath"
sys.path.insert(0, str(GRADPATH_ROOT))

import numpy as np

spec = importlib.util.spec_from_file_location("fel_selection", GRADPATH_ROOT / "fel_selection.py")
fel_selection = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["fel_selection"] = fel_selection
spec.loader.exec_module(fel_selection)

channel_selection_from_fel_result = fel_selection.channel_selection_from_fel_result
select_fel_kde_centers = fel_selection.select_fel_kde_centers
plot_fel_projection = fel_selection.plot_fel_projection
weighted_average_paths_by_fel_cluster = fel_selection.weighted_average_paths_by_fel_cluster


def sigmoid_channel(points, slope=7.0):
    q0 = 1.0 / (1.0 + np.exp(float(slope) * points[:, 0]))
    return np.stack([q0, 1.0 - q0], axis=1)


def test_select_fel_kde_centers_synthetic_channel():
    rng = np.random.default_rng(11)
    cv_left = rng.normal(loc=(-0.7, 0.0), scale=0.12, size=(200, 2))
    cv_right = rng.normal(loc=(0.7, 0.0), scale=0.12, size=(200, 2))
    cv_bridge = rng.normal(loc=(0.0, 0.0), scale=0.08, size=(80, 2))
    cv = np.vstack([cv_left, cv_right, cv_bridge])
    q0 = 1.0 / (1.0 + np.exp(7.0 * cv[:, 0]))
    q1 = 1.0 - q0
    q = np.stack([q0, q1], axis=1)

    result = select_fel_kde_centers(
        cv,
        q,
        (0, 1),
        bins=24,
        bandwidth_bins=1.0,
        ranges=[[-1.2, 1.2], [-0.6, 0.6]],
        threshold=0.20,
        kmin=1,
        kmax=4,
        points_per_cluster=2,
        q_evaluator=sigmoid_channel,
        random_state=0,
    )

    assert result.n_clusters >= 1
    assert result.selected_cv_points.shape[0] == 2 * result.n_clusters
    assert np.all(result.selected_q[:, 0] * result.selected_q[:, 1] > 0.20)
    assert np.all(result.selected_weights > 0.0)
    assert np.allclose(result.selected_weights, result.grid_density[result.selected_grid_indices])


def test_channel_selection_from_fel_result_and_weighted_cluster_average():
    cv = np.array([[-0.1], [0.0], [0.1], [0.2]], dtype=np.float64)
    q = np.array([[0.55, 0.45], [0.50, 0.50], [0.45, 0.55], [0.40, 0.60]], dtype=np.float64)
    result = select_fel_kde_centers(
        cv,
        q,
        (0, 1),
        bins=8,
        bandwidth_bins=0.5,
        ranges=[[-0.2, 0.3]],
        threshold=0.20,
        n_clusters=1,
        points_per_cluster=2,
        q_evaluator=lambda points: np.stack([0.5 - points[:, 0], 0.5 + points[:, 0]], axis=1),
    )
    selection = channel_selection_from_fel_result(result, result.selected_cv_points, (0, 1), threshold=0.20)

    assert selection.points.shape == result.selected_cv_points.shape
    assert selection.indices.tolist() == [0, 1]

    paths = np.stack(
        [
            np.column_stack([np.linspace(0.0, 1.0, 4), np.zeros(4)]),
            np.column_stack([np.linspace(0.0, 2.0, 4), np.ones(4)]),
        ],
        axis=0,
    )
    averaged = weighted_average_paths_by_fel_cluster(paths, np.array([1, 1]), np.array([3.0, 1.0]))

    assert set(averaged) == {1}
    assert np.allclose(averaged[1], np.average(paths, axis=0, weights=[3.0, 1.0]))


def test_plot_fel_projection_writes_plot_axes_projection():
    rng = np.random.default_rng(7)
    cv = rng.normal(size=(80, 2))
    q0 = 1.0 / (1.0 + np.exp(3.0 * cv[:, 0]))
    q = np.stack([q0, 1.0 - q0], axis=1)
    result = select_fel_kde_centers(
        cv,
        q,
        (0, 1),
        bins=12,
        bandwidth_bins=0.8,
        threshold=0.15,
        n_clusters=1,
        points_per_cluster=2,
        q_evaluator=lambda points: sigmoid_channel(points, slope=3.0),
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "fel_projection.png"
        plot_fel_projection(result, axes=["x", "y"], axis_names=["x", "y"], save_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0


def test_grid_q_product_uses_center_evaluator_not_kde_weights():
    rng = np.random.default_rng(5)
    cv = rng.normal(size=(120, 2))
    q = sigmoid_channel(cv, slope=4.0)
    ranges = [[-2.0, 2.0], [-2.0, 2.0]]
    result_a = select_fel_kde_centers(
        cv,
        q,
        (0, 1),
        weights=np.ones(cv.shape[0]),
        bins=10,
        bandwidth_bins=0.0,
        ranges=ranges,
        threshold=0.0,
        n_clusters=1,
        points_per_cluster=2,
        q_evaluator=lambda points: sigmoid_channel(points, slope=4.0),
        random_state=0,
    )
    result_b = select_fel_kde_centers(
        cv,
        q,
        (0, 1),
        weights=np.linspace(1.0, 5.0, cv.shape[0]),
        bins=10,
        bandwidth_bins=2.0,
        ranges=ranges,
        threshold=0.0,
        n_clusters=1,
        points_per_cluster=2,
        q_evaluator=lambda points: sigmoid_channel(points, slope=4.0),
        random_state=0,
    )

    assert np.allclose(result_a.grid_centers, result_b.grid_centers)
    assert np.allclose(result_a.grid_q_product, result_b.grid_q_product)

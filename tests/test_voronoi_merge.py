from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tensorq.voronoi_merge.core import (
    assign_voronoi_cells,
    cell_probabilities,
    kl_divergence,
    periodic_sincos_embed,
    periodic_sincos_project,
    periodic_sincos_weighted_mean,
)
from tensorq.voronoi_merge.io import coarsen_path_images
from tensorq.voronoi_merge.iterative import (
    assign_pathway_expansions,
    reparameterize_path_with_geometry,
    run_iterative_pathway_expansion,
    wrap_periodic_points,
)
from tensorq.voronoi_merge.plot import plot_pathway_iteration_2d


def test_voronoi_assignment_uses_periodic_minimum_image_distance():
    centers = np.array([[0.0], [350.0]], dtype=np.float64)
    points = np.array([[5.0], [349.0], [180.0]], dtype=np.float64)

    labels, distances = assign_voronoi_cells(points, centers, periods=[360.0])

    assert labels[:2].tolist() == [0, 1]
    assert np.allclose(distances[:2], [5.0, 1.0])


def test_voronoi_assignment_can_use_sincos_geometry():
    centers = np.array([[179.0], [0.0]], dtype=np.float64)
    points = np.array([[-179.0]], dtype=np.float64)

    labels, distances = assign_voronoi_cells(points, centers, periods=[360.0], periodic_geometry="sincos")

    assert labels.tolist() == [0]
    assert distances[0] < 3.0


def test_cell_probabilities_and_kld_match_formula():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    p = cell_probabilities(labels, 2)
    q = np.array([0.25, 0.75], dtype=np.float64)

    assert np.allclose(p, [0.5, 0.5])
    assert np.isclose(kl_divergence(p, q), 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75))


def test_coarsen_path_images_preserves_endpoints():
    path = np.arange(20, dtype=np.float64).reshape(10, 2)

    coarse = coarsen_path_images(path, image_stride=3)
    limited = coarsen_path_images(path, max_images_per_path=4)

    assert coarse[0].tolist() == path[0].tolist()
    assert coarse[-1].tolist() == path[-1].tolist()
    assert limited.shape == (4, 2)
    assert limited[0].tolist() == path[0].tolist()
    assert limited[-1].tolist() == path[-1].tolist()


def test_iterative_pathway_expansion_aligns_shared_segments_without_merging_pathways():
    path_a = np.column_stack([np.linspace(0.0, 1.0, 5), np.zeros(5)])
    path_b = path_a + np.array([0.02, 0.01])
    path_c = path_a + np.array([0.0, 5.0])
    paths = np.stack([path_a, path_b, path_c], axis=0)
    points = np.vstack([path_a, path_b, path_c])
    traj_id = np.repeat([0, 1, 2], 5)

    result = run_iterative_pathway_expansion(
        paths,
        points,
        traj_id=traj_id,
        terminal_image_margin=1,
        exchange_weight_mode="count",
        min_exchange_count=2.0,
        max_cell_distance=0.4,
        max_iterations=3,
        smooth_iterations=0,
    )

    assert len(result.paths) == 3
    assert all(p.shape == (5, 2) for p in result.paths)
    assert result.history[0].shared_segments


def test_pathway_iteration_plot_writes_png(tmp_path):
    paths = np.stack(
        [
            np.column_stack([np.linspace(0.0, 1.0, 4), np.zeros(4)]),
            np.column_stack([np.linspace(0.0, 1.0, 4), np.ones(4)]),
        ],
        axis=0,
    )
    points = np.vstack([paths[0], paths[1]])
    distances = np.vstack(
        [
            np.linalg.norm(points[:, None, :] - paths[0][None, :, :], axis=2).min(axis=1),
            np.linalg.norm(points[:, None, :] - paths[1][None, :, :], axis=2).min(axis=1),
        ]
    )
    out = tmp_path / "iteration.png"

    fig = plot_pathway_iteration_2d(points, paths, distances, save_path=out, config={"plot_max_points": 0})
    import matplotlib.pyplot as plt

    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 0


def test_wrap_periodic_points_uses_requested_bounds():
    values = np.array([[-190.0, 181.0], [540.0, -540.0]], dtype=np.float64)

    wrapped = wrap_periodic_points(values, periods=[360.0, 360.0], wrap_bounds=[[-180.0, 180.0], [-180.0, 180.0]])

    assert np.all(wrapped >= -180.0)
    assert np.all(wrapped < 180.0)
    assert np.allclose(wrapped[0], [170.0, -179.0])


def test_periodic_sincos_projection_and_mean_do_not_cross_branch_cut():
    points = np.array([[179.0], [-179.0]], dtype=np.float64)

    projected = periodic_sincos_project(periodic_sincos_embed(points, periods=[360.0]), periods=[360.0])
    center = periodic_sincos_weighted_mean(points, np.ones(2), periods=[360.0])

    assert np.allclose(wrap_periodic_points(projected, periods=[360.0]), points)
    assert np.isclose(abs(center[0]), 180.0)


def test_sincos_reparameterization_uses_short_periodic_arc():
    path = np.array([[170.0], [-170.0]], dtype=np.float64)

    reparam = reparameterize_path_with_geometry(path, 3, periods=[360.0], periodic_geometry="sincos")
    wrapped = wrap_periodic_points(reparam, periods=[360.0])

    assert abs(wrapped[1, 0]) > 175.0


def test_pathway_assignment_can_use_sincos_periodic_geometry():
    paths = np.array([[[179.0], [0.0]], [[-90.0], [179.0]]], dtype=np.float64)
    points = np.array([[-179.0]], dtype=np.float64)

    labels, distances = assign_pathway_expansions(points, paths, periods=[360.0], periodic_geometry="sincos")

    assert labels[:, 0].tolist() == [0, 1]
    assert distances[0, 0] < 3.0


def test_iterative_pathway_keeps_fixed_endpoints_and_relaxes_dynamic_cell_images():
    path_a = np.column_stack([np.linspace(0.0, 1.0, 5), np.zeros(5)])
    path_b = path_a.copy()
    path_b[1:-1] += np.array([0.2, 0.1])
    paths = np.stack([path_a, path_b], axis=0)
    points = np.vstack([path_a, path_b])
    traj_id = np.repeat([0, 1], 5)

    result = run_iterative_pathway_expansion(
        paths,
        points,
        traj_id=traj_id,
        terminal_image_margin=1,
        min_exchange_count=2.0,
        max_cell_distance=0.5,
        max_iterations=1,
        smooth_iterations=0,
        fixed_endpoints=True,
        exchange_weight_mode="count",
        cell_relaxation=1.0,
    )

    assert np.allclose(result.paths[0][0], paths[0, 0])
    assert np.allclose(result.paths[0][-1], paths[0, -1])
    assert len(result.paths) == paths.shape[0]
    assert not np.allclose(result.history[0].paths[0][2], result.paths[0][2])

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from tensorq.common.data import pair_labels_from_state
from tensorq.gradpath.cluster import cluster_paths
from tensorq.gradpath.coordinates import model_inputs_to_projected_cv, projected_cv_to_model_inputs
from tensorq.gradpath.selection import select_channel_points
from tensorq.gradpath.shooting import build_channel_paths, reparameterize_path, shoot_batch_to_state
from tensorq.next_hit.model import NextHitCommittorNet
from tensorq.next_hit.plot import (
    average_binned_field,
    collapse_binned_field,
    destination_field_from_q_fields,
    weighted_mean_2d,
    weighted_mean_nd,
)
from tensorq.next_hit.rate_constant import (
    add_mfpt_rates_to_table,
    assemble_generator,
    build_rate_std_table,
    build_rate_table,
    compute_mfpt_matrix,
    compute_mfpt_rate_matrix,
    estimate_flux_profiles,
    positive_weight_masks,
    sanitize_rate_matrix,
)
from tensorq.pairwise.model import PairwiseCommittorNet
from tensorq.pairwise.predict import infer_n_states_from_pair_dim, reconstruct_state_probabilities


def test_next_hit_model_outputs_simplex():
    model = NextHitCommittorNet(in_dim=4, n_states=3, hidden=[8])
    q = model(torch.randn(5, 4))
    assert q.shape == (5, 3)
    assert torch.all(q >= 0)
    assert torch.allclose(q.sum(dim=1), torch.ones(5), atol=1e-6)


def test_next_hit_rate_table_kij_uses_mfpt_rate():
    pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    direct_rates = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    table, _J_matrix, k_direct = build_rate_table(
        pairs=pairs,
        J=direct_rates[:, None],
        variance=np.zeros(len(pairs), dtype=np.float64),
        pi=np.ones(3, dtype=np.float64),
        time_unit="step",
    )
    k_mfpt = compute_mfpt_rate_matrix(compute_mfpt_matrix(assemble_generator(k_direct)))

    table = add_mfpt_rates_to_table(table, pairs, k_mfpt)

    assert np.isclose(table.loc[0, "k_direct_ij"], direct_rates[0])
    assert np.isclose(table.loc[0, "k_mfpt_ij"], k_mfpt[0, 1])
    assert np.isclose(table.loc[0, "k_ij"], k_mfpt[0, 1])
    assert not np.isclose(table.loc[0, "k_ij"], table.loc[0, "k_direct_ij"])


def test_next_hit_rate_std_table_kij_uses_mfpt_std():
    pairs = [(0, 1), (1, 0)]
    pi_std = np.array([0.01, 0.02], dtype=np.float64)
    J_matrix_std = np.array([[np.nan, 0.1], [0.2, np.nan]], dtype=np.float64)
    k_direct_std = np.array([[np.nan, 1.0], [2.0, np.nan]], dtype=np.float64)
    k_mfpt_std = np.array([[np.nan, 0.3], [0.4, np.nan]], dtype=np.float64)

    table = build_rate_std_table(pairs, pi_std, J_matrix_std, k_direct_std, "step", k_mfpt_std=k_mfpt_std)

    assert np.isclose(table.loc[0, "k_direct_ij_std"], 1.0)
    assert np.isclose(table.loc[0, "k_mfpt_ij_std"], 0.3)
    assert np.isclose(table.loc[0, "k_ij_std"], 0.3)


def test_next_hit_sanitizes_negative_direct_rates_before_generator():
    raw = np.array([[0.0, -1e-12, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]], dtype=np.float64)

    rates, stats = sanitize_rate_matrix(raw)
    K = assemble_generator(raw)

    assert stats["n_negative_offdiag_rates"] == 1
    assert rates[0, 1] == 0.0
    assert K[0, 1] == 0.0
    assert np.all(K[~np.eye(3, dtype=bool)] >= 0.0)
    assert np.allclose(K.sum(axis=1), 0.0)


def test_next_hit_rate_masks_zero_weight_lagged_endpoints():
    q = np.array(
        [
            [0.9, 0.1],
            [0.0, 1.0],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )
    weights = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    idx0 = np.array([0, 0, 1], dtype=np.int64)
    idx1 = np.array([1, 2, 2], dtype=np.int64)

    frame_mask, pair_mask, stats = positive_weight_masks(weights, idx0, idx1)

    assert frame_mask.tolist() == [True, False, True]
    assert pair_mask.tolist() == [False, True, False]
    assert stats["n_lagged_pairs_removed_by_zero_weight_mask"] == 2

    J, _variance = estimate_flux_profiles(
        q=q,
        weights=weights,
        idx0=idx0[pair_mask],
        idx1=idx1[pair_mask],
        pairs=[(0, 1)],
        thresholds=np.array([0.5], dtype=np.float64),
        eps=1e-3,
        tau=1.0,
        divide_by_tau=False,
        surface="qi_decrease",
        chunk_size=4,
        weighted=True,
        device="cpu",
    )

    assert np.allclose(J, [[0.7]], atol=1e-6)


def test_pairwise_labels_derive_from_meta_state():
    labels = pair_labels_from_state(torch.tensor([0, 1, 2, -1]), n_states=3)
    assert labels.tolist() == [[0, 0, -1], [1, -1, 0], [-1, 1, 1], [-1, -1, -1]]


def test_pairwise_model_outputs_pairwise_committors():
    model = PairwiseCommittorNet(in_dim=4, n_pairs=3, hidden=[8])
    q = model(torch.randn(5, 4))
    assert q.shape == (5, 3)
    assert torch.all(q >= 0)
    assert torch.all(q <= 1)


def test_reconstruct_state_probabilities_from_pairwise_q():
    P_ref = np.array([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]], dtype=np.float32)
    Q = np.stack(
        [
            P_ref[:, 1] / (P_ref[:, 0] + P_ref[:, 1]),
            P_ref[:, 2] / (P_ref[:, 0] + P_ref[:, 2]),
            P_ref[:, 2] / (P_ref[:, 1] + P_ref[:, 2]),
        ],
        axis=1,
    )
    P = reconstruct_state_probabilities(Q, n_states=3, eps=1e-6)
    assert P.shape == P_ref.shape
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(P, P_ref, atol=1e-4)


def test_infer_n_states_from_pair_dim():
    assert infer_n_states_from_pair_dim(6) == 4


def test_binned_3d_committor_collapse_matches_direct_2d_average():
    x = np.array([0.25, 0.25, 0.25, 1.25], dtype=np.float64)
    y = np.array([0.25, 0.25, 0.25, 1.25], dtype=np.float64)
    z = np.array([0.25, 1.25, 1.25, 0.25], dtype=np.float64)
    q = np.array([0.2, 0.6, 1.0, 0.7], dtype=np.float64)
    weights = np.array([1.0, 1.0, 3.0, 2.0], dtype=np.float64)
    edges = [np.array([0.0, 1.0, 2.0], dtype=np.float64) for _ in range(3)]

    field_3d, denom_3d = weighted_mean_nd([x, y, z], q, weights, edges)
    collapsed = collapse_binned_field(field_3d, denom_3d, axis=2)
    direct = weighted_mean_2d(x, y, q, weights, edges[0], edges[1])

    assert np.allclose(collapsed, direct, equal_nan=True)
    assert np.isclose(field_3d[0, 0, 1], 0.9)
    assert np.isnan(collapsed[0, 1])


def test_destination_field_uses_projected_committor_values():
    q0 = np.array([[0.8, np.nan], [0.3, 0.4]], dtype=np.float64)
    q1 = np.array([[0.2, np.nan], [0.7, 0.6]], dtype=np.float64)

    dest = destination_field_from_q_fields([q0, q1])

    expected = np.array([[0.0, np.nan], [1.0, 1.0]], dtype=np.float64)
    assert np.allclose(dest, expected, equal_nan=True)


def test_center_grid_projection_averages_finite_bin_values():
    field = np.array(
        [
            [[0.2, 0.4], [np.nan, np.nan]],
            [[0.8, 1.0], [0.1, np.nan]],
        ],
        dtype=np.float64,
    )

    projected = average_binned_field(field, axis=2)

    expected = np.array([[0.3, np.nan], [0.9, 0.1]], dtype=np.float64)
    assert np.allclose(projected, expected, equal_nan=True)


def test_gradpath_select_channel_points_uses_weighted_sampling():
    points = np.arange(10, dtype=np.float64).reshape(5, 2)
    q = np.array(
        [
            [0.90, 0.10],
            [0.50, 0.50],
            [0.45, 0.55],
            [0.10, 0.90],
            [0.80, 0.20],
        ],
        dtype=np.float64,
    )
    weights = np.array([1.0, 10.0, 1.0, 1.0, 1.0], dtype=np.float64)

    selection = select_channel_points(points, q, 0, 1, threshold=0.20, weights=weights, max_points=2, seed=4)

    assert selection.points.shape == (2, 2)
    assert np.all(selection.channel_score >= 0.20)
    assert np.all(selection.weights > 0.0)


class TwoStateLinearCommittor(torch.nn.Module):
    def forward(self, x):
        logits = torch.stack([-x[:, 0], x[:, 0]], dim=1)
        return torch.softmax(logits, dim=1)


def test_gradpath_shooting_stitches_i_to_j_path():
    model = TwoStateLinearCommittor()
    points = np.array([[0.0], [0.1]], dtype=np.float64)
    q = np.array([[0.5, 0.5], [0.45, 0.55]], dtype=np.float64)
    selection = select_channel_points(points, q, 0, 1, threshold=0.20, weights=np.ones(2), max_points=1, seed=1)

    paths = build_channel_paths(
        model,
        selection,
        step_size=0.25,
        max_steps=20,
        target_q=0.80,
        num_images=9,
        device="cpu",
        dtype=torch.float32,
    )

    assert len(paths) == 1
    assert paths[0].path.shape == (9, 1)
    assert paths[0].q_path.shape == (9, 2)
    assert paths[0].path[0, 0] < paths[0].path[-1, 0]
    assert paths[0].q_path[0, 0] > paths[0].q_path[-1, 0]


def test_gradpath_batch_shooting_preserves_path_order():
    model = TwoStateLinearCommittor()
    starts = np.array([[-0.2], [0.0], [0.2]], dtype=np.float64)

    paths, q_paths = shoot_batch_to_state(
        model,
        starts,
        1,
        step_size=0.2,
        max_steps=10,
        target_q=0.75,
        device="cpu",
        dtype=torch.float32,
        integration_batch_size=2,
    )

    assert len(paths) == 3
    assert len(q_paths) == 3
    assert np.isclose(paths[0][0, 0], starts[0, 0])
    assert np.isclose(paths[2][0, 0], starts[2, 0])
    assert q_paths[0][-1, 1] >= q_paths[0][0, 1]


def test_gradpath_expansion_stops_by_basin_radius_not_raw_q():
    model = TwoStateLinearCommittor()
    starts = np.array([[0.0]], dtype=np.float64)

    paths, q_paths = shoot_batch_to_state(
        model,
        starts,
        1,
        step_size=0.2,
        max_steps=20,
        target_q=0.9999,
        device="cpu",
        dtype=torch.float32,
        expansion=True,
        basin_center=np.array([0.6], dtype=np.float64),
        basin_radius=0.05,
    )

    assert len(paths) == 1
    assert abs(paths[0][-1, 0] - 0.6) <= 0.05
    assert q_paths[0][-1, 1] < 0.9999


def test_gradpath_weighted_clustering_finds_centers():
    base = np.column_stack([np.linspace(0.0, 1.0, 6), np.linspace(0.0, 0.5, 6)])
    paths = np.stack(
        [
            base,
            base + 0.01,
            base + np.array([0.0, 2.0]),
        ],
        axis=0,
    )
    labels, clusters, distance = cluster_paths(paths, weights=np.array([3.0, 1.0, 1.0]), distance_threshold=0.05)

    assert labels.tolist() == [1, 1, 2]
    assert len(clusters) == 2
    assert distance.shape == (3, 3)
    assert np.allclose(clusters[0].center_path, np.average(paths[:2], axis=0, weights=[3.0, 1.0]))
    assert reparameterize_path(base, 4).shape == (4, 2)


def test_gradpath_periodic_cv_projection_roundtrip_and_unwraps():
    meta = {
        "model_input_space": "cv",
        "model_feature_names": ["sinphi1", "cosphi1", "sinphi2", "cosphi2"],
        "model_cvs_to_use": ["phi1", "phi2"],
        "model_periodic_cvs": ["phi1", "phi2"],
        "model_periodic_cv_units": "degrees",
    }
    cv = np.array([[60.0, -70.0], [170.0, -170.0], [-170.0, 170.0]], dtype=np.float64)

    model_points = projected_cv_to_model_inputs(cv, meta)
    wrapped = model_inputs_to_projected_cv(model_points, meta)
    unwrapped = model_inputs_to_projected_cv(model_points, meta, unwrap=True)

    assert model_points.shape == (3, 4)
    assert np.allclose(wrapped, cv, atol=1e-10)
    assert np.allclose(unwrapped[:, 0], [60.0, 170.0, 190.0], atol=1e-10)
    assert np.allclose(projected_cv_to_model_inputs(cv[0], meta), model_points[0])

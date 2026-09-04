from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tensorq.next_hit.rate_constant import filter_kinetic_edges, trajectory_burn_in_mask


def test_trajectory_burn_in_is_applied_to_each_consecutive_block():
    traj_id = torch.tensor([4, 4, 4, 9, 9, 4, 4])

    keep, stats = trajectory_burn_in_mask(traj_id, n_frames=7, discard_first_n_frames=2)

    assert keep.tolist() == [False, False, True, False, False, False, False]
    assert stats["n_trajectory_blocks"] == 3
    assert stats["n_frames_discarded"] == 6
    assert stats["n_frames_retained"] == 1


def test_trajectory_burn_in_rejects_negative_discard():
    with pytest.raises(ValueError, match="must be nonnegative"):
        trajectory_burn_in_mask(None, n_frames=3, discard_first_n_frames=-1)


def test_jump_probability_filter_removes_only_directed_edges_below_threshold():
    rates = np.array(
        [
            [0.0, 0.04, 0.01],
            [0.20, 0.0, 0.10],
            [0.30, 0.20, 0.0],
        ],
        dtype=np.float64,
    )
    probabilities = np.array(
        [
            [0.0, 0.05, 0.049],
            [0.04, 0.0, 0.96],
            [0.60, 0.40, 0.0],
        ],
        dtype=np.float64,
    )

    filtered, removed, stats = filter_kinetic_edges(
        rates,
        probabilities,
        config={
            "kinetic_edge_filter": {
                "enabled": True,
                "min_jump_probability": 0.05,
                "min_rate_zscore": None,
                "preserve_connectivity": False,
            }
        },
    )

    assert filtered[0, 1] == rates[0, 1]  # Equality is retained: the rule is P_ij < cutoff.
    assert filtered[0, 2] == 0.0
    assert filtered[1, 0] == 0.0
    assert removed[0, 2]
    assert removed[1, 0]
    assert stats["n_removed_edges"] == 2


def test_jump_probability_filter_is_disabled_by_default():
    rates = np.array([[0.0, 0.01], [0.02, 0.0]], dtype=np.float64)
    probabilities = np.array([[0.0, 0.01], [0.01, 0.0]], dtype=np.float64)

    filtered, removed, stats = filter_kinetic_edges(rates, probabilities)

    np.testing.assert_array_equal(filtered, rates)
    assert not np.any(removed)
    assert not stats["enabled"]

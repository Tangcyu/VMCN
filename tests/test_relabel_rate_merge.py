from pathlib import Path

import numpy as np

from tensorq.relabel.rate_merge import merge_macrostates_from_rates


def _config(tmp_path: Path, *, bidirectional: bool = False) -> dict:
    probability = tmp_path / "P_jump.npy"
    mfpt = tmp_path / "MFPT.npy"
    np.save(probability, np.asarray([[0.0, 0.96], [0.2, 0.0]]))
    np.save(mfpt, np.asarray([[0.0, 20.0], [500.0, 0.0]]))
    return {
        "relabel": {
            "rate_merge_enabled": True,
            "rate_merge_probability_path": str(probability),
            "rate_merge_mfpt_path": str(mfpt),
            "rate_merge_probability_cutoff": 0.95,
            "rate_merge_mfpt_cutoff_frames": 100,
            "rate_merge_require_bidirectional": bidirectional,
        }
    }


def test_rate_merge_requires_probability_and_short_mfpt_in_one_direction(tmp_path):
    merged, groups, edges, mapping = merge_macrostates_from_rates(
        np.asarray([0, 1, -1]), _config(tmp_path), 2
    )
    assert merged.tolist() == [0, 0, -1]
    assert groups[0]["states"] == [0, 1]
    assert edges[0]["qualifies_ij"] is True
    assert mapping == [{"label_before_compaction": 0, "label": 0}]


def test_rate_merge_bidirectional_mode_requires_both_directions(tmp_path):
    merged, groups, edges, _ = merge_macrostates_from_rates(
        np.asarray([0, 1]), _config(tmp_path, bidirectional=True), 2
    )
    assert merged.tolist() == [0, 1]
    assert groups == []
    assert edges == []


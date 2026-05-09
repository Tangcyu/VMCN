from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from tensorq.common.data import pair_labels_from_state
from tensorq.next_hit.model import NextHitCommittorNet
from tensorq.pairwise.model import PairwiseCommittorNet
from tensorq.pairwise.predict import infer_n_states_from_pair_dim, reconstruct_state_probabilities


def test_next_hit_model_outputs_simplex():
    model = NextHitCommittorNet(in_dim=4, n_states=3, hidden=[8])
    q = model(torch.randn(5, 4))
    assert q.shape == (5, 3)
    assert torch.all(q >= 0)
    assert torch.allclose(q.sum(dim=1), torch.ones(5), atol=1e-6)


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

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _activation(name: str) -> type[nn.Module]:
    acts: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    return acts.get(str(name).lower(), nn.ELU)


class PairwiseCommittorNet(nn.Module):
    """
    MLP mapping features z to unordered pair-wise committors q_ij(z).

    For pair (i, j), q_ij is trained as 0 in state i and 1 in state j.
    """

    def __init__(
        self,
        in_dim: int,
        n_pairs: int,
        hidden: Iterable[int] = (256, 256, 128),
        activation: str = "elu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_pairs = int(n_pairs)
        self.hidden = tuple(int(x) for x in hidden)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)
        self.output_activation = str(output_activation).lower()
        if self.in_dim < 1:
            raise ValueError("in_dim must be positive.")
        if self.n_pairs < 1:
            raise ValueError("n_pairs must be positive.")
        if self.output_activation not in {"sigmoid", "identity"}:
            raise ValueError("output_activation must be 'sigmoid' or 'identity'.")

        Act = _activation(self.activation)
        layers: list[nn.Module] = []
        prev = self.in_dim
        for h in self.hidden:
            layers.append(nn.Linear(prev, h))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(Act())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, self.n_pairs))
        self.net = nn.Sequential(*layers)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.logits(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(raw)
        return raw

    def model_kwargs(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "n_pairs": self.n_pairs,
            "hidden": list(self.hidden),
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "output_activation": self.output_activation,
        }

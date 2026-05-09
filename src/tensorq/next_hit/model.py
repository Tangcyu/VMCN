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


def _normalize_output_name(name: str) -> str:
    value = str(name).lower().replace("-", "_")
    aliases = {
        "softmax": "softmax",
        "positive_l1": "positive_l1",
        "softplus_l1": "positive_l1",
        "l1": "positive_l1",
        "normalize": "positive_l1",
        "normalized": "positive_l1",
    }
    if value not in aliases:
        raise ValueError("output_normalization must be 'softmax' or 'positive_l1'.")
    return aliases[value]


class NextHitCommittorNet(nn.Module):
    """
    MLP mapping features z to a normalized multi-state next-hit committor q(z).

    forward(x) returns probabilities with shape (batch_size, n_states):
      q_j >= 0 and sum_j q_j = 1.
    """

    def __init__(
        self,
        in_dim: int,
        n_states: int,
        hidden: Iterable[int] = (256, 256, 128),
        activation: str = "elu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        output_normalization: str = "softmax",
        positive_eps: float = 1e-8,
        use_softmax_output: bool | None = None,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_states = int(n_states)
        self.hidden = tuple(int(x) for x in hidden)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)
        if use_softmax_output is not None:
            output_normalization = "softmax" if bool(use_softmax_output) else "positive_l1"
        self.output_normalization = _normalize_output_name(output_normalization)
        self.positive_eps = float(positive_eps)
        if self.positive_eps <= 0.0:
            raise ValueError("positive_eps must be positive.")

        if self.in_dim < 1:
            raise ValueError("in_dim must be positive.")
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2.")

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

        layers.append(nn.Linear(prev, self.n_states))
        self.net = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        if self.output_normalization == "softmax":
            return self.softmax(logits)
        positive = self.softplus(logits) + self.positive_eps
        return positive / positive.sum(dim=-1, keepdim=True).clamp_min(self.positive_eps)

    def model_kwargs(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "n_states": self.n_states,
            "hidden": list(self.hidden),
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "output_normalization": self.output_normalization,
            "positive_eps": self.positive_eps,
        }

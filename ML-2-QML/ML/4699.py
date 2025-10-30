"""FraudDetectionHybrid – classical implementation.

The module contains a reusable neural stack that mirrors the
photonic fraud‑detection circuit and augments it with a
self‑attention layer.  The design is intentionally modular so
that the same weight tensors can be transferred to the quantum
counterpart if desired.

Typical usage:

    from FraudDetection__gen146 import FraudDetectionHybrid
    hybrid = FraudDetectionHybrid(num_features=2, depth=2, embed_dim=4)
    model = hybrid.build_classical()
    # model is a torch.nn.Sequential ready for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class SelfAttentionModule(nn.Module):
    """Drop‑in replacement for the classical self‑attention helper."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        query = torch.matmul(inputs, rotation_params.reshape(self.embed_dim, -1))
        key = torch.matmul(inputs, entangle_params.reshape(self.embed_dim, -1))
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, inputs)


def build_classifier_circuit(num_features: int, depth: int) -> nn.Sequential:
    """Create a feed‑forward classifier with metadata mirroring the quantum version."""
    layers: list[nn.Module] = []
    in_dim = num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    return nn.Sequential(*layers)


class FraudDetectionHybrid:
    """
    Public façade that produces either a classical or a quantum fraud‑detection model.
    The constructor stores the hyper‑parameters; the build methods instantiate
    the appropriate PyTorch or Qiskit objects.
    """

    def __init__(
        self,
        num_features: int = 2,
        depth: int = 2,
        embed_dim: int = 4,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.embed_dim = embed_dim
        self.fraud_params = list(fraud_params) if fraud_params else []

    def build_classical(self) -> nn.Sequential:
        """Return a torch.nn.Sequential that mimics the photonic architecture."""
        # Base fraud‑detection layers
        modules: list[nn.Module] = [
            _layer_from_params(self.fraud_params[0], clip=False)
        ]
        modules.extend(
            _layer_from_params(p, clip=True) for p in self.fraud_params[1:]
        )
        modules.append(nn.Linear(2, 1))

        # Attach the self‑attention block
        sa = SelfAttentionModule(self.embed_dim)
        modules.append(sa)

        # Classifier head
        modules.append(build_classifier_circuit(self.num_features, self.depth))

        return nn.Sequential(*modules)


__all__ = ["FraudLayerParameters", "_layer_from_params", "SelfAttentionModule",
           "build_classifier_circuit", "FraudDetectionHybrid"]

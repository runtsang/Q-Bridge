"""Hybrid classical GraphQNN implementation combining GNN utilities, self‑attention, regression, and fraud‑detection style scaling."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Self‑Attention helper
# --------------------------------------------------------------------------- #
class SelfAttention:
    """Simple self‑attention block using linear projections and soft‑max."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.rotation_params = torch.randn(embed_dim, embed_dim)
        self.entangle_params = torch.randn(embed_dim, embed_dim)

    def run(self, inputs: Tensor) -> Tensor:
        query = inputs @ self.rotation_params
        key = inputs @ self.entangle_params
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


# --------------------------------------------------------------------------- #
# Fraud‑Detection style scaling layer
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


# --------------------------------------------------------------------------- #
# Simple regression estimator
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Tiny fully‑connected regressor used as a final estimator."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(inputs)


# --------------------------------------------------------------------------- #
# GraphQNN class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """
    Hybrid GNN that can optionally include self‑attention, fraud‑style scaling,
    and a regression estimator.  Methods mirror the original GraphQNN
    utilities while adding new layers.
    """
    def __init__(
        self,
        arch: Sequence[int],
        use_attention: bool = True,
        use_fraud: bool = True,
        use_estimator: bool = True,
    ) -> None:
        self.arch = list(arch)
        self.use_attention = use_attention
        self.use_fraud = use_fraud
        self.use_estimator = use_estimator

        self.modules: List[nn.Module] = []
        for i in range(1, len(self.arch)):
            linear = nn.Linear(self.arch[i - 1], self.arch[i])
            self.modules.append(linear)
            self.modules.append(nn.Tanh())
            if self.use_attention:
                self.modules.append(SelfAttention(self.arch[i]))
            if self.use_fraud:
                params = FraudLayerParameters(
                    bs_theta=random.random(),
                    bs_phi=random.random(),
                    phases=(random.random(), random.random()),
                    squeeze_r=(random.random(), random.random()),
                    squeeze_phi=(random.random(), random.random()),
                    displacement_r=(random.random(), random.random()),
                    displacement_phi=(random.random(), random.random()),
                    kerr=(random.random(), random.random()),
                )
                self.modules.append(_layer_from_params(params, clip=False))
        # Final output layer
        self.modules.append(nn.Linear(self.arch[-1], 1))
        self.net = nn.Sequential(*self.modules)

        self.estimator = EstimatorNN() if self.use_estimator else None

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def random_network(self, samples: int) -> Tuple[List[int], List[nn.Parameter], List[Tuple[Tensor, Tensor]], Tensor]:
        """Re‑initialize all learnable parameters and generate training data."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.net.modules():
            if isinstance(m, SelfAttention):
                m.rotation_params = torch.randn(m.embed_dim, m.embed_dim)
                m.entangle_params = torch.randn(m.embed_dim, m.embed_dim)
        for m in self.net.modules():
            if hasattr(m, "scale") and hasattr(m, "shift"):
                m.scale = torch.randn_like(m.scale)
                m.shift = torch.randn_like(m.shift)
        dataset = self.random_training_data(samples)
        target = self.net(torch.randn(self.arch[0], dtype=torch.float32))
        return self.arch, list(self.net.parameters()), dataset, target

    def random_training_data(self, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic regression data using the current network."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(self.arch[0], dtype=torch.float32)
            target = self.net(features)
            dataset.append((features, target))
        return dataset

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return layer‑wise activations for each sample."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for module in self.net.modules():
                current = module(current)
                activations.append(current)
            stored.append(activations)
        return stored

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "SelfAttention",
    "FraudLayerParameters",
    "_layer_from_params",
    "EstimatorNN",
    "GraphQNN",
]

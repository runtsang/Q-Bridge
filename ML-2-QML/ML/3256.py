"""FraudGraphHybrid: classical side of the hybrid fraud‑detection model.

The module defines a class `FraudGraphHybrid` that implements a photonic‑style neural network, utilities for generating synthetic data, a feed‑forward routine that records intermediate activations, and a graph construction routine based on state fidelities.  The design mirrors the quantum counterpart, enabling side‑by‑side experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Optional

import torch
from torch import nn
import networkx as nx
import itertools


# --------------------------------------------------------------------------- #
# Dataclass for photonic layer parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""

    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clamp values to the range [-bound, +bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> nn.Module:
    """Return a single photonic‑style linear block.

    The block implements the same mapping as the original quantum
    photonic circuit: a linear transformation, a tanh activation,
    and a normalisation/shift operation.
    """
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
            return outputs * self.scale + self.shift

    return Layer()


class FraudGraphHybrid:
    """Hybrid fraud‑detection model that mirrors a photonic circuit.

    The class exposes a classical neural network, synthetic data generation,
    a feed‑forward routine that captures intermediate activations, and a
    fidelity‑based adjacency graph construction.  It is designed to be
    used side‑by‑side with the quantum counterpart defined in the QML
    module.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.model = self.build_fraud_detection_program(input_params, layers)

    @staticmethod
    def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> nn.Sequential:
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    @staticmethod
    def random_training_data(
        weight: torch.Tensor,
        samples: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate synthetic training data for a linear target.

        Each sample is a pair (x, target) where target = weight @ x.
        """
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(
        self,
        samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """Run a batch of samples through the model and record every layer output."""
        activations: List[List[torch.Tensor]] = []
        for features, _ in samples:
            current = features
            layer_outputs: List[torch.Tensor] = [current]
            for layer in self.model:
                current = layer(current)
                layer_outputs.append(current)
            activations.append(layer_outputs)
        return activations

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Return the normalized squared dot‑product between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(
            enumerate(states), 2
        ):
            fid = FraudGraphHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "FraudGraphHybrid",
    "FraudLayerParameters",
]

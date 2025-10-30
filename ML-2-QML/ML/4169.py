"""Hybrid fraud‑detection model combining classical photonic layers,
graph‑based similarity, and a simple quantum classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn

# ----------------------------------------------------------------------
#  Classical photonic‑style layers
# ----------------------------------------------------------------------
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            return outputs * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))  # final regression head
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
#  Graph‑based similarity utilities
# ----------------------------------------------------------------------
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Iterable[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Return a list of edges (i, j, weight) for graph construction."""
    edges = []
    states = list(states)
    for i, state_i in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = state_fidelity(state_i, states[j])
            if fid >= threshold:
                edges.append((i, j, 1.0))
            elif secondary is not None and fid >= secondary:
                edges.append((i, j, secondary_weight))
    return edges

# ----------------------------------------------------------------------
#  Main hybrid classifier
# ----------------------------------------------------------------------
class FraudGraphQNNClassifier(nn.Module):
    """
    Combines a classical photonic network, a fidelity‑based adjacency
    graph, and a lightweight quantum classifier (see :mod:`qml`).
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: List[FraudLayerParameters],
        adjacency_threshold: float,
        secondary_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.network = build_fraud_detection_program(input_params, layer_params)
        self.adj_threshold = adjacency_threshold
        self.secondary_threshold = secondary_threshold

    def _extract_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the network and collect all intermediate activations."""
        activations = [x]
        current = x
        for layer in self.network[:-1]:
            current = layer(current)
            activations.append(current)
        return activations

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int, float]]]:
        """
        Returns:
            logits: output of the final linear layer
            graph_edges: list of edges from fidelity adjacency
        """
        activations = self._extract_activations(x)
        logits = self.network[-1](activations[-1])
        edges = fidelity_adjacency(
            activations,
            self.adj_threshold,
            secondary=self.secondary_threshold,
        )
        return logits, edges

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "state_fidelity",
    "fidelity_adjacency",
    "FraudGraphQNNClassifier",
]

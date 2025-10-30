"""FraudGraphHybrid – classical component.

This module implements a hybrid fraud‑detection architecture that
combines a learnable two‑layer perceptron with a graph‑based
regulariser built from pairwise state fidelities.  The architecture is
inspired by the photonic fraud‑detection seed and the graph‑based QNN
utility.  The `FraudGraphHybrid` class exposes a `forward` method that
returns the classical prediction, the intermediate state vectors
produced by the perceptron, and a `networkx.Graph` that captures the
topology of state similarities.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Shared dataclass
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a two‑mode photonic layer – used by both
    the classical and quantum implementations for interoperability.
    """
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
    """Constrain a scalar to the interval [-bound, bound]."""
    return max(-bound, min(value, bound))

def _layer_from_params(
    params: FraudLayerParameters,
    *, clip: bool = False,
) -> nn.Module:
    """Return a linear‑Tanh‑scale‑shift module that mimics the
    photonic layer defined by `params`.
    """
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2, bias=True)
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a PyTorch `Sequential` that mirrors the photonic circuit."""
    seq = [_layer_from_params(input_params, clip=False)]
    seq += [_layer_from_params(l, clip=True) for l in layers]
    seq.append(nn.Linear(2, 1))
    return nn.Sequential(*seq)

# --------------------------------------------------------------------------- #
# Graph utilities (inspired by GraphQNN)
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared cosine similarity between two state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted adjacency graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = state_fidelity(a, states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Synthetic data utilities
# --------------------------------------------------------------------------- #
def random_network(
    qnn_arch: Sequence[int],
    samples: int = 128,
) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random feed‑forward network and a training dataset
    that targets the final weight matrix.
    """
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
    target_weight = weights[-1]
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(qnn_arch[0], dtype=torch.float32)
        target = target_weight @ features
        dataset.append((features, target))
    return list(qnn_arch), weights, dataset, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Return the list of activations for each sample."""
    activations: List[List[torch.Tensor]] = []
    for features, _ in samples:
        current = features
        layer_acts = [features]
        for w in weights:
            current = torch.tanh(w @ current)
            layer_acts.append(current)
        activations.append(layer_acts)
    return activations

# --------------------------------------------------------------------------- #
# Main wrapper
# --------------------------------------------------------------------------- #
class FraudGraphHybrid(nn.Module):
    """Hybrid fraud‑detection model that couples a classical feed‑forward
    network to a graph based on pairwise state fidelities.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer of the photonic circuit.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers (all clipped).
    graph_threshold : float, optional
        Fidelity threshold to generate edges with weight 1.0.
    graph_secondary : float | None, optional
        Secondary threshold for softer edges.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        graph_threshold: float = 0.9,
        graph_secondary: float | None = None,
    ) -> None:
        super().__init__()
        self.clf = build_fraud_detection_program(input_params, layers)
        self.input_params = input_params
        self.layers = list(layers)
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], nx.Graph]:
        """Return (prediction, intermediate states, fidelity graph)."""
        # 1. Classical forward
        states: List[torch.Tensor] = [x]
        current = x
        for layer in self.clf:
            current = layer(current)
            states.append(current)
        pred = current.squeeze(-1)

        # 2. Build fidelity graph from intermediate states
        graph = fidelity_adjacency(
            states,
            self.graph_threshold,
            secondary=self.graph_secondary,
        )
        return pred, states, graph

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudGraphHybrid",
    "state_fidelity",
    "fidelity_adjacency",
    "random_network",
    "feedforward",
]

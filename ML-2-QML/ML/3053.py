"""GraphQNNHybrid: classical graph neural network with parameter clipping.

This module merges the graph‑based neural network utilities from
`GraphQNN.py` with the clipping strategy of the `FraudDetection.py`
seed.  The resulting `GraphQNNHybrid` class provides a compact interface
for generating random networks, running feed‑forward passes, and building
fidelity‑based adjacency graphs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

import torch
import networkx as nx

Tensor = torch.Tensor

# ----------------------------------------------------------------------
# Parameter handling – a lightweight analogue of FraudDetection
# ----------------------------------------------------------------------
@dataclass
class LayerParams:
    """Container for a 2‑by‑2 linear layer with optional clipping."""
    weight: Tuple[Tuple[float, float], Tuple[float, float]]
    bias: Tuple[float, float]
    scale: Tuple[float, float]
    shift: Tuple[float, float]

def _clip_tensor(tensor: Tensor, bound: float) -> Tensor:
    return tensor.clamp(-bound, bound)

def _layer_from_params(params: LayerParams, *, clip: bool) -> torch.nn.Module:
    weight = torch.tensor(list(params.weight), dtype=torch.float32)
    bias = torch.tensor(params.bias, dtype=torch.float32)
    if clip:
        weight = _clip_tensor(weight, 5.0)
        bias = _clip_tensor(bias, 5.0)
    linear = torch.nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = torch.nn.Tanh()
    scale = torch.tensor(params.scale, dtype=torch.float32)
    shift = torch.tensor(params.shift, dtype=torch.float32)

    class Layer(torch.nn.Module):
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

def _random_layer_params() -> LayerParams:
    weight = (tuple(torch.randn(2).tolist()), tuple(torch.randn(2).tolist()))
    bias = tuple(torch.randn(2).tolist())
    scale = tuple(torch.randn(2).tolist())
    shift = tuple(torch.randn(2).tolist())
    return LayerParams(weight=weight, bias=bias, scale=scale, shift=shift)

# ----------------------------------------------------------------------
# Core graph‑neural‑network helpers
# ----------------------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(arch: Sequence[int], samples: int, *, clip: bool = True) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    weights: List[Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        w = _random_linear(in_f, out_f)
        if clip:
            w = _clip_tensor(w, 5.0)
        weights.append(w)
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----------------------------------------------------------------------
# Convenience wrapper class
# ----------------------------------------------------------------------
class GraphQNNHybrid:
    """
    A simple wrapper that keeps a graph‑based neural network together
    with its architecture, weights and training data.  The class
    exposes the same API as the standalone functions, making it
    straightforward to swap between classical and quantum back‑ends.
    """

    def __init__(self, arch: Sequence[int], *, clip: bool = True) -> None:
        self.arch = list(arch)
        self.clip = clip
        self.weights, self.training_data, self.target_weight = self._create_network()

    def _create_network(self) -> Tuple[List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        _, weights, training_data, target = random_network(self.arch, 10, clip=self.clip)
        return weights, training_data, target

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(self.arch, self.weights, samples)

    def fidelity_graph(self, states: Sequence[Tensor], threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

__all__ = [
    "LayerParams",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "GraphQNNHybrid",
]

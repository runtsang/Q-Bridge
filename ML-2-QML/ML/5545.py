"""
GraphQNNGen448: classical utilities for graph neural networks.

The module extends the original GraphQNN.py with additional
building blocks taken from the fraud‑detection, convolution, and
classifier examples.  It remains fully classical (PyTorch,
NetworkX) and can be used as a drop‑in replacement while still
providing a clean API for the quantum side.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Any

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1.  Fraud‑detection style parameters and helper
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters that describe a single fraud‑detection layer."""
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
    # Construct a 2‑to‑2 linear layer followed by a tanh and a custom
    # scale/shift operation that mimics the photonic circuit.
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

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a PyTorch Sequential model that mimics the photonic
    fraud‑detection architecture."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 2.  Conv filter (classical quanvolution)
# --------------------------------------------------------------------------- #
def Conv() -> nn.Module:
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data: Any) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()


# --------------------------------------------------------------------------- #
# 3.  Classical graph neural network helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs (x, y) where y = weight @ x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random classical network and its training data."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return the activations per layer for each sample."""
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
    """Squared overlap of two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 4.  Classical classifier factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Create a simple feed‑forward classifier mirroring the quantum version."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# 5.  Unified GraphQNNGen448 class
# --------------------------------------------------------------------------- #
class GraphQNNGen448:
    """Hybrid classical‑quantum graph neural network helper.

    The class exposes the same public API as the original GraphQNN.py
    while adding convenience wrappers for fraud‑detection layers,
    a convolution filter, and a simple classifier.  All methods are
    pure Python/NumPy/PyTorch and do not depend on quantum back‑ends.
    """

    def __init__(self, arch: Sequence[int], device: str = "cpu") -> None:
        self.arch = list(arch)
        self.device = device

    # ---- Classical network utilities ------------------------------------
    def random_network(self, samples: int) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        return random_network(self.arch, samples)

    def feedforward(self, weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(self.arch, weights, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    # ---- Fraud‑detection helpers -----------------------------------------
    def build_fraud_layer(self, params: FraudLayerParameters, clip: bool = True) -> nn.Module:
        return _layer_from_params(params, clip=clip)

    def build_fraud_detection_program(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> nn.Sequential:
        return build_fraud_detection_program(input_params, layers)

    # ---- Convolution filter ---------------------------------------------
    def conv(self, data: Any) -> float:
        return Conv().run(data)

    # ---- Classifier factory ---------------------------------------------
    def build_classifier_circuit(
        self, num_features: int, depth: int
    ) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        return build_classifier_circuit(num_features, depth)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "Conv",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "build_classifier_circuit",
    "GraphQNNGen448",
]

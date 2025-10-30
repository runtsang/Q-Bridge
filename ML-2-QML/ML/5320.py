"""Hybrid Graph Neural Network for classical and quantum-inspired layers.

This module extends the original GraphQNN utilities by adding a
HybridGraphQNN class that can mix classical feed‑forward layers,
sampler networks, fraud‑detection inspired layers, and QCNN-inspired
convolutional networks.  The interface mirrors the original
implementation but supports random network generation, training data
creation, and fidelity‑based adjacency graph construction for both
classical and quantum regimes.
"""

import itertools
import random
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --- Random data helpers -------------------------------------------------------

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
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

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- Layer implementations -------------------------------------------------------

class SamplerModule(nn.Module):
    """Simple feed‑forward sampler network used in hybrid graph."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class FraudLayerParameters:
    """Container for parameters of a fraud detection layer."""
    def __init__(self,
                 bs_theta: float,
                 bs_phi: float,
                 phases: Tuple[float, float],
                 squeeze_r: Tuple[float, float],
                 squeeze_phi: Tuple[float, float],
                 displacement_r: Tuple[float, float],
                 displacement_phi: Tuple[float, float],
                 kerr: Tuple[float, float]):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _random_fraud_params() -> FraudLayerParameters:
    """Generate random fraud layer parameters bounded to mimic photonic constraints."""
    def rand(): return random.uniform(-5.0, 5.0)
    return FraudLayerParameters(
        bs_theta=rand(), bs_phi=rand(),
        phases=(rand(), rand()),
        squeeze_r=(rand(), rand()),
        squeeze_phi=(rand(), rand()),
        displacement_r=(rand(), rand()),
        displacement_phi=(rand(), rand()),
        kerr=(rand(), rand())
    )

class FraudLayer(nn.Module):
    """Classical implementation of a fraud detection layer with linear + activation."""
    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        self.shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        outputs = self.activation(self.linear(inputs))
        outputs = outputs * self.scale + self.shift
        return outputs

class QCNNModel(nn.Module):
    """Convolution‑inspired classical network emulating QCNN steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --- Hybrid network ---------------------------------------------------------------

class HybridGraphQNN:
    """Hybrid graph neural network that stitches together classical and quantum‑inspired layers."""
    def __init__(self,
                 architecture: Sequence[int],
                 layer_types: Sequence[str],
                 seed: int | None = None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        self.architecture = list(architecture)
        self.layer_types = list(layer_types)
        if len(self.layer_types)!= len(self.architecture) - 1:
            raise ValueError("layer_types length must be architecture length minus one")
        self.layers: List[nn.Module] = []
        self._build_layers()

    def _build_layers(self) -> None:
        for idx, ltype in enumerate(self.layer_types):
            in_f, out_f = self.architecture[idx], self.architecture[idx+1]
            if ltype == "feedforward":
                linear = nn.Linear(in_f, out_f)
                self.layers.append(nn.Sequential(linear, nn.Tanh()))
            elif ltype == "sampler":
                self.layers.append(SamplerModule())
            elif ltype == "fraud":
                params = _random_fraud_params()
                self.layers.append(FraudLayer(params))
            elif ltype == "qcnn":
                self.layers.append(QCNNModel())
                if in_f!= 8:
                    self.layers.insert(-1, nn.Sequential(nn.Linear(in_f, 8), nn.Tanh()))
            else:
                raise ValueError(f"Unsupported layer type: {ltype}")

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.architecture[:-1], self.architecture[1:]):
            weights.append(_random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = random_training_data(target_weight, samples)
        return list(self.architecture), weights, training_data, target_weight

    def feedforward(self,
                    samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for layer in self.layers:
                current = layer(current)
                activations.append(current)
            stored.append(activations)
        return stored

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(self,
                           states: Sequence[Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                   secondary_weight=secondary_weight)

__all__ = [
    "HybridGraphQNN",
    "SamplerModule",
    "FraudLayerParameters",
    "FraudLayer",
    "QCNNModel",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]

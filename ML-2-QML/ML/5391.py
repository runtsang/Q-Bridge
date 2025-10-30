"""Combined sampler network with fraud detection layers and graph utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Any

# ------------------------------------------------------------------
# Fraud‑detection inspired parameter container
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Helper to build a single fraud‑layer
# ------------------------------------------------------------------
def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

# ------------------------------------------------------------------
# Main hybrid sampler network
# ------------------------------------------------------------------
class SamplerQNNGen282(nn.Module):
    """
    A hybrid sampler network that:
      * starts with a fraud‑detection inspired 2‑to‑2 linear layer
      * continues with a user‑defined feed‑forward architecture
      * can return a fidelity‑based adjacency graph of intermediate activations
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        fraud_params: FraudLayerParameters,
        graph_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        self.fraud_layer = _layer_from_params(fraud_params, clip=False)

        # Build feed‑forward layers
        layers: List[nn.Module] = []
        in_features = qnn_arch[0]
        for out_features in qnn_arch[1:]:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Tanh())
            in_features = out_features
        self.feedforward = nn.Sequential(*layers)
        self.graph_threshold = graph_threshold

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fraud_layer(inputs)
        activations = [x]
        for layer in self.feedforward:
            x = layer(x)
            activations.append(x)
        return activations[-1]

    def compute_fidelity_adjacency(self, states: torch.Tensor) -> nx.Graph:
        """
        Build a weighted graph from cosine similarity of state vectors.
        """
        def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
            return float(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12))

        graph = nx.Graph()
        graph.add_nodes_from(range(states.shape[0]))
        for i in range(states.shape[0]):
            for j in range(i + 1, states.shape[0]):
                fid = cosine_sim(states[i], states[j])
                if fid >= self.graph_threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif fid >= self.graph_threshold * 0.8:
                    graph.add_edge(i, j, weight=0.5)
        return graph

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Generate a random network architecture, weights, and synthetic training data
        similar to GraphQNN.random_network but for this hybrid model.
        """
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f))
        target_weight = weights[-1]
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.shape[1])
            target = target_weight @ features
            dataset.append((features, target))
        return list(qnn_arch), weights, dataset, target_weight

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility from QuantumRegression to generate noisy sine‑cosine labels.
        """
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

__all__ = ["SamplerQNNGen282", "FraudLayerParameters", "_layer_from_params"]

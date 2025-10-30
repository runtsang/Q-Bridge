"""Hybrid classical GraphQNN module.

Provides a neural network class that can be trained with PyTorch optimizers.
Also includes utilities for generating random networks, fidelity computation,
and graph-based adjacency construction for state clustering.
"""

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility helpers – unchanged from the seed
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
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

# --------------------------------------------------------------------------- #
# Hybrid variational QNN – new
# --------------------------------------------------------------------------- #
class GraphQNNGen318(nn.Module):
    """A minimal variational graph‑QNN that maps input features to a quantum state.

    Parameters
    ----------
    * qnn_arch : sequence of layer sizes.
      Each entry defines the number of qubits in the corresponding layer.
      The final layer is the output qubit‑register.
    * num_layers : number of variational blocks per layer.
    * noise_level : (float) noise level added to each rotation angle.
    """

    def __init__(self, qnn_arch: Sequence[int], num_layers: int = 1, noise_level: float = 0.0):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.num_layers = num_layers
        self.noise_level = noise_level
        # Classical linear layers
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: Tensor) -> Tensor:
        """Classical feed‑forward through linear layers."""
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    def loss(self, inputs: Tensor, targets: Tensor, threshold: float = 0.9) -> Tensor:
        """Graph‑regularised mean‑squared‑error loss."""
        outputs = self.forward(inputs)
        base_loss = F.mse_loss(outputs, targets)
        # Build adjacency from pairwise fidelities
        n = outputs.shape[0]
        adj = torch.zeros((n, n), device=outputs.device)
        for i in range(n):
            for j in range(i + 1, n):
                fid = state_fidelity(outputs[i], outputs[j])
                if fid >= threshold:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        penalty = torch.sum(adj) / (n * (n - 1))
        return base_loss + 0.1 * penalty

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNGen318",
]

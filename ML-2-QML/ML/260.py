import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import Iterable, List, Sequence, Tuple

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (feature, target) pairs for supervised learning."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create random weights and training data for a linear network."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target = weights[-1]
    training_data = random_training_data(target, samples)
    return list(qnn_arch), weights, training_data, target

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Standard feed‑forward using tanh activations."""
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
    """Squared overlap of two vectors."""
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
    """Build a weighted graph from state fidelities."""
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
#   Extension: Learned Layer Weights
# --------------------------------------------------------------------------- #

class _LayerWeightLearner(nn.Module):
    """Small MLP that predicts a scalar weight for a layer."""
    def __init__(self, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(out_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, activation: Tensor) -> Tensor:
        return self.net(activation)

def random_network_with_learned_weights(qnn_arch: Sequence[int], samples: int):
    """
    Generate a network where each layer has an associated weight learner.
    Returns (arch, weights, weight_modules, training_data, target).
    """
    weights: List[Tensor] = []
    weight_modules: List[_LayerWeightLearner] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
        weight_modules.append(_LayerWeightLearner(out_f))
    target = weights[-1]
    training_data = random_training_data(target, samples)
    return list(qnn_arch), weights, weight_modules, training_data, target

def feedforward_with_weights(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    weight_modules: Sequence[_LayerWeightLearner],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """
    Feed‑forward that multiplies each layer's activation by a learned scalar.
    """
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight, learner in zip(weights, weight_modules):
            current = torch.tanh(weight @ current)
            scalar = learner(current).item()
            current = current * scalar
            activations.append(current)
        stored.append(activations)
    return stored

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "feedforward_with_weights",
    "random_network_with_learned_weights",
]

"""Hybrid graph neural network training pipeline using PyTorch.

This module keeps the original GraphQNN utilities but adds a
classical GNN that trains on the target weight’s action on random
states.  It also provides a simple training routine that returns the
learned weight matrix, the fidelity graph, and the trained model.
"""

import itertools
from typing import List, Tuple, Iterable, Sequence

import torch
from torch import nn
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=False)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate pairs (input, target) where target = weight @ input."""
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and training data."""
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
    """Return the activation sequence for each sample."""
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
    """Squared overlap between two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class SimpleGNN(nn.Module):
    """Feed‑forward network matching the architecture of qnn_arch."""
    def __init__(self, qnn_arch: Sequence[int]):
        super().__init__()
        layers = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            layers.append(nn.Linear(in_f, out_f, bias=False))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_graph_qnn(
    qnn_arch: Sequence[int],
    samples: int,
    epochs: int = 200,
    lr: float = 1e-3,
    fidelity_threshold: float = 0.95,
    secondary_threshold: float | None = None,
    device: str | torch.device = "cpu",
) -> Tuple[Tensor, nx.Graph, SimpleGNN]:
    """
    Train a SimpleGNN to approximate the target weight matrix.

    Returns:
        final_weight: The weight matrix of the last layer.
        graph: Fidelity graph built from final activations.
        model: The trained SimpleGNN instance.
    """
    arch, init_weights, training_data, target_weight = random_network(qnn_arch, samples)
    model = SimpleGNN(arch).to(device)

    # Initialise model weights with the same random values
    with torch.no_grad():
        for layer, w in zip(model.net, init_weights):
            if isinstance(layer, nn.Linear):
                layer.weight.copy_(w.T.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = 0.0
        for inp, tgt in training_data:
            inp = inp.to(device)
            tgt = tgt.to(device)
            out = model(inp)
            loss += criterion(out, tgt)
        loss /= len(training_data)
        loss.backward()
        optimizer.step()

    # Extract final weight matrix
    final_weight = next(model.net[-2].weight.data).T.clone().cpu()

    # Compute fidelity graph from final layer activations
    sample_inputs = torch.randn(samples, target_weight.size(1), dtype=torch.float32, device=device)
    activations = [sample_inputs]
    current = sample_inputs
    for layer in model.net:
        current = layer(current)
        activations.append(current.detach().cpu())
    final_states = activations[-1]
    graph = fidelity_adjacency(final_states, fidelity_threshold, secondary=secondary_threshold)

    return final_weight, graph, model


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_graph_qnn",
    "SimpleGNN",
]

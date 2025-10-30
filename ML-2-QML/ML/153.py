"""Hybrid classical Graph Neural Network module with extended functionality.

This module expands the original seed by adding:
1. A *feature‑to‑state* mapping that injects a learnable linear layer before each quantum layer.
2. A simple variational optimizer (torch.optim) that trains the linear weights to minimise the mean‑squared error between
   the predicted quantum state and the known target unitary.
3. A `sparsity_analysis` helper that builds a graph of layer‑wise activation sparsity and returns
   the number of edges that exceed a given sparsity threshold.

The public API remains compatible with the original seed – all previous helper functions are exported – and
the new functions can be imported without breaking existing code.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a list of (feature, target) pairs for the given linear weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network that mimics a quantum circuit with many layers.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture: list of hidden sizes per layer (the number of qubits per layer).
    samples : int
        The number of training samples used to generate the target unitary.

    Returns
    -------
    Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]
        (architecture, weights, training_data, target_weight)
    """
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass through the classical network, returning activations per layer."""
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
    """Compute the squared overlap between two classical vectors."""
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
    """Construct a weighted graph from state fidelities with optional secondary edge weights."""
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
# NEW FEATURES: hybrid training and sparsity analysis
# --------------------------------------------------------------------------- #

def _feature_to_state(features: Tensor, linear_weight: Tensor) -> Tensor:
    """Map a feature vector to a quantum‑style state vector (normalised)."""
    normalized = features / (torch.norm(features) + 1e-12)
    return torch.tanh(linear_weight @ normalized)


def _train_linear_weights(
    *,
    input_dim: int,
    target_weight: Tensor,
    training_data: List[Tuple[Tensor, Tensor]],
    lr: float = 0.01,
    epochs: int = 200,
    device: str = "cpu",
    weight_decay: float = 0.0,
) -> Tensor:
    """Train a linear mapping that approximates the target weight matrix.

    The loss is the mean‑squared error between the target linear transformation
    and the learned linear transformation applied to the same inputs.
    """
    device = torch.device(device)
    weight = torch.randn(target_weight.size(0), input_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([weight], lr=lr, weight_decay=weight_decay)
    target = target_weight.to(device)

    for _ in range(epochs):
        epoch_loss = 0.0
        for features, _ in training_data:
            features = features.to(device)
            pred = _feature_to_state(features, weight)
            true = _feature_to_state(features, target)
            loss = torch.mean((pred - true) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch_loss / len(training_data) < 1e-6:
            break
    return weight.detach().cpu()


def hybrid_train(
    qnn_arch: Sequence[int],
    training_data: List[Tuple[Tensor, Tensor]],
    target_weight: Tensor,
    lr: float = 0.01,
    epochs: int = 200,
    device: str = "cpu",
) -> List[Tensor]:
    """Train a hybrid network that inserts a learnable linear mapping before the quantum layer.

    The returned list contains the trained weights for each layer of the classical network.
    """
    input_dim = qnn_arch[-2]
    trained_weight = _train_linear_weights(
        input_dim=input_dim,
        target_weight=target_weight,
        training_data=training_data,
        lr=lr,
        epochs=epochs,
        device=device,
    )
    weights: List[Tensor] = []
    for i in range(len(qnn_arch) - 1):
        if i == len(qnn_arch) - 2:
            weights.append(trained_weight)
        else:
            weights.append(_random_linear(qnn_arch[i], qnn_arch[i + 1]))
    return weights


def sparsity_analysis(
    activations: List[List[Tensor]],
    threshold: float = 0.1,
) -> nx.Graph:
    """Build a graph where nodes are layers and edges indicate similarity of activation sparsity.

    The sparsity of a layer is defined as the fraction of elements below ``threshold``.
    """
    layer_sparsity: List[float] = []
    for layer_index in range(len(activations[0])):  # assume all samples have same depth
        values = torch.stack([sample[layer_index] for sample in activations], dim=0)
        sparsity_per_sample = (values.abs() < threshold).float().mean(dim=1).mean().item()
        layer_sparsity.append(sparsity_per_sample)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(layer_sparsity)))
    for i, s_i in enumerate(layer_sparsity):
        for j, s_j in enumerate(layer_sparsity):
            if i >= j:
                continue
            weight = 1.0 - abs(s_i - s_j)  # higher weight for similar sparsity
            graph.add_edge(i, j, weight=weight)
    return graph


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "hybrid_train",
    "sparsity_analysis",
    "_feature_to_state",
    "_train_linear_weights",
]

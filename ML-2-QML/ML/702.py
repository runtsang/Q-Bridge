"""Hybrid classical Graph Neural Network with gradient‑based training and automatic differentiation.

This module extends the original GraphQNN seed by adding full back‑propagation support.
The ``train`` function now accepts an ``optimizer`` (e.g. torch.optim.Adam) and
returns a dictionary with loss and fidelity histories.  The ``forward`` method
uses a torch ``nn.Module`` that stores the linear layers as ``nn.Linear`` objects,
allowing PyTorch autograd to compute gradients.  The module also exposes
``sample_fidelity`` and ``plot_fidelity`` helpers for quick diagnostics.
"""

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict, Any

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> nn.Linear:
    """Return a linear layer with random weights and bias."""
    return nn.Linear(in_features, out_features, bias=True)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate dataset of feature‑vector pairs (x, y) that target the
    ``weight`` matrix.  The dataset is kept small for simplicity.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random linear network and a training set for the last layer."""
    weights: List[nn.Linear] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1].weight.data
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[nn.Linear],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass for every sample and return every activation."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight(current))
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two pure states."""
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
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN__gen369(nn.Module):
    """Hybrid classical graph neural network with learnable linear layers.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[3, 5, 2]``.
    weights : Sequence[nn.Linear] | None, optional
        Pre‑initialized layers.  If ``None`` random layers are created.
    """

    def __init__(
        self,
        arch: Sequence[int],
        weights: Sequence[nn.Linear] | None = None,
    ):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList()
        if weights is None:
            for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
                self.layers.append(nn.Linear(in_f, out_f, bias=True))
        else:
            assert len(weights) == len(self.arch) - 1, "Weight list length mismatch."
            for layer in weights:
                self.layers.append(layer)

    def forward(self, x: Tensor) -> List[Tensor]:
        activations = [x]
        current = x
        for layer in self.layers:
            current = torch.tanh(layer(current))
            activations.append(current)
        return activations

    @staticmethod
    def train(
        model: "GraphQNN__gen369",
        training_data: List[Tuple[Tensor, Tensor]],
        optimizer: optim.Optimizer,
        epochs: int = 200,
        loss_fn: nn.Module = nn.MSELoss(),
        device: torch.device | None = None,
    ) -> Dict[str, List[float]]:
        """Train the network with the supplied optimizer.

        Returns a history dictionary with keys ``loss`` and ``fidelity``.
        """
        if device is None:
            device = torch.device("cpu")
        model.to(device)
        history = {"loss": [], "fidelity": []}
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_fid = 0.0
            for x, y in training_data:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                activations = model(x)
                output = activations[-1]
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_fid += state_fidelity(output.detach().cpu(), y.cpu())
            epoch_loss /= len(training_data)
            epoch_fid /= len(training_data)
            history["loss"].append(epoch_loss)
            history["fidelity"].append(epoch_fid)
        return history

    @staticmethod
    def plot_fidelity(fidelity_history: List[float]) -> None:
        """Plot fidelity over training epochs."""
        plt.figure(figsize=(6, 4))
        plt.plot(fidelity_history, label="Fidelity")
        plt.xlabel("Epoch")
        plt.ylabel("Fidelity")
        plt.title("Training Fidelity")
        plt.legend()
        plt.tight_layout()
        plt.show()


__all__ = [
    "GraphQNN__gen369",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]

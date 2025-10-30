"""GraphQNNGen116 – classical implementation with optional FCL/Conv utilities.

This module extends the original GraphQNN to support both fully‑connected and
convolutional drop‑in replacements, while preserving the fidelity‑based graph
utilities from the seed.  The public API mirrors the quantum counterpart so
that the same class name can be swapped between back‑ends.
"""
from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


class FCL(nn.Module):
    """Simple fully‑connected layer with a single output."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0).detach().numpy()


class ConvFilter(nn.Module):
    """2‑D convolutional filter that emulates a quanvolution layer."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class GraphQNNGen116(nn.Module):
    """Hybrid graph neural network that can operate in classical or quantum mode.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes.  For a classical network each entry is the
        dimensionality of the linear layer input.  For a quantum
        network it represents the number of qubits per layer.
    mode : {"classical", "quantum"}
        Determines which back‑end to use.
    device : str, optional
        Device for torch tensors (default: "cpu").
    """

    def __init__(
        self,
        arch: Sequence[int],
        mode: str = "classical",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.arch = arch
        self.mode = mode
        self.device = device

        if mode == "classical":
            self.layers: nn.ModuleList = nn.ModuleList()
            for in_f, out_f in zip(arch[:-1], arch[1:]):
                self.layers.append(nn.Linear(in_f, out_f))
            self.to(device)
        else:
            # Quantum mode – the actual quantum implementation lives in the
            # separate qml module.  Here we only keep a placeholder.
            self.layers = None

    def forward(self, x: Tensor) -> Tensor:
        """Return the network output for a single sample."""
        if self.mode!= "classical":
            raise RuntimeError("Quantum forward not implemented in the classical module.")
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    def run(self, data: np.ndarray) -> np.ndarray:
        """Convenience wrapper that accepts a numpy array."""
        x = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        return self.forward(x).detach().cpu().numpy()

    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random linear network and a training set for its final layer."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(_random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNNGen116.random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Create synthetic data for supervised learning."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run the network on a batch of samples and return activations per layer."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for layer in self.layers:
                current = torch.tanh(layer(current))
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Cosine‑like fidelity for classical vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
            fid = GraphQNNGen116.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "GraphQNNGen116",
    "FCL",
    "ConvFilter",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]

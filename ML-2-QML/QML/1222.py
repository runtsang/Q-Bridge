"""GraphQNN hybrid module – quantum side.

This module keeps the original helper functions and adds a
variational quantum circuit that consumes classical node embeddings
as parameters.  The circuit outputs a probability distribution
which can be used for classification tasks.  The class is named
GraphQNN__gen165 and follows the same public API as the classical
counterpart.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane.optimize import GradientDescentOptimizer
import torch

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Original helper functions – unchanged for backward compatibility
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

# --------------------------------------------------------------------------- #
#  Variational quantum circuit – new functionality
# --------------------------------------------------------------------------- #
class GraphQNN__gen165:
    """
    Variational quantum circuit that accepts classical node embeddings
    as parameters.  The circuit is implemented with PennyLane and
    returns a probability distribution over the computational basis.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        device: str = "default.qubit",
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_qubits)
        # Initialise random parameters
        self.params = np.random.randn(num_layers, num_qubits, 3)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, *params, embeddings):
        """
        Simple encoding: each embedding value is rotated around Y.
        The variational layers consist of RY(θ) gates followed by a
        chain of CNOTs.  The output is the probability vector.
        """
        for i, val in enumerate(embeddings):
            qml.RY(val, wires=i)
        for layer in range(self.num_layers):
            for q in range(self.num_qubits):
                qml.Rot(*params[layer, q], wires=q)
            for q in range(self.num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.probs(wires=range(self.num_qubits))

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Run the circuit on the mean embedding of a batch of nodes.
        """
        emb_mean = embeddings.mean(dim=0).numpy()
        probs = self.qnode(self.params, embeddings=emb_mean)
        return torch.tensor(probs, dtype=torch.float32)

    def train(
        self,
        dataset: List[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 0.01,
    ):
        """
        Simple training loop that optimises the circuit parameters
        to minimise a cross‑entropy loss between the circuit output
        and the target labels.
        """
        opt = GradientDescentOptimizer(lr)
        for epoch in range(epochs):
            for embeddings, labels in dataset:
                def loss_fn(p):
                    probs = self.qnode(p, embeddings=embeddings.mean(dim=0).numpy())
                    probs = np.clip(probs, 1e-12, 1.0)
                    return -np.sum(labels.numpy() * np.log(probs))
                self.params = opt.step(loss_fn, self.params)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen165",
]

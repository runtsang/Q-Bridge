"""GraphQNN__gen244 - Quantum implementation with Pennylane.

This module mirrors the classical API but implements the feed‑forward
using a variational circuit on a quantum device.  It supports
random network generation, synthetic training data, fidelity
adjacency, and a simple gradient‑based training loop.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch

Tensor = torch.Tensor


def _random_unitary(num_qubits: int, seed: int | None = None) -> np.ndarray:
    """Return a random unitary matrix for `num_qubits` qubits."""
    rng = np.random.default_rng(seed)
    dim = 2 ** num_qubits
    random_matrix = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(random_matrix)
    return q


def random_training_data(unitary: np.ndarray, samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate pairs (psi, U psi) with random pure states."""
    dataset: list[tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        vec = np.random.standard_normal(dim) + 1j * np.random.standard_normal(dim)
        vec /= np.linalg.norm(vec)
        dataset.append((vec, unitary @ vec))
    return dataset


def random_network(qnn_arch: list[int], samples: int, seed: int | None = None) -> tuple[list[int], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Create a random parametrised variational circuit and training data."""
    params: list[np.ndarray] = []
    for layer in range(1, len(qnn_arch)):
        layer_params = np.random.uniform(0, 2 * np.pi, size=(qnn_arch[layer],))
        params.append(layer_params)
    target_unitary = _random_unitary(qnn_arch[-1], seed)
    training_data = random_training_data(target_unitary, samples)
    return qnn_arch, params, training_data, target_unitary


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure state vectors."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def _make_qnode(num_qubits: int):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(params, input_state):
        qml.StatePrep(input_state, wires=range(num_qubits))
        for theta in params:
            qml.RX(theta, wires=range(num_qubits))
        return qml.state()

    return circuit


def feedforward(qnn_arch: Sequence[int], params: Sequence[np.ndarray], samples: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[list[np.ndarray]]:
    """Apply the variational circuit to each input state."""
    stored: list[list[np.ndarray]] = []
    circuit = _make_qnode(qnn_arch[-1])
    for inp, _ in samples:
        layerwise = [inp]
        current = inp
        for theta in params:
            current = circuit([theta], current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


@dataclass
class GraphQNN__gen244:
    """Hybrid quantum‑classical graph neural network base."""
    arch: Sequence[int]
    params: list[np.ndarray] | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        self.device = self.device or "default.qubit"
        if self.params is None:
            self.params = np.random.uniform(0, 2 * np.pi, size=(self.arch[-1],))
        self.circuit = _make_qnode(self.arch[-1])

    def forward(self, input_state: np.ndarray) -> np.ndarray:
        """Apply the full circuit to a single state."""
        return self.circuit(self.params, input_state)

    def train_step(self, data: list[tuple[np.ndarray, np.ndarray]], lr: float = 0.01) -> float:
        """One gradient descent step on the circuit parameters."""
        loss_fn = lambda pred, target: 1 - state_fidelity(pred, target)
        grads = np.zeros_like(self.params)
        loss = 0.0
        for inp, tgt in data:
            pred = self.forward(inp)
            loss += loss_fn(pred, tgt)
            grad_fn = qml.grad(lambda p: 1 - state_fidelity(self.circuit(p, inp), tgt))
            grads += grad_fn(self.params)
        loss /= len(data)
        grads /= len(data)
        self.params -= lr * grads
        return float(loss)

    def train(self, data: list[tuple[np.ndarray, np.ndarray]], epochs: int = 10, lr: float = 0.01) -> list[float]:
        losses: list[float] = []
        for _ in range(epochs):
            loss = self.train_step(data, lr)
            losses.append(loss)
        return losses


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen244",
]
